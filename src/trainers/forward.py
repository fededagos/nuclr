from typing import Optional
import json
import math
from contextlib import nullcontext
from omegaconf import OmegaConf

from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from torch_brain.data import Dataset
from torch_brain.data.sampler import SequentialFixedWindowSampler
from temporaldata import Interval

from src.models import BaseNeuronEncoder
from src.dataset import ViewData, get_window
from torch_brain.data.dataset import DatasetIndex
from src.allgcl import split_selection
from src.samplers import (
    DistributedSamplerWrapper,
)
from src.utils import Precision
from src.evaluators.collector import EmbeddingCollector
from src.trainers.trainer import Trainer, CheckpointDict
from src.utils import instantiate, expand_path


class ForwardPass(Trainer):
    """Export neuron embeddings, with split and block scopes for All-GCL.

    Existing configs retain the original train-dataset .pt output path.
    """

    def setup(self, ckpt: Optional[CheckpointDict]):
        if ckpt is None:
            raise ValueError("Must provide ckpt")

        self.scoped = bool(self.cfg.get("scopes"))
        if self.scoped:
            if self.world_size != 1:
                raise ValueError("Scoped forward currently requires one process/GPU")
            unknown = set(self.cfg.scopes) - {"all", "condition", "stimulus"}
            if unknown:
                raise ValueError(f"Unknown scopes: {unknown}")
            self._setup_model(ckpt)
            self.forward_epoch = ckpt["epoch"]
            self.model_config = OmegaConf.to_container(
                ckpt["cfg"]["model"], resolve=True
            )
            self.dataset = Dataset(
                root=self.cfg.data_root,
                config=split_selection(self.cfg.data, self.cfg.splits),
                unit_id_prefix_fn=lambda _: "",
                session_id_prefix_fn=lambda _: "",
            )
            return

        embs_dir = expand_path(self.cfg.embs_root) / ckpt["run_id"]
        embs_dir.mkdir(exist_ok=True, parents=True)
        self.embs_save_filepath: Path = (
            embs_dir / f"embs_{self.cfg.ckpt.load_from.name}"
        )
        self.logger.info(f"Will save embeddings at {self.embs_save_filepath}")
        if self.embs_save_filepath.exists():
            if self.cfg.overwrite:
                self.logger.warn(f"Embeddings exist, will overwrite")
            else:
                raise ValueError(
                    f"Embeddings exist at that location."
                    f" Use `overwrite=true` if you want to overwrite them."
                )

        self._setup_model(ckpt)
        self._setup_loader()
        self._setup_embedding_collector()

        self.model = self.make_ddp(self.model)  # type: ignore

    def _setup_model(self, ckpt):
        self.precision = Precision(self.cfg.precision)
        self.model: BaseNeuronEncoder = instantiate(
            ckpt["cfg"]["model"],
            precision=self.precision,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.eval().to(self.device)

    def train(self):
        if self.scoped:
            self._forward_scopes()
            return
        self.val_epoch()
        unit_emb, unit_ids = self.emb_collector.compute()
        if self.rank == 0:
            save_data = {
                "embs": unit_emb,
                "ids": unit_ids,
            }
            torch.save(save_data, self.embs_save_filepath)
            self.logger.info(f"Saved to {self.embs_save_filepath}")

    @torch.inference_mode()
    def val_epoch(self):
        if self.cfg.debug == "train":
            return

        self.emb_collector.reset()

        for batch in self.logger.get_pbar(self.loader, "Forward Pass"):
            assert isinstance(batch, ViewData)
            batch.to(self.device)

            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                y1 = self.model(**batch.data["enc_input"])
                self.emb_collector.update(y1, batch.unit_ids)

    def _setup_loader(self):
        ds = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.data.train_dataset,
            unit_id_prefix_fn=lambda _: "",
        )
        ds.transform = self.model.tokenize  # type: ignore
        collate_fn = ViewData.collate

        sampling_intervals = ds.get_sampling_intervals()

        if self.cfg.limit_observation_time is not None:
            self.logger.info(
                f"Limiting observation time to {self.cfg.limit_observation_time}s"
            )
            for sid, samp_interval in sampling_intervals.items():
                start = samp_interval.start[0]
                orig_duration = (samp_interval.end - samp_interval.start).sum()

                window = Interval(start, start + self.cfg.limit_observation_time)
                new_interval = samp_interval & window
                new_duration = (new_interval.end - new_interval.start).sum()
                sampling_intervals[sid] = new_interval

                self.logger.info(f"{sid}: {orig_duration} -> {new_duration}")

        sampler = DistributedSamplerWrapper(
            SequentialFixedWindowSampler(
                sampling_intervals=sampling_intervals,
                window_length=self.model.ctx_duration,
                step=None,
                drop_short=True,
            )
        )

        multi_wrkrs = self.cfg.num_workers > 0
        self.loader = DataLoader(
            dataset=ds,
            sampler=sampler,
            collate_fn=collate_fn,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            multiprocessing_context=mp.get_context("fork") if multi_wrkrs else None,
            persistent_workers=multi_wrkrs,
        )

        self.unit_ids = np.sort(ds.get_unit_ids())

        self.logger.info(f"Num sessions: {len(ds.get_session_ids())}")
        self.logger.info(f"Num units: {len(self.unit_ids)}")
        self.logger.info(f"Val steps/epoch: {len(self.loader)}")

    def _setup_embedding_collector(self):
        self.emb_collector = EmbeddingCollector(
            self.unit_ids,
            dim=self.model.emb_dim,
            device=self.device,
        )

    @torch.inference_mode()
    def _forward_scopes(self):
        """Encode each non-overlapping block window once, then average by scope."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        output = Path(self.cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "embeddings.parquet.tmp"
        schema = pa.schema(
            [
                ("unit_id", pa.string()),
                ("scope", pa.string()),
                ("n_windows", pa.int64()),
                ("emb", pa.list_(pa.float32(), self.model.emb_dim)),
            ]
        )
        with pq.ParquetWriter(temporary, schema) as writer:
            for rec in self.dataset.recording_dict:
                data = self.dataset._get_data_object(rec)
                ids = np.asarray(data.units.id).astype(str)
                unit_index = {uid: i for i, uid in enumerate(ids)}
                accumulators = {}
                for j, (start, end) in enumerate(
                    zip(data.blocks.start, data.blocks.end)
                ):
                    scopes = []
                    if "all" in self.cfg.scopes:
                        scopes.append("all")
                    for kind in ("condition", "stimulus"):
                        if kind in self.cfg.scopes:
                            label = np.asarray(getattr(data.blocks, kind)).astype(str)[
                                j
                            ]
                            scopes.append(f"{kind}:{label}")
                    for scope in scopes:
                        accumulators.setdefault(
                            scope,
                            [
                                np.zeros(
                                    (len(ids), self.model.emb_dim), dtype=np.float64
                                ),
                                np.zeros(len(ids), dtype=np.int64),
                            ],
                        )
                    duration = self.model.ctx_duration
                    count = math.floor((end - start + 1e-8) / duration)
                    for k in range(count):
                        index = DatasetIndex(
                            rec,
                            float(start + k * duration),
                            float(start + (k + 1) * duration),
                        )
                        view = self.model.tokenize(get_window(self.dataset, index))
                        if not len(view.unit_ids):
                            continue
                        positions = np.array([unit_index[uid] for uid in view.unit_ids])
                        batch = ViewData.collate([view]).to(self.device)
                        context = (
                            torch.autocast("cuda", dtype=self.precision.dtype)
                            if self.device.type == "cuda"
                            and self.precision.dtype != torch.float32
                            else nullcontext()
                        )
                        with context:
                            values = (
                                self.model(**batch.data["enc_input"])
                                .float()
                                .cpu()
                                .numpy()
                            )
                        if not np.isfinite(values).all():
                            raise ValueError(
                                f"Non-finite embedding in {rec} at {index.start}"
                            )
                        for scope in scopes:
                            accumulators[scope][0][positions] += values
                            accumulators[scope][1][positions] += 1
                for scope, (total, counts) in accumulators.items():
                    observed = counts > 0
                    if not observed.any():
                        continue
                    values = (total[observed] / counts[observed, None]).astype(
                        np.float32
                    )
                    observed_ids = ids[observed]
                    emb = pa.FixedSizeListArray.from_arrays(
                        pa.array(values.ravel(), type=pa.float32()), self.model.emb_dim
                    )
                    writer.write_table(
                        pa.Table.from_arrays(
                            [
                                pa.array(observed_ids),
                                pa.array([scope] * len(observed_ids)),
                                pa.array(counts[observed], type=pa.int64()),
                                emb,
                            ],
                            schema=schema,
                        )
                    )
        temporary.replace(output / "embeddings.parquet")
        metadata = {
            "ckpt": str(Path(self.cfg.ckpt.load_from).resolve()),
            "epoch": self.forward_epoch,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "model_config": self.model_config,
        }
        meta_path = output / "embeddings_meta.json"
        temporary_meta = meta_path.with_suffix(".json.tmp")
        temporary_meta.write_text(json.dumps(metadata, default=str, indent=2))
        temporary_meta.replace(meta_path)
