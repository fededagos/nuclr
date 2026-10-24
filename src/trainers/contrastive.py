from collections import defaultdict
import math
import time
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from torch_brain.data import Dataset
from torch_brain.data.sampler import SequentialFixedWindowSampler
from torch_brain.transforms import Compose

from src.models import BaseNeuronEncoder
from src.losses import Loss
from src.dataset import TwoViewDataset, TwoViewData, ViewData, get_window
from torch_brain.data.dataset import DatasetIndex
from src.samplers import (
    RandomFixedWindowSampler,
    DistributedSamplerWrapper,
)
from src.utils import (
    Precision,
    rank_zero_only,
    get_num_params,
    get_num_trainable_params,
    validate_weights,
    seed_everything,
)
from src.evaluators.collector import EmbeddingCollector
from src.trainers.trainer import Trainer, CheckpointDict
from src.utils import instantiate


def seed_train_worker(worker_id):
    """Also seed the dataset's explicit Generator, beyond PyTorch's worker RNGs."""
    worker = torch.utils.data.get_worker_info()
    worker.dataset.np_rng = np.random.default_rng(torch.initial_seed())


def train_loader_randomness(seed, rank):
    """Preserve legacy rank 0 exactly; give other ranks private worker streams."""
    if rank == 0:
        return {}
    return {
        "generator": torch.Generator().manual_seed(seed + rank),
        "worker_init_fn": seed_train_worker,
    }


class CleanWindowDataset(torch.utils.data.Dataset):
    """Own clean HDF5 data explicitly; no local class can capture it or the model."""

    def __init__(self, dataset, tokenize):
        self.dataset = dataset
        self.tokenize = tokenize

    def __getitem__(self, index):
        if self.dataset is None:
            raise RuntimeError("Clean monitor dataset is closed")
        return self.tokenize(get_window(self.dataset, index))

    def close(self):
        dataset, self.dataset = self.dataset, None
        self.tokenize = None
        if dataset is not None:
            dataset._close_open_files()


class ContrastiveTrainer(Trainer):
    def setup(self, ckpt: Optional[CheckpointDict]):
        self.cfg.model.ctx_duration = self.cfg.views.duration

        self._setup_model()
        self._setup_train_loader()
        self._setup_val_loader()
        self._setup_embedding_collectors()
        self._setup_evaluator()
        self._setup_clean_monitor_loader()

        self._setup_optimizer()
        if self.cfg.get("debias", {}).get("enabled", False) or self.cfg.get(
            "debias", {}
        ).get("track_normalizer", False):
            self._setup_debias()
        if self.cfg.get("target_steps") is not None:
            total = self.steps_per_epoch * self.cfg.num_epochs
            warmup = max(1, self.steps_per_epoch)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: (
                    step / warmup
                    if step < warmup
                    else 0.5
                    * (
                        1
                        + math.cos(
                            math.pi * min(1.0, (step - warmup) / max(1, total - warmup))
                        )
                    )
                ),
            )
            self.add_checkpoint_items(scheduler=self.scheduler)
        self._apply_ckpt(ckpt)
        resuming = ckpt is not None and self.cfg.ckpt.resume
        self.best_val_loss = (
            ckpt.get("best_val_loss", float("inf")) if resuming else float("inf")
        )
        self.best_record = ckpt.get("best_record") if resuming else None
        self.model = self.make_ddp(self.model)  # type: ignore
        self.loss = self.make_ddp(self.loss)  # type: ignore
        if self.cfg.get("debias", {}).get("enabled", False) or self.cfg.get(
            "debias", {}
        ).get("track_normalizer", False):
            self.debias.wrap(self)
        # Initialization and both parameter broadcasts must precede rank offsets.
        # Rank 0 is deliberately a no-op: resetting it would change legacy draws.
        if self.is_distributed and self.rank != 0:
            seed_everything(self.cfg.seed + self.rank)
            self.train_loader.dataset.np_rng = np.random.default_rng(
                self.cfg.seed + self.rank
            )

    def train(self):
        try:
            return super().train()
        finally:
            self.close()

    def close(self):
        """Release only the HDF5 files owned by the optional clean monitor."""
        loader = getattr(self, "clean_monitor_loader", None)
        self.clean_monitor_loader = None
        if loader is not None:
            loader.dataset.close()
        if hasattr(self, "debias"):
            self.debias.close()

    def _setup_debias(self):
        from src.debias_runtime import DebiasRuntime

        self.debias = DebiasRuntime(self)

    def _setup_clean_monitor_loader(self):
        self.clean_monitor_loader = None
        self.clean_monitor_setup_seconds = 0.0
        if (
            self.cfg.data.evaluator is None
            or self.cfg.train_transform.get("augmentation", "off") == "off"
            or self.cfg.debug is not None
            or self.cfg.val.first
        ):
            return

        started = time.perf_counter()
        dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.data.train_dataset,
            unit_id_prefix_fn=lambda _: "",
        )
        # setup runs before DDP wrapping: retain only the underlying tokenizer.
        clean = CleanWindowDataset(dataset, self.model.tokenize)
        try:
            intervals = self.train_loader.dataset.sanitized_sampling_intervals
            sampler = DistributedSamplerWrapper(
                [
                    DatasetIndex(
                        recording, float(start), float(start) + self.cfg.views.duration
                    )
                    for recording, blocks in sorted(intervals.items())
                    for start in blocks.start
                ]
            )
            self.clean_monitor_loader = DataLoader(
                clean,
                sampler=sampler,
                batch_size=self.cfg.batch_size // self.world_size,
                collate_fn=ViewData.collate,
                num_workers=0,
                generator=torch.Generator().manual_seed(self.cfg.seed),
            )
        except BaseException:
            clean.close()
            raise
        self.clean_monitor_setup_seconds = time.perf_counter() - started
        self.logger.info(
            f"Clean monitor setup: {self.clean_monitor_setup_seconds:.4f}s (once)"
        )

    def train_epoch(self):
        if self.cfg.get("debias", {}).get("enabled", False) or self.cfg.get(
            "debias", {}
        ).get("track_normalizer", False):
            return self.debias.train_epoch(self)
        self.model.train()
        self.loss.train()
        self.train_emb_collector.reset()
        self.train_loader.sampler.set_epoch(self.epoch)  # pyright: ignore

        for batch in self.logger.get_pbar(self.train_loader, "Train"):
            limit = self.cfg.get("max_steps") or self.cfg.get("target_steps")
            if limit is not None and self.train_step >= limit:
                break
            self.lr_step()

            batch: TwoViewData
            batch.to(self.device)

            with self.autocast():
                y1 = self.model(**batch.view1.data["enc_input"])
                y2 = self.model(**batch.view2.data["enc_input"])

                self.train_emb_collector.update(y1, batch.view1.unit_ids)

                loss = self.loss(
                    y1,
                    y2,
                    batch,
                    prefix="train",
                    logger=self.logger,
                )

                assert y1.isfinite().all(), "Y1 is not finite"
                assert y2.isfinite().all(), "Y2 is not finite"
                assert loss.isfinite().all(), "Loss is not finite"

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.grad_clip.enable:
                    max_norm = self.cfg.grad_clip.max_norm
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    torch.nn.utils.clip_grad_norm_(self.loss.parameters(), max_norm)
                self.optimizer.step()
                if hasattr(self, "scheduler"):
                    self.scheduler.step()

            validate_weights(self.model)
            validate_weights(self.loss)
            self.log_weight_stats()
            self.log_grad_stats()
            self.log_lr()
            self.push_logs()

            self.train_step += 1

    def autocast(self):
        return (
            torch.autocast(device_type="cuda", dtype=self.precision.dtype)
            if self.device.type == "cuda" and self.precision.dtype != torch.float32
            else nullcontext()
        )

    def lr_step(self):
        """Currently only implements linear-rise, cosine decay schedule"""
        if hasattr(self, "scheduler"):
            return
        lr = self.cfg.lr.max

        if self.cfg.lr.rise is not None:
            rise_until_step = self.cfg.lr.rise * self.steps_per_epoch
            if self.train_step < rise_until_step:
                lr = lr * self.train_step / rise_until_step

        if self.cfg.lr.decay is not None:
            decay_start_epoch = self.cfg.lr.decay
            if isinstance(decay_start_epoch, float):
                decay_start_epoch = decay_start_epoch * self.cfg.num_epochs

            if self.epoch >= decay_start_epoch:
                decay_start_step = decay_start_epoch * self.steps_per_epoch
                decay_step = self.train_step - decay_start_step
                total_steps = self.steps_per_epoch * self.cfg.num_epochs
                total_decay_steps = total_steps - decay_start_step
                lr = lr * (1 + np.cos(np.pi * decay_step / total_decay_steps)) * 0.5

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @torch.inference_mode()
    def val_epoch(self):
        if self.cfg.get("debias", {}).get("enabled", False) and self.cfg.debias.pilot:
            # Calibration is train-only. Never use validation loss or probes to
            # select the coefficient (SPEC §8, 2026-10-24 15:40).
            return
        if self.cfg.debug == "train":
            return

        self.last_val_loss = None
        self.model.eval()
        self.loss.eval()
        self.val_emb_collector.reset()

        if not self.cfg.val.loss:
            for batch in self.logger.get_pbar(self.val_loader, "Test"):
                assert isinstance(batch, ViewData)
                batch.to(self.device)

                with self.autocast():
                    y1 = self.model(**batch.data["enc_input"])
                    self.val_emb_collector.update(y1, batch.unit_ids)

        else:
            running_loss_dict = defaultdict(lambda: 0.0)
            steps = 0
            for batch in self.logger.get_pbar(self.val_loader, "Test"):
                assert isinstance(batch, TwoViewData)
                batch.to(self.device)

                with self.autocast():
                    y1 = self.model(**batch.view1.data["enc_input"])
                    y2 = self.model(**batch.view2.data["enc_input"])
                    self.val_emb_collector.update(y1, batch.view1.unit_ids)
                    loss_dict = self.loss(
                        y1,
                        y2,
                        batch,
                        return_loss_dict=True,
                        prefix="val",
                        logger=self.logger,
                    )

                for k, v in loss_dict.items():
                    running_loss_dict[k] = v + running_loss_dict[k]
                steps += 1

            # Average loss and log
            for k, v in running_loss_dict.items():
                running_loss_dict[k] /= steps
            self.logger.log_dict(running_loss_dict)
            self.last_val_loss = float(running_loss_dict["val/loss"])

        self._run_evaluator()
        self.push_logs()

    def _setup_model(self):
        self.precision = Precision(self.cfg.precision)
        self.model: BaseNeuronEncoder = instantiate(
            self.cfg.model, precision=self.precision
        ).to(self.device)
        self.loss: Loss = instantiate(self.cfg.loss, model=self.model).to(self.device)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.logger.info(f"Loss: {self.loss.__class__}")

        num_params = get_num_params(self.model) + get_num_params(self.loss)
        self.logger.info(f"Num params: {num_params:,}")

        num_trainable_params = get_num_trainable_params(self.model)
        num_trainable_params += get_num_trainable_params(self.loss)
        self.logger.info(f"Num trainable params: {num_trainable_params:,}")

        self.add_checkpoint_items(
            model=self.model,
            loss=self.loss,
        )

    def _setup_optimizer(self):
        decay = []
        no_decay = []

        def update_params(module: torch.nn.Module):
            for name, param in module.named_parameters():
                if "norm" in name or "bias" in name:
                    no_decay.append(param)
                else:
                    decay.append(param)

        update_params(self.model)
        update_params(self.loss)

        params = [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.optimizer: torch.optim.Optimizer = instantiate(self.cfg.optim, params)
        self.logger.info(f"Optimizer: {self.optimizer.__class__}")

        self.add_checkpoint_items(
            optimizer=self.optimizer,
        )

    def _apply_ckpt(self, ckpt: Optional[CheckpointDict]):
        if ckpt is None:
            return

        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.cfg.ckpt.resume:
            self.loss.load_state_dict(ckpt["loss_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if hasattr(self, "scheduler"):
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if hasattr(self, "debias"):
                self.debias.load_checkpoint(ckpt)
        elif self.cfg.ckpt.model_only:
            self.loss.load_state_dict(ckpt["loss_state_dict"])

    def _setup_train_loader(self):
        ds = TwoViewDataset(
            root=self.cfg.data_root,
            config=self.cfg.data.train_dataset,
            unit_id_prefix_fn=lambda _: "",
        )
        if self.cfg.get("debias", {}).get("enabled", False) or self.cfg.get(
            "debias", {}
        ).get("track_normalizer", False):
            ds.debias_conditions = True
        ds.transform = Compose(
            [
                instantiate(self.cfg.train_transform),
                self.model.tokenize,
            ]
        )
        ds.common_transform = Compose([TwoViewData.from_two_views, self.loss.tokenize])  # type: ignore
        ds.set_second_view_props(
            window_length=self.cfg.views.duration,
            max_time_diff=self.cfg.views.max_distance,
            seed=self.cfg.seed,
            pairing=self.cfg.data.get("pairing"),
        )

        sampler = DistributedSamplerWrapper(
            RandomFixedWindowSampler(
                sampling_intervals=ds.sanitized_sampling_intervals,
                window_length=self.cfg.views.duration,
                seed=self.cfg.seed,
            ),
        )

        multi_wrkrs = self.cfg.num_workers > 0
        self.train_loader = DataLoader(
            dataset=ds,
            sampler=sampler,
            collate_fn=TwoViewData.collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            multiprocessing_context=mp.get_context("fork") if multi_wrkrs else None,
            persistent_workers=multi_wrkrs,
            drop_last=True,
            **train_loader_randomness(self.cfg.seed, self.rank),
        )

        self.steps_per_epoch = len(self.train_loader)
        if self.steps_per_epoch == 0:
            raise ValueError(
                "Training selection has no complete batch; reduce batch_size"
            )
        if self.cfg.get("target_steps") is not None:
            self.cfg.num_epochs = math.ceil(
                self.cfg.target_steps / self.steps_per_epoch
            )
            self.logger.info(
                f"Training {self.cfg.num_epochs} epochs, {self.steps_per_epoch} steps/epoch, target {self.cfg.target_steps} steps"
            )

        self.logger.info(f"Train sessions: {len(self.train_session_ids)}")
        self.logger.info(f"Train units: {len(self.train_unit_ids)}")
        self.logger.info(f"Train steps/epoch: {self.steps_per_epoch}")

        if hasattr(self.loss, "init_recording_classifier"):
            self.loss.init_recording_classifier(self.train_session_ids)  # type: ignore
            self.loss.to(self.device)

    def _setup_val_loader(self):
        if not self.cfg.val.loss:
            ds = Dataset(
                root=self.cfg.data_root,
                config=self.cfg.data.val_dataset,
                unit_id_prefix_fn=lambda _: "",
            )
            ds.transform = self.model.tokenize  # type: ignore
            collate_fn = ViewData.collate
        else:
            # Need two view dataset if we want to compute loss
            ds = TwoViewDataset(
                root=self.cfg.data_root,
                config=self.cfg.data.val_dataset,
                unit_id_prefix_fn=lambda _: "",
            )
            ds.transform = self.model.tokenize  # type: ignore
            ds.common_transform = Compose([TwoViewData.from_two_views, self.loss.tokenize])  # type: ignore
            ds.set_second_view_props(
                window_length=self.cfg.views.duration,
                max_time_diff=self.cfg.views.max_distance,
                seed=self.cfg.seed,
                pairing=self.cfg.data.get("pairing"),
                deterministic=True,
            )
            collate_fn = TwoViewData.collate

        sampler = DistributedSamplerWrapper(
            SequentialFixedWindowSampler(
                sampling_intervals=ds.get_sampling_intervals(),
                window_length=self.cfg.views.duration,
                step=None,
                drop_short=True,
            )
        )

        multi_wrkrs = self.cfg.num_workers > 0
        self.val_loader = DataLoader(
            dataset=ds,
            sampler=sampler,
            collate_fn=collate_fn,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            multiprocessing_context=mp.get_context("fork") if multi_wrkrs else None,
            persistent_workers=multi_wrkrs,
        )

        self.logger.info(f"Val sessions: {len(self.val_session_ids)}")
        self.logger.info(f"Val units: {len(self.val_unit_ids)}")
        self.logger.info(f"Val steps/epoch: {len(self.val_loader)}")

        if hasattr(self.loss, "init_recording_classifier"):
            raise NotImplementedError(
                "Val loss with Domain-Adversarial loss not implemented"
            )

    def _setup_embedding_collectors(self):
        self.train_emb_collector = EmbeddingCollector(
            self.train_unit_ids,
            dim=self.model.emb_dim,
            device=self.device,
        )
        self.val_emb_collector = EmbeddingCollector(
            self.val_unit_ids,
            dim=self.model.emb_dim,
            device=self.device,
        )

    @rank_zero_only
    def _setup_evaluator(self):
        if self.cfg.debug is not None or self.cfg.data.evaluator is None:
            self.evaluator = None
            return

        self.evaluator = instantiate(
            self.cfg.data.evaluator,
            world_size=self.world_size,
        )

    def _run_evaluator(self):
        if self.cfg.debug is not None:
            return

        if self.cfg.val.first:
            return

        clean_monitor_started = None
        if self.clean_monitor_loader is not None:
            clean_monitor_started = time.perf_counter()
            # AUG: the train collector contains augmented training embeddings.
            # Rebuild it from clean, block-aligned windows at the current eval weights.
            assert not self.model.training
            self.train_emb_collector.reset()
            with torch.inference_mode():
                for clean in self.clean_monitor_loader:
                    clean.to(self.device)
                    with self.autocast():
                        embeddings = self.model(**clean.data["enc_input"])
                    self.train_emb_collector.update(embeddings, clean.unit_ids)
        train_unit_embs, train_unit_ids = self.train_emb_collector.compute()
        val_unit_embs, val_unit_ids = self.val_emb_collector.compute()
        if self.rank == 0 and self.evaluator is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = (
                clean_monitor_started
                if clean_monitor_started is not None
                else time.perf_counter()
            )
            compute_seconds = started - getattr(self, "epoch_started", started)
            metrics = self.evaluator(
                X_train=train_unit_embs,
                train_unit_ids=train_unit_ids,
                X_val=val_unit_embs,
                val_unit_ids=val_unit_ids,
            )
            self.logger.log_dict(metrics)
            if self.cfg.data.get("brainset") == "euler_allgcl_2025":
                seconds = time.perf_counter() - started
                fraction = seconds / max(compute_seconds, 1e-9)
                self.last_monitor_seconds = seconds
                self.last_monitor_epoch_fraction = fraction
                self.logger.log("monitor/seconds", seconds)
                self.logger.log("monitor/epoch_fraction", fraction)
                self.logger.info(
                    f"Evaluator: {seconds:.4f}s / {compute_seconds:.4f}s ({fraction:.2%})"
                )
