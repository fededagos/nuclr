"""Opt-in trainer integration. All epoch resources have explicit ownership/close.

Pilot behavior and audit acceptance follow SPEC §8 (2026-10-24 15:40).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_brain.data import Dataset
from torch_brain.data.dataset import DatasetIndex

from src.debias import (
    Calibration,
    DebiasHead,
    DebiasPopulation,
    FrozenNormalizer,
    EMANormalizer,
    HeadPrior,
    attach_condition,
    auxiliary_loss,
    row_coefficients,
)
from src.dataset import ViewData, get_window
from src.distributed import AllReduce, AllReduceSum
from src.utils import validate_weights


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def encoder_alpha(cfg, step):
    if cfg.pilot:
        return 0.0
    start = cfg.normalization_fit_steps + cfg.head_warmup_steps
    return cfg.lambda_max * min(1.0, max(0.0, (step - start) / cfg.encoder_ramp_steps))


def freeze_common_normalizer(normalizer, reference):
    """Prefix fits once; consumers load that exact scaler, including after resume."""
    if reference is None:
        normalizer.freeze()
        return
    if (
        not bool(reference["frozen"])
        or reference["count"] < 2
        or not torch.isfinite(reference["mean"]).all()
        or not torch.isfinite(reference["std"]).all()
        or (reference["std"] <= 0).any()
    ):
        raise RuntimeError("Invalid common frozen normalizer")
    normalizer.load_state_dict(reference)


def clip_with_stats(parameters, max_norm):
    parameters = list(parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite pre-clip gradient norm")
    return dict(
        preclip_norm=float(norm),
        active=int(norm > max_norm),
        scale=min(1.0, max_norm / (float(norm) + 1e-6)),
    )


class AuditWindows(torch.utils.data.Dataset):
    """One owned H5 dataset; no closure captures a trainer or a bound epoch."""

    def __init__(self, dataset, tokenize, indices):
        self.dataset, self.tokenize, self.indices = dataset, tokenize, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.dataset is None:
            raise RuntimeError("Audit dataset is closed")
        return attach_condition(
            self.tokenize(get_window(self.dataset, self.indices[index])), True
        )

    def close(self):
        dataset, self.dataset, self.tokenize = self.dataset, None, None
        if dataset is not None:
            dataset._close_open_files()


class DebiasRuntime:
    def __init__(self, trainer):
        import pyarrow.parquet as pq

        self.cfg = trainer.cfg.debias
        if not isinstance(self.cfg.get("track_normalizer", False), bool):
            raise TypeError("debias.track_normalizer must be a bool")
        if not isinstance(self.cfg.enabled, bool):
            raise TypeError("debias.enabled must be a bool")
        if self.cfg.head_lr_schedule != "constant":
            raise ValueError("The registered head LR schedule is constant")
        self.device, self.rank = trainer.device, trainer.rank
        ledger_path = Path(self.cfg.ledger_file)
        self.ledger = json.loads(ledger_path.read_text())
        frame = pq.read_table(
            trainer.cfg.data.units_file, filters=[("split", "=", "train")]
        ).to_pandas()
        self.population = DebiasPopulation(
            frame,
            self.cfg.target,
            self.ledger["audit_fields"],
            self.ledger["exposure"][self.cfg.target],
        )
        if not self.population.corrections:
            raise RuntimeError("No supported head-training population")
        self.common_population = DebiasPopulation(
            frame,
            "split_group",
            self.ledger["audit_fields"],
            self.ledger["exposure"]["split_group"],
        )
        self.head = DebiasHead(
            trainer.model.emb_dim,
            len(self.population.class_map),
            self.cfg.target == "setupid",
            trainer.cfg.seed,
        ).to(self.device)
        self.normalizer_mode = self.cfg.get("normalizer_mode", "frozen")
        if self.normalizer_mode not in ("ema", "frozen"):
            raise ValueError("Unknown normalizer mode")
        self.normalizer = (
            EMANormalizer(
                trainer.model.emb_dim, self.cfg.get("normalizer_momentum", 0.01)
            )
            if self.normalizer_mode == "ema"
            else FrozenNormalizer(trainer.model.emb_dim)
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.cfg.head_optimizer.lr,
            weight_decay=self.cfg.head_optimizer.weight_decay,
            betas=tuple(self.cfg.head_optimizer.betas),
            eps=self.cfg.head_optimizer.eps,
        )
        self.calibration = Calibration()
        self.prior = HeadPrior(
            len(self.population.class_map), self.cfg.target == "setupid"
        ).to(self.device)
        self.clip_counts = {}
        self.steps_seen = 0
        self.exposure_seen = {}
        self.cumulative_lost_mass = 0.0
        self.metadata_diagnostics = {
            (str(row.session_id), str(row.unit_id)): (
                str(row.experimenter),
                str(row.setupid),
                str(row.supergroup),
            )
            for row in frame.itertuples()
        }
        self.common_reference = None
        if self.cfg.get("normalizer_file"):
            self.common_reference = torch.load(
                self.cfg.normalizer_file, map_location=self.device, weights_only=False
            )
        self.audit_loader = None
        # Construct once, with a private generator and no persistent workers.
        dataset = Dataset(
            root=trainer.cfg.data_root,
            config=trainer.cfg.data.train_dataset,
            unit_id_prefix_fn=lambda _: "",
        )
        records = list(self.ledger["audit_windows"])
        # Replicate this small fixed audit set: no padded/duplicated observation
        # changes its population, and every rank executes identical collectives.
        if not records:
            dataset._close_open_files()
            raise RuntimeError("No registered non-panel train audit windows")
        indices = [DatasetIndex(**r) for r in records]
        audit = AuditWindows(dataset, trainer.model.tokenize, indices)
        self.audit_loader = DataLoader(
            audit,
            batch_size=1,
            collate_fn=ViewData.collate,
            num_workers=0,
            generator=torch.Generator().manual_seed(trainer.cfg.seed + 991),
        )
        trainer.add_checkpoint_items(
            debias=self, debias_head=self.head, debias_optimizer=self.optimizer
        )
        self.output = Path(trainer.cfg.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            (self.output / "debias_population.json").write_text(
                json.dumps(self.population.state_dict(), indent=2)
            )

    def wrap(self, trainer):
        if trainer.is_distributed:
            population = [self.population.state_dict() if trainer.rank == 0 else None]
            dist.broadcast_object_list(population, src=0)
            if population[0] != self.population.state_dict():
                raise RuntimeError("Class maps/coefficients differ across ranks")
            for buffer in self.normalizer.buffers():
                dist.broadcast(buffer, src=0)
        self.head = trainer.make_ddp(self.head)
        trainer.debias_head = self.head
        trainer.add_checkpoint_items(debias_head=self.head)

    def state_dict(self):
        return dict(
            normalizer=self.normalizer.state_dict(),
            normalizer_mode=self.normalizer_mode,
            track_normalizer=bool(self.cfg.get("track_normalizer", False)),
            prior=self.prior.state_dict(),
            population=self.population.state_dict(),
            ledger=self.ledger,
            calibration=self.calibration.state_dict(),
            lambda_max=float(self.cfg.lambda_max),
            pilot=bool(self.cfg.pilot),
            clip_counts=self.clip_counts,
            steps_seen=self.steps_seen,
            exposure_seen=self.exposure_seen,
            cumulative_lost_mass=self.cumulative_lost_mass,
        )

    def load_state_dict(self, state):
        self.population.load_state_dict(state["population"])
        if (
            state.get("normalizer_mode", "frozen") != self.normalizer_mode
            or state.get("track_normalizer", False)
            != self.cfg.get("track_normalizer", False)
            or state["ledger"] != self.ledger
            or state["lambda_max"] != self.cfg.lambda_max
            or state["pilot"] != self.cfg.pilot
        ):
            raise ValueError("Debias ledger/coefficient/pilot changed on resume")
        self.normalizer.load_state_dict(state["normalizer"])
        self.prior.load_state_dict(state["prior"])
        self.calibration.load_state_dict(state["calibration"])
        self.clip_counts, self.steps_seen = state["clip_counts"], state["steps_seen"]
        self.exposure_seen = state["exposure_seen"]
        self.cumulative_lost_mass = state["cumulative_lost_mass"]

    def load_checkpoint(self, ckpt):
        self.head.load_state_dict(ckpt["debias_head_state_dict"])
        self.optimizer.load_state_dict(ckpt["debias_optimizer_state_dict"])
        self.load_state_dict(ckpt["debias_state_dict"])

    def close(self):
        loader, self.audit_loader = self.audit_loader, None
        if loader is not None:
            loader.dataset.close()

    def rows(self, ys, views, batch_items, pilot=None, audit=False, common=False):
        features, labels, correction, experimenter, items, view_ids, animals = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        population = self.common_population if common else self.population
        for v, (y, view) in enumerate(zip(ys, views)):
            row = population.lookup(
                view.unit_ids,
                view.recording_ids,
                self.cfg.pilot if pilot is None else pilot,
                audit,
            )
            mask = row["eligible"] & view.debias_control
            idx = torch.as_tensor(np.flatnonzero(mask), device=y.device)
            features.append(y[idx])
            labels.extend(row["label"][mask])
            correction.extend(row["correction"][mask])
            experimenter.extend(row["experimenter"][mask])
            animals.extend(row["animal"][mask])
            item = np.repeat(np.arange(batch_items), view.unit_seqlen)
            items.extend(item[mask])
            view_ids.extend([v] * int(mask.sum()))
        x = torch.cat(features)
        tensor = lambda value, dtype: torch.as_tensor(
            np.array(value), dtype=dtype, device=x.device
        )
        weights, info = row_coefficients(
            tensor(items, torch.long),
            tensor(view_ids, torch.long),
            tensor(correction, torch.float32),
            batch_items,
        )
        return (
            x,
            tensor(labels, torch.long),
            weights,
            tensor(experimenter, torch.long),
            info,
            animals,
        )

    def update_normalizer(self, ys, batch):
        """Update BEFORE normalizing the current step; never called by audits/val.

        One shared eligibility policy: A cohort, non-panel control rows, both views.

        Pilot audit-field exclusion applies only to head fitting, not to these
        target-independent moments. This path is identical in Z, A, B and pilots.
        """
        common = self.rows(
            ys, [batch.view1, batch.view2], batch.batch_size, pilot=False, common=True
        )[0]
        self.normalizer.update(common)

    def head_loss(self, ys, batch, alpha, head=None, update_prior=False):
        x, labels, weights, e, info, animals = self.rows(
            ys, [batch.view1, batch.view2], batch.batch_size
        )
        normalized = self.normalizer(x)
        if alpha == 0:
            normalized = normalized.detach()
        loss, ce, correct = auxiliary_loss(
            head or self.head,
            normalized,
            labels,
            weights,
            batch.batch_size,
            alpha,
            e,
            return_details=True,
        )
        if update_prior:
            self.prior.observe(labels, e, weights / info["global_base_items"])
            if (self.steps_seen + 1) % 25 == 0:
                selected = []
                for view in (batch.view1, batch.view2):
                    rows = self.population.lookup(
                        view.unit_ids, view.recording_ids, self.cfg.pilot
                    )
                    mask = rows["eligible"] & view.debias_control
                    selected.extend(
                        self.metadata_diagnostics[(str(r).rsplit("/", 1)[-1], str(u))]
                        for u, r in zip(view.unit_ids[mask], view.recording_ids[mask])
                    )
                mass = weights.detach().cpu().numpy() / info["global_base_items"]
                values = ce.cpu().numpy()
                hits = correct.cpu().numpy()
                for animal in sorted(set(animals)):
                    mask = np.array([a == animal for a in animals])
                    total = float(mass[mask].sum())
                    info[f"animal/{animal}/mass"] = total
                    info[f"animal/{animal}/rows"] = int(mask.sum())
                    if total:
                        info[f"animal/{animal}/ce"] = float(
                            (values[mask] * mass[mask]).sum() / total
                        )
                        info[f"animal/{animal}/recall"] = float(
                            (hits[mask] * mass[mask]).sum() / total
                        )
                for row, weight in zip(selected, mass):
                    key = "cell_mass/" + "/".join(row)
                    info[key] = info.get(key, 0.0) + float(weight)
                info["distinct_animals"] = len(set(animals))
                from collections import Counter

                base_animals = []
                start = 0
                for count in batch.view1.unit_seqlen:
                    uid = batch.view1.unit_ids[start]
                    sid = str(batch.view1.recording_ids[start]).rsplit("/", 1)[-1]
                    base_animals.append(self.population.rows[(sid, str(uid))]["animal"])
                    start += count
                for animal, count in Counter(base_animals).items():
                    info[f"base_animal/{animal}/items"] = count
        return loss, info, animals

    def _clip(self, trainer):
        modules = [
            ("encoder", trainer.model),
            ("projector", trainer.loss),
            ("head", self.head),
        ]
        values = {}
        for name, module in modules:
            bound = (
                self.cfg.head_clip_norm
                if name == "head"
                else trainer.cfg.grad_clip.max_norm
            )
            stats = clip_with_stats(module.parameters(), bound)
            self.clip_counts[name] = self.clip_counts.get(name, 0) + stats["active"]
            stats["activation_fraction"] = self.clip_counts[name] / max(
                1, self.steps_seen
            )
            values.update({f"{name}/{k}": v for k, v in stats.items()})
        self._log(trainer, values)

    def _log(self, trainer, values, step=None):
        # Logger suppresses nonzero ranks: gather rank-labelled scalar diagnostics to rank 0.
        rows = [None] * trainer.world_size
        if trainer.is_distributed:
            dist.all_gather_object(rows, values)
        else:
            rows = [values]
        if self.rank == 0:
            from collections import Counter

            base_counts = Counter()
            eligible_animals = set()
            for row in rows:
                for key, value in row.items():
                    if key.startswith("base_animal/"):
                        base_counts[key] += value
                    if key.startswith("animal/") and key.endswith("/rows"):
                        eligible_animals.add(key)
            if base_counts:
                trainer.logger.log(
                    "debias/global_distinct_base_animals", len(base_counts)
                )
                trainer.logger.log(
                    "debias/global_repeat_base_items",
                    sum(max(0, n - 1) for n in base_counts.values()),
                )
                trainer.logger.log(
                    "debias/global_distinct_eligible_animals", len(eligible_animals)
                )
            for rank, row in enumerate(rows):
                trainer.logger.log_dict(
                    {f"debias/rank{rank}/{k}": v for k, v in row.items()}
                )
            with (self.output / "debias_history.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            step=trainer.train_step + 1 if step is None else step,
                            ranks=rows,
                        )
                    )
                    + "\n"
                )

    def audit(self, trainer, drift=True, require_prior=False):
        """Fixed train-field windows only; no valid/panel target is evaluated here."""
        model = unwrap(trainer.model)
        was_training = model.training
        model.eval()
        unwrap(self.head).eval()
        chunks, numerator, prior, denominator, recall, mass = (
            [],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        head_metrics = bool(self.prior.counts.sum() > 0)
        prior_supported = True
        with torch.no_grad(), trainer.autocast():
            for view in self.audit_loader:
                view.to(self.device)
                y = model(**view.data["enc_input"])
                common = self.common_population.lookup(
                    view.unit_ids, view.recording_ids
                )
                chunks.append(y[torch.as_tensor(common["eligible"], device=y.device)])
                if not head_metrics:
                    continue
                x, labels, weights, e, info, _ = self.rows(
                    [y], [view], 1, pilot=False, audit=True
                )
                logits = unwrap(self.head)(self.normalizer(x), e)
                n = (
                    F.cross_entropy(logits.float(), labels, reduction="none") * weights
                ).sum()
                try:
                    p = self.prior.probabilities(labels, e)
                except RuntimeError:
                    if require_prior:
                        raise
                    prior_supported = False
                else:
                    prior += float(AllReduce.apply((-p.log() * weights).sum()))
                numerator += float(AllReduce.apply(n))
                recall += float(
                    AllReduce.apply(((logits.argmax(-1) == labels) * weights).sum())
                )
                mass += float(AllReduce.apply(weights.sum()))
                denominator += 1
        model.train(was_training)
        unwrap(self.head).train()
        result = (
            dict(
                audit_ce=numerator / denominator,
                prior_ce=prior / denominator if prior_supported else None,
                prior_supported=prior_supported,
                audit_recall_descriptive=recall / mass if mass else None,
            )
            if head_metrics
            else {}
        )
        if drift:
            result.update(self.normalizer.audit(torch.cat(chunks)))
        return result

    def measure(self, trainer, batch, step, update_normalizer=False):
        # Raw modules avoid DDP autograd.grad reducer hooks. Explicitly average the
        # measurement gradients once, matching the DDP average of the applied path.
        devices = [trainer.device.index] if trainer.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), trainer.autocast():
            model, loss_module, head = (
                unwrap(trainer.model),
                unwrap(trainer.loss),
                unwrap(self.head),
            )
            ys = [model(**v.data["enc_input"]) for v in (batch.view1, batch.view2)]
            if update_normalizer:
                self.update_normalizer(ys, batch)
            base = loss_module(*ys, batch, prefix="calibration", logger=None)
            aux, _, _ = self.head_loss(ys, batch, alpha=1.0, head=head)
            params = tuple(p for p in model.parameters() if p.requires_grad)
            base_grads = torch.autograd.grad(base, params, retain_graph=True)
            aux_grads = torch.autograd.grad(aux, params, allow_unused=True)
            squares = []
            for gradients in (base_grads, aux_grads):
                total = torch.zeros((), device=self.device, dtype=torch.float64)
                for parameter, gradient in zip(params, gradients):
                    gradient = (
                        torch.zeros_like(parameter)
                        if gradient is None
                        else gradient.detach().clone()
                    )
                    if trainer.is_distributed:
                        dist.all_reduce(gradient)
                        gradient /= trainer.world_size
                    total += gradient.double().square().sum()
                squares.append(total)
            if not squares[0] > 0:
                raise RuntimeError("Zero/nonfinite calibration base gradient")
            ratio = float((squares[1] / squares[0]).sqrt())
        audit = (
            self.audit(trainer, drift=False, require_prior=step >= 1150)
            if self.cfg.pilot
            else {}
        )
        if self.cfg.pilot:
            self.calibration.record(
                step,
                ratio,
                audit["audit_ce"],
                audit["prior_ce"],
                self.prior.counts.cpu().tolist(),
            )
        self._log(trainer, dict(unit_alpha_gradient_ratio=ratio, **audit))

    def train_epoch(self, trainer):
        started = time.perf_counter()
        start_step = trainer.train_step
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        trainer.model.train()
        trainer.loss.train()
        self.head.train()
        trainer.train_emb_collector.reset()
        trainer.train_loader.sampler.set_epoch(trainer.epoch)
        for batch in trainer.logger.get_pbar(trainer.train_loader, "Train"):
            limit = trainer.cfg.get("max_steps") or trainer.cfg.target_steps
            if trainer.train_step >= limit:
                break
            trainer.lr_step()
            batch.to(self.device)
            step = trainer.train_step + 1
            measured_ema = False
            if (
                self.cfg.enabled
                and self.normalizer.frozen
                and (
                    (self.cfg.pilot and Calibration.is_measurement(step))
                    or (not self.cfg.pilot and step > 250 and step % 25 == 0)
                )
            ):
                measured_ema = self.normalizer_mode == "ema"
                self.measure(trainer, batch, step, update_normalizer=measured_ema)
            with trainer.autocast():
                ys = [
                    trainer.model(**v.data["enc_input"])
                    for v in (batch.view1, batch.view2)
                ]
                trainer.train_emb_collector.update(ys[0], batch.view1.unit_ids)
                base = trainer.loss(*ys, batch, prefix="train", logger=trainer.logger)
                if not self.normalizer.frozen:
                    common = self.rows(
                        ys,
                        [batch.view1, batch.view2],
                        batch.batch_size,
                        pilot=False,
                        common=True,
                    )[0]
                    self.normalizer.observe(common)
                    aux = None
                    if step == self.cfg.normalization_fit_steps:
                        freeze_common_normalizer(self.normalizer, self.common_reference)
                        if trainer.is_distributed:
                            for buffer in self.normalizer.buffers():
                                dist.broadcast(buffer, src=0)
                        if self.rank == 0:
                            torch.save(
                                self.normalizer.state_dict(),
                                self.output / "normalizer.pt",
                            )
                else:
                    if self.normalizer_mode == "ema" and not measured_ema:
                        self.update_normalizer(ys, batch)
                    aux = None
                    if self.cfg.enabled:
                        # Pilots measure unit-alpha VJPs without applying them.
                        aux, info, animals = self.head_loss(
                            ys, batch, encoder_alpha(self.cfg, step), update_prior=True
                        )
                        self._log(
                            trainer,
                            dict(
                                aux_ce=float(aux.detach()),
                                alpha=encoder_alpha(self.cfg, step),
                                **info,
                            ),
                        )
                        self.cumulative_lost_mass += (
                            info["lost_mass"] / info["global_base_items"]
                        )
                        for animal in animals:
                            self.exposure_seen[animal] = (
                                self.exposure_seen.get(animal, 0) + 1
                            )
            trainer.optimizer.zero_grad()
            self.optimizer.zero_grad()
            if (
                not all(torch.isfinite(y).all() for y in ys)
                or not torch.isfinite(base)
                or (aux is not None and not torch.isfinite(aux))
            ):
                raise FloatingPointError("Nonfinite encoder/base/auxiliary value")
            (base if aux is None else base + aux).backward()
            self.steps_seen += 1
            self._clip(trainer)
            trainer.optimizer.step()
            if aux is not None:
                self.optimizer.step()
            if hasattr(trainer, "scheduler"):
                trainer.scheduler.step()
            for module in (trainer.model, trainer.loss, self.head):
                validate_weights(module)
            trainer.log_weight_stats()
            trainer.log_grad_stats()
            trainer.log_lr()
            trainer.logger.log("debias/head_lr", self.optimizer.param_groups[0]["lr"])
            trainer.push_logs()
            trainer.train_step += 1
        if self.normalizer.frozen:
            self._log(trainer, self.audit(trainer, drift=True), step=trainer.train_step)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self._log(
                trainer,
                dict(
                    peak_allocated_gib=torch.cuda.max_memory_allocated(self.device)
                    / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(self.device)
                    / 2**30,
                    epoch_steps_per_second=(trainer.train_step - start_step)
                    / (time.perf_counter() - started),
                ),
                step=trainer.train_step,
            )
        exposures = [None] * trainer.world_size
        local = dict(
            rows=self.exposure_seen,
            cumulative_lost_mass=self.cumulative_lost_mass,
            head_prior_counts=self.prior.counts.cpu().tolist(),
        )
        if trainer.is_distributed:
            dist.all_gather_object(exposures, local)
        else:
            exposures = [local]
        seen = set().union(*(r["rows"] for r in exposures))
        if self.rank == 0:
            (self.output / "debias_exposure.json").write_text(
                json.dumps(
                    dict(
                        ranks=exposures,
                        zero_realized_animals=sorted(
                            set(self.population.corrections) - seen
                        ),
                    ),
                    indent=2,
                )
            )
        if self.cfg.pilot and trainer.train_step >= 1250:
            if set(self.population.corrections) - seen:
                raise RuntimeError(
                    "Zero realized head exposure; calibration is inconclusive"
                )
            result = self.calibration.finish()
            if self.rank == 0:
                (self.output / "calibration.json").write_text(
                    json.dumps(result, indent=2)
                )
