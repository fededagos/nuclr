"""Opt-in nuisance heads; the contrastive loss and its parameter groups stay separate.

The reduction follows the average-forward/identity-backward convention in
``src.distributed``. Alpha scales only the encoder gradient, never the head CE.
"""

import hashlib
import math
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.distributed import AllReduce, AllReduceSum
from src.nn.functional.grad_reverse import grad_reverse


PANEL = frozenset(
    {
        "Experimenter_2_2023-02-14",
        "Experimenter_2_2024-02-20",
        "Experimenter_2_2022-12-08",
        "Experimenter_2_2022-12-12",
    }
)
AUDIT_SALT = "ADV-SPEC-2026-10-24-1540-audit-v1"


def attach_condition(view, control):
    # Called after transforms/tokenization, so dropout and finite masks are done.
    view.debias_control = np.full(len(view.unit_ids), bool(control))
    return view


def audit_fields(fields, field_animals=None, total_fields=None):
    """SPEC §8 16:00: salted hash draw, retaining a fit field for every animal.

    ``field_animals`` maps eligible non-panel control fields to animals. Single-
    field animals never enter the audit. The target is 10% of non-panel fields;
    insufficient support raises rather than silently changing the audit fraction.
    """
    fields = sorted(set(fields))
    ranked = sorted(
        fields, key=lambda s: hashlib.sha256(f"{AUDIT_SALT}|{s}".encode()).hexdigest()
    )
    target = math.ceil((len(fields) if total_fields is None else total_fields) * 0.1)
    if field_animals is None:
        return sorted(ranked[:target])
    from collections import Counter

    remaining = Counter(field_animals[f] for f in fields)
    selected = []
    for field in ranked:
        animal = field_animals[field]
        if remaining[animal] > 1:
            selected.append(field)
            remaining[animal] -= 1
        if len(selected) == target:
            return sorted(selected)
    raise RuntimeError("Insufficient eligible fields for class-preserving 10% audit")


class HeadPrior(nn.Module):
    """Weighted-CE-optimal constant predictor fitted on actual head-training rows.

    Accumulate *final* row coefficients (after global cap and base denominator).
    No smoothing, class balancing, metadata-cell-count surrogate or E pooling.
    The frozen audit set never enters these counts. Counts are AllReduced and
    checkpointed; each audit uses the head-training population observed so far.
    """

    def __init__(self, num_classes, conditioned=False):
        super().__init__()
        self.conditioned = conditioned
        self.register_buffer(
            "counts",
            torch.zeros(8 if conditioned else 1, num_classes, dtype=torch.float64),
        )

    @torch.no_grad()
    def observe(self, labels, experimenter, weights):
        e = experimenter if self.conditioned else torch.zeros_like(experimenter)
        counts = torch.zeros_like(self.counts).flatten()
        counts.scatter_add_(
            0, e * self.counts.shape[1] + labels, weights.detach().double()
        )
        self.counts.add_(AllReduceSum.apply(counts.reshape_as(self.counts)))

    def probabilities(self, labels, experimenter):
        e = experimenter if self.conditioned else torch.zeros_like(experimenter)
        p = self.counts[e, labels] / self.counts[e].sum(-1)
        if not torch.isfinite(p).all() or (p <= 0).any():
            raise RuntimeError(
                "Audit target has no observed head-training prior support"
            )
        return p.float()


def metadata_eligibility(frame):
    import pandas as pd

    return (
        (
            (pd.to_numeric(frame.chirp_qidx, errors="coerce") > 0.35)
            | (pd.to_numeric(frame.bar_qidx, errors="coerce") > 0.6)
        )
        & frame.supergroup.fillna("").str.strip().ne("")
        & frame.label_condition.fillna("").str.strip().ne("")
        & ~frame.label_condition_is_drug.fillna(True).astype(bool)
    )


class DebiasPopulation:
    """Immutable train metadata; no type values enter labels/features/coefficients."""

    def __init__(self, frame, target, audit_sessions, exposure):
        if target not in ("split_group", "setupid"):
            raise ValueError("Unknown debias target")
        if set(frame.split) != {"train"}:
            raise ValueError("Head metadata must contain train rows only")
        if set(frame.setupid.astype(str)) - {"1", "3"}:
            raise ValueError("Head metadata is not setup-2-free")
        self.target, self.exposure = target, dict(exposure)
        self.audit_sessions = sorted(audit_sessions)
        frame = frame.copy()
        frame["eligible"] = metadata_eligibility(frame) & ~frame.split_group.isin(PANEL)
        eligible = frame.loc[frame.eligible]
        if target == "setupid":
            crossed = eligible.groupby("experimenter").setupid.nunique()
            frame["eligible"] &= frame.experimenter.isin(crossed[crossed == 2].index)
        supported = sorted(set(frame.loc[frame.eligible, "split_group"]))
        zero = sorted(a for a in supported if exposure.get(a, 0) <= 0)
        self.zero_exposure = zero
        frame["eligible"] &= ~frame.split_group.isin(zero)
        if target == "setupid":
            crossed = (
                frame.loc[frame.eligible].groupby("experimenter").setupid.nunique()
            )
            frame["eligible"] &= frame.experimenter.isin(crossed[crossed == 2].index)
        supported = sorted(set(frame.loc[frame.eligible, "split_group"]))
        self.class_map = (
            {a: i for i, a in enumerate(supported)}
            if target == "split_group"
            else {"1": 0, "3": 1}
        )
        total = float(
            exposure.get("__base_items__", sum(float(exposure[a]) for a in supported))
        )
        self.corrections = {
            a: total / (len(supported) * exposure[a]) for a in supported
        }
        self.field_corrections = {}
        if "__field_exposure__" in exposure:
            eligible_frame = frame.loc[frame.eligible]
            animal_cells = eligible_frame.groupby("split_group").size()
            for (animal, session), group in eligible_frame.groupby(
                ["split_group", "session_id"]
            ):
                field_exp = exposure["__field_exposure__"].get(str(session), 0.0)
                # Animal correction times cell share / conditional field exposure.
                # A small field must not gain a field-macro boost within an animal.
                self.field_corrections[str(session)] = (
                    (
                        total
                        * len(group)
                        / (len(supported) * animal_cells[animal] * field_exp)
                    )
                    if field_exp > 0
                    else 0.0
                )
        self.rows = {}
        for row in frame.itertuples():
            e = int(re.search(r"(\d+)$", str(row.experimenter)).group(1)) - 1
            label = self.class_map.get(
                row.split_group if target == "split_group" else str(row.setupid), 0
            )
            self.rows[(str(row.session_id), str(row.unit_id))] = dict(
                eligible=bool(row.eligible),
                audit=str(row.session_id) in self.audit_sessions,
                label=label,
                experimenter=e,
                animal=str(row.split_group),
                correction=float(
                    self.field_corrections.get(
                        str(row.session_id), self.corrections.get(row.split_group, 0)
                    )
                ),
            )

    def lookup(self, unit_ids, recording_ids, pilot=False, audit=False):
        rows = [
            self.rows[(str(r).rsplit("/", 1)[-1], str(u))]
            for u, r in zip(unit_ids, recording_ids)
        ]
        out = {
            k: np.array([row[k] for row in rows])
            for k in ("eligible", "label", "experimenter", "animal", "correction")
        }
        if pilot or audit:
            out["eligible"] &= np.array(
                [row["audit"] == audit for row in rows], dtype=bool
            )
        return out

    def state_dict(self):
        return dict(
            target=self.target,
            exposure=self.exposure,
            audit_sessions=self.audit_sessions,
            class_map=self.class_map,
            corrections=self.corrections,
            zero_exposure=self.zero_exposure,
            field_corrections=self.field_corrections,
            rows=[(list(k), v) for k, v in sorted(self.rows.items())],
        )

    def load_state_dict(self, state):
        if self.state_dict() != state:
            raise ValueError(
                "Frozen population/weights/class map changed across resume"
            )


def row_coefficients(item, view, correction, base_items):
    """Cell mean -> available-view mean -> fixed animal correction -> global cap.

    Returned coefficients are numerator weights. Divide by the global base-item
    count, never by the surviving post-cap mass. View indices must be 0 or 1.
    """
    global_rows = AllReduceSum.apply(correction.new_tensor(float(len(item))))
    global_items = AllReduceSum.apply(correction.new_tensor(float(base_items)))
    if len(item):
        key = item * 2 + view
        count = torch.bincount(key, minlength=base_items * 2)
        views = (count.reshape(-1, 2) > 0).sum(1)
        raw = correction / count[key] / views[item]
        cap = 5 * global_items / global_rows.clamp_min(1)
        weights = raw.clamp_max(cap)
    else:
        raw = weights = correction
    lost = AllReduceSum.apply((raw - weights).sum().detach())
    return weights, dict(
        lost_mass=float(lost),
        global_eligible_rows=int(global_rows),
        global_base_items=int(global_items),
        local_mass=float(weights.sum()),
    )


class DebiasHead(nn.Module):
    def __init__(self, dim, num_classes, conditioned=False, seed=42):
        super().__init__()
        self.conditioned = conditioned
        # Initializing an optional head must consume none of the base RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.net = nn.Sequential(
                nn.Linear(dim + (8 if conditioned else 0), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            )

    def forward(self, features, experimenter=None):
        if self.conditioned:
            if experimenter is None:
                raise ValueError("The setup head requires experimenter indices")
            features = torch.cat(
                (features, F.one_hot(experimenter, 8).to(features)), -1
            )
        return self.net(features)


def auxiliary_loss(
    head,
    features,
    labels,
    coefficients,
    base_items,
    alpha,
    experimenter=None,
    return_details=False,
):
    """Global weighted numerator / base-item count, with graph-connected empty ranks.

    ``coefficients`` already include cell/view means, fixed exposure and global cap.
    Every rank executes a wrapped head forward, even with no eligible local row.
    """
    denominator = AllReduce.apply(features.new_tensor(float(base_items))).clamp_min(1)
    if len(labels):
        logits = head(grad_reverse(features, alpha), experimenter)
        ce = F.cross_entropy(logits.float(), labels, reduction="none")
        numerator = (ce * coefficients).sum()
        correct = logits.detach().argmax(-1) == labels
    else:
        dummy = features.sum(0, keepdim=True) * 0
        e = torch.zeros(1, device=features.device, dtype=torch.long)
        numerator = head(grad_reverse(dummy, alpha), e).sum() * 0
        ce = features.new_empty(0)
        correct = labels.new_empty(0, dtype=torch.bool)
    loss = AllReduce.apply(numerator / denominator)
    return (loss, ce.detach(), correct) if return_details else loss


class FrozenNormalizer(nn.Module):
    """Distributed detached population moments; freeze once, never refresh silently."""

    def __init__(self, dim):
        super().__init__()
        for name, value in [
            ("count", torch.zeros((), dtype=torch.float64)),
            ("total", torch.zeros(dim, dtype=torch.float64)),
            ("squares", torch.zeros(dim, dtype=torch.float64)),
            ("mean", torch.zeros(dim)),
            ("std", torch.ones(dim)),
            ("frozen", torch.tensor(False)),
            ("bad_audits", torch.tensor(0)),
        ]:
            self.register_buffer(name, value)

    @torch.no_grad()
    def observe(self, x):
        if self.frozen:
            raise RuntimeError("Normalizer is frozen")
        x = x.detach().double()
        self.count.add_(AllReduceSum.apply(x.new_tensor(len(x))))
        self.total.add_(AllReduceSum.apply(x.sum(0)))
        self.squares.add_(AllReduceSum.apply(x.square().sum(0)))

    @torch.no_grad()
    def freeze(self):
        if self.count < 2:
            raise RuntimeError("Insufficient normalizer coverage")
        mean = self.total / self.count
        var = self.squares / self.count - mean.square()
        if not torch.isfinite(var).all() or (var <= 0).any():
            raise RuntimeError("Nonfinite/zero normalizer scale")
        self.mean.copy_(mean)
        self.std.copy_(var.sqrt())
        self.frozen.fill_(True)

    def forward(self, x):
        if not self.frozen:
            raise RuntimeError("Normalizer has not been frozen")
        return (x.float() - self.mean) / self.std

    @torch.no_grad()
    def audit(self, x, enforce_drift=True):
        x = x.detach().double()
        n = AllReduceSum.apply(x.new_tensor(len(x)))
        total = AllReduceSum.apply(x.sum(0))
        squares = AllReduceSum.apply(x.square().sum(0))
        if n < 2:
            raise RuntimeError("Insufficient fixed audit coverage")
        mean = total / n
        std = (squares / n - mean.square()).clamp_min(0).sqrt()
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise RuntimeError("Nonfinite/zero audit scale")
        shift = ((mean - self.mean) / self.std).abs().quantile(0.95)
        ratio = (std / self.std).log2().abs().quantile(0.95)
        valid = bool(
            torch.isfinite(std).all() and (std > 0).all() and shift <= 1 and ratio <= 1
        )
        if enforce_drift:
            self.bad_audits.fill_(0 if valid else self.bad_audits + 1)
        result = dict(
            valid=valid,
            mean_shift_p95=float(shift),
            abs_log2_sd_p95=float(ratio),
            n=int(n),
        )
        if enforce_drift and self.bad_audits >= 2:
            raise RuntimeError(
                f"Normalizer invalid after two consecutive audits: {result}"
            )
        return result


class EMANormalizer(FrozenNormalizer):
    """Protocol v2: EMA of global population mean AND SD, not EMA variance.

    For global eligible rows x, N=sum_r N_r, mu_b=sum_r sum(x)/N,
    sd_b=sqrt(max(0, sum_r sum(x*x)/N-mu_b**2)). Then
    mu_t=.99*mu_(t-1)+.01*mu_b; sd_t=.99*sd_(t-1)+.01*sd_b.
    These are direct mean/SD EMAs (not a variance/second-moment EMA).
    Prefix fitting is unchanged. After initialization, each training step uses
    detached eligible non-panel rows from both views. Empty local shards still
    participate in all reductions; an empty global population leaves state intact.
    """

    def __init__(self, dim, momentum=0.01):
        super().__init__(dim)
        if momentum != 0.01:
            raise ValueError("Protocol v2 fixes EMA momentum at 0.01")
        self.register_buffer(
            "ema_momentum", torch.tensor(momentum, dtype=torch.float64)
        )
        self.register_buffer("ema_updates", torch.tensor(0))
        self.register_buffer("ema_empty_steps", torch.tensor(0))

    def validate(self):
        if (
            not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or (self.std <= 0).any()
        ):
            raise RuntimeError("Nonfinite/zero EMA normalizer scale")

    @torch.no_grad()
    def update(self, x):
        if not self.frozen:
            raise RuntimeError("EMA requires initialized prefix moments")
        self.validate()
        x = x.detach().double()
        n = AllReduceSum.apply(x.new_tensor(len(x)))
        total = AllReduceSum.apply(x.sum(0))
        squares = AllReduceSum.apply(x.square().sum(0))
        if n == 0:
            self.ema_empty_steps.add_(1)
            return
        mean = total / n
        variance = squares / n - mean.square()
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise RuntimeError("Nonfinite EMA population moments")
        sd = variance.clamp_min(0).sqrt()
        momentum = self.ema_momentum
        self.mean.copy_((1 - momentum) * self.mean.double() + momentum * mean)
        self.std.copy_((1 - momentum) * self.std.double() + momentum * sd)
        self.validate()
        self.ema_updates.add_(1)

    def forward(self, x):
        self.validate()
        return super().forward(x)

    @torch.no_grad()
    def audit(self, x):
        self.validate()
        result = super().audit(x, enforce_drift=False)
        result["drift_within_v1_threshold"] = result["valid"]
        result.update(
            valid=True,
            drift_diagnostic_only=True,
            ema_updates=int(self.ema_updates),
            ema_empty_steps=int(self.ema_empty_steps),
        )
        return result


class Calibration:
    """Binding SPEC §8, 2026-10-24 15:40: base-only pilot, frozen audit CE gate."""

    grid = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
    grid_version = "v3"
    grid_policy = "SPEC-8-2026-10-24-21:15-grid-v3"

    def __init__(self):
        self.points = []

    @staticmethod
    def is_measurement(step):
        return 275 <= step <= 1250 and step % 25 == 0

    def record(self, step, ratio, audit_ce, prior_ce, prior_counts=None):
        if not self.is_measurement(step) or any(p["step"] == step for p in self.points):
            raise ValueError("Unregistered/repeated calibration measurement")
        if prior_ce is None and step >= 1150:
            raise RuntimeError("Unsupported prior in required final-five audit")
        statistics = (
            (ratio, audit_ce) if prior_ce is None else (ratio, audit_ce, prior_ce)
        )
        if not all(math.isfinite(x) and x >= 0 for x in statistics):
            raise RuntimeError("Nonfinite/invalid calibration statistic")
        self.points.append(
            dict(step=step, ratio=ratio, audit_ce=audit_ce, prior_ce=prior_ce)
        )
        if prior_counts is not None:
            self.points[-1]["head_prior_counts"] = prior_counts

    def finish(self):
        if [p["step"] for p in self.points] != list(range(275, 1251, 25)):
            raise RuntimeError(
                "Incomplete calibration; all prescribed measurements required"
            )
        for point in self.points:
            values = [point["ratio"], point["audit_ce"]]
            if point["step"] >= 1150:
                values.append(point["prior_ce"])
            if any(v is None or not math.isfinite(v) or v < 0 for v in values):
                raise RuntimeError("Nonfinite/invalid calibration statistic or prior")
        last = self.points[-5:]
        ce = float(np.mean([p["audit_ce"] for p in last]))
        prior = float(np.mean([p["prior_ce"] for p in last]))
        if not prior > 0 or ce > 0.95 * prior:
            raise RuntimeError(f"Frozen audit CE gate failed: {ce} > .95 * {prior}")
        ratio = float(np.median([p["ratio"] for p in self.points]))
        candidates = [a for a in self.grid if 0.05 <= a * ratio <= 0.20]
        if not candidates:
            raise RuntimeError(
                "No registered positive lambda meets the gradient budget"
            )
        selected = min(candidates, key=lambda a: (abs(math.log(a * ratio / 0.10)), a))
        return dict(
            accepted=True,
            lambda_max=selected,
            median_unit_alpha_ratio=ratio,
            audit_ce=ce,
            prior_ce=prior,
            points=self.points,
            policy="SPEC-8-2026-10-24-15:40",
            grid_version=self.grid_version,
            grid_policy=self.grid_policy,
            base_only_pilot=True,
        )

    def state_dict(self):
        return dict(points=self.points)

    def load_state_dict(self, state):
        self.points = state["points"]
