"""Cheap, split-safe All-GCL monitoring. Quality rule matches export_all_rgcs_fd.ipynb."""

import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import ThreadpoolController
from .rankme import rankme


class AllGCLEvaluator:
    def __init__(self, units_file, n_units_subsample=5000, seed=0, world_size=1):
        # Arrow applies the split predicate before materializing labels in Python.
        md = pd.read_parquet(units_file, filters=[("split", "in", ["train", "valid"])])
        key = "unit_id" if "unit_id" in md else "id"
        md[key] = md[key].astype(str)
        self.md = md.set_index(key, verify_integrity=True)
        self.selected = {}
        self.threadpools = ThreadpoolController()
        rng = np.random.default_rng(seed)
        for split in ("train", "valid"):
            ids = np.sort(self.md.index[self.md.split == split].to_numpy())
            if len(ids) > n_units_subsample:
                ids = rng.choice(ids, n_units_subsample, replace=False)
            self.selected[split] = set(ids)

    def __call__(self, X_train, train_unit_ids, X_val, val_unit_ids):
        started = time.perf_counter()
        arrays, metadata, metrics = {}, {}, {}
        for split, prefix, x, ids in (
            ("train", "train", X_train, train_unit_ids),
            ("valid", "val", X_val, val_unit_ids),
        ):
            ids = np.asarray(ids).astype(str)
            x = (
                x.detach().cpu().float().numpy()
                if isinstance(x, torch.Tensor)
                else np.asarray(x)
            )
            mask = np.array([uid in self.selected[split] for uid in ids]) & np.isfinite(
                x
            ).all(axis=1)
            arrays[split] = x[mask]
            metadata[split] = self.md.loc[ids[mask]]
            metrics[f"{prefix}/n_units"] = int(mask.sum())
            metrics[f"{prefix}/rankme"] = (
                rankme(torch.from_numpy(x[mask]).float())
                if mask.sum() >= 2
                else float("nan")
            )
        with self.threadpools.limit(limits=1):
            for target in ("group_id", "supergroup"):
                selected = {}
                for split in ("train", "valid"):
                    md = metadata[split]
                    quality = (pd.to_numeric(md.chirp_qidx, errors="coerce") > 0.35) | (
                        pd.to_numeric(md.bar_qidx, errors="coerce") > 0.6
                    )
                    labels = (
                        md[target]
                        if target in md
                        else pd.Series(np.nan, index=md.index)
                    )
                    selected[split] = (
                        quality
                        & labels.notna()
                        & ~labels.astype(str)
                        .str.lower()
                        .isin(["", "nan", "none", "unknown"])
                    )
                train_mask, valid_mask = selected["train"], selected["valid"]
                if not train_mask.any() or not valid_mask.any():
                    continue
                y_train = (
                    metadata["train"].loc[train_mask, target].astype(str).to_numpy()
                )
                if len(np.unique(y_train)) < 2:
                    continue
                probe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        solver=(
                            "liblinear"
                            if len(y_train) < arrays["train"].shape[1]
                            else "lbfgs"
                        ),
                        max_iter=100,
                        random_state=0,
                    ),
                )
                probe.fit(arrays["train"][train_mask], y_train)
                pred = probe.predict(arrays["valid"][valid_mask])
                y_valid = (
                    metadata["valid"].loc[valid_mask, target].astype(str).to_numpy()
                )
                metrics[f"val/{target}_balanced_accuracy"] = float(
                    balanced_accuracy_score(y_valid, pred)
                )
        # Timing is kept separate from deterministic scientific metrics.
        self.elapsed_seconds = time.perf_counter() - started
        return metrics
