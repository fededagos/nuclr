"""Deterministic SPEC 4.1 fixture. Generated binaries stay outside the checkout."""

import argparse
import datetime
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries
from brainsets.core import serialize_fn_map

BRAINSET = "euler_allgcl_2025"
RATE = 7.8125


def make_synthetic(output: Path, block_duration=120.0, valid_conditions=None):
    output = Path(output)
    processed = output / "processed" / BRAINSET
    processed.mkdir(parents=True, exist_ok=True)
    rows, condition_rows = [], []
    sessions = {"train": ["synthetic_train"], "valid": ["synthetic_valid"], "test": []}
    rng = np.random.default_rng(42)
    for split in ("train", "valid"):
        sid = sessions[split][0]
        date = "2026-01-01" if split == "train" else "2026-01-02"
        ids = np.array([f"{sid}_roi{i}" for i in range(40)])
        # Every condition has both stimuli; each 120 s block is separated by 60 s.
        start = np.arange(4, dtype=float) * (block_duration + 60.0)
        end = start + block_duration
        conditions = np.array(
            valid_conditions
            if split == "valid" and valid_conditions is not None
            else ["c1", "c1", "d", "d"]
        )
        blocks = Interval(
            start=start,
            end=end,
            condition=conditions,
            condition_raw=np.char.upper(conditions),
            condition_class=np.array(
                [
                    (
                        "washout"
                        if c == "wo"
                        else "control" if c.startswith("c") else "drug"
                    )
                    for c in conditions
                ]
            ),
            stimulus=np.array(["chirp", "movingbar", "chirp", "movingbar"]),
        )
        timestamps = np.concatenate(
            [
                s + np.arange(int(np.ceil(block_duration * RATE - 1e-8))) / RATE
                for s in start
            ]
        )
        values = rng.normal(size=(len(timestamps), 40)).astype("float32")
        # Distinct signal per unit makes ordering bugs visible.
        values += np.arange(40, dtype="float32")[None, :] / 40
        meta = pd.DataFrame(
            dict(
                id=ids,
                unit_id=ids,
                session_id=sid,
                split=split,
                experimenter="Experimenter synthetic",
                date=date,
                exp_num=1,
                field="GCL0",
                roi_id=np.arange(40),
                cluster_id=np.arange(40) % 2 + 1,
                group_id=np.arange(40) % 2 + 1,
                supergroup=np.where(np.arange(40) % 2, "OFF", "ON"),
                chirp_qidx=0.8,
                bar_qidx=0.8,
                label_condition="C1",
                label_condition_is_drug=False,
                prob_cluster=0.95,
                prob_group=0.95,
                prob_supergroup=0.95,
                prob_class=0.95,
                d_qi=np.nan,
                roi_size_um2=np.nan,
                ventral_dorsal_pos_um=np.nan,
                temporal_nasal_pos_um=np.nan,
                in_preprint_table=False,
            )
        )
        rows.append(meta)
        for condition in np.unique(conditions):
            condition_meta = meta.copy()
            condition_meta["condition"] = condition
            condition_meta["condition_raw"] = condition.upper()
            condition_meta["condition_class"] = (
                "washout"
                if condition == "wo"
                else "control" if condition.startswith("c") else "drug"
            )
            condition_rows.append(condition_meta)
        units = ArrayDict(**{c: meta[c].to_numpy() for c in meta.columns})
        data = Data(
            brainset=Data(
                id=BRAINSET,
                origin_version="synthetic",
                derived_version="0.0.0",
                source="synthetic",
                description="SPEC 4.1 test fixture",
            ),
            subject=Data(
                id=f"retina_{split}",
                species="MUS_MUSCULUS",
                sex="UNKNOWN",
                age=100.0,
                genotype="synthetic",
            ),
            session=Data(
                id=sid,
                recording_date=datetime.datetime.fromisoformat(date),
                split=split,
                experimenter="Experimenter synthetic",
                exp_num=1,
                region="LR",
                field="GCL0",
                eye="left",
                setup="synthetic",
                orientation="synthetic",
                project="synthetic",
                missing_presentation_sequence=False,
                mask_table_mismatch=False,
            ),
            device=Data(
                id="synthetic_2p",
                recording_tech="TWO_PHOTON_IMAGING",
                indicator="OGB-1",
                imaging_rate=RATE,
                pixel_size_um=np.nan,
                target_area="retina GCL",
            ),
            units=units,
            blocks=blocks,
            calcium_traces=IrregularTimeSeries(
                timestamps=timestamps,
                df_over_f=values,
                df_over_f_norm=(values / np.percentile(values, 99)).astype("float32"),
                domain=Interval(start=start, end=end),
            ),
            domain=Interval(start=start, end=end),
        )
        with h5py.File(processed / f"{sid}.h5", "w") as f:
            data.to_hdf5(f, serialize_fn_map=serialize_fn_map)
    pd.concat(rows).to_parquet(output / "units.parquet", index=False)
    pd.concat(condition_rows).to_parquet(
        output / "unit_conditions.parquet", index=False
    )
    (output / "splits.json").write_text(
        json.dumps(
            dict(
                seed=42,
                fractions=[0.7, 0.15, 0.15],
                group_key=["experimenter", "date"],
                groups={"retina_train": "train", "retina_valid": "valid"},
                sessions=sessions,
            )
        )
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    make_synthetic(parser.parse_args().output)
