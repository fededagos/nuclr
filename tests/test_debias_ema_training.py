"""Actual trainer: matched EMA bookkeeping leaves the lambda-zero base unchanged."""

import copy
import json
from pathlib import Path
import shutil

import h5py
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
import wandb

from src.trainers.contrastive import ContrastiveTrainer


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def test_matched_ema_control_zero_head_and_bypass_base_updates_and_resume(
    synthetic_export, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    shutil.copytree(synthetic_export, data)
    with h5py.File(data / "processed/euler_allgcl_2025/synthetic_train.h5", "r+") as h5:
        h5["blocks/condition_class"][:] = ["control"] * 4
    frame = pd.read_parquet(data / "units.parquet")
    frame["setupid"] = 1
    frame["experimenter"] = "Experimenter 1"
    frame["split_group"] = "Experimenter_1_2026-01-01"
    frame.to_parquet(data / "units.parquet")
    animal = "Experimenter_1_2026-01-01"
    ledger = dict(
        audit_fields=["synthetic_train"],
        audit_windows=[
            dict(
                recording_id="euler_allgcl_2025/synthetic_train", start=0.0, end=32.768
            )
        ],
        exposure={"split_group": {animal: 10}},
    )
    (data / "ledger.json").write_text(json.dumps(ledger))
    from src.debias_runtime import DebiasRuntime

    original_head_loss = DebiasRuntime.head_loss

    def ordered_head_loss(runtime, *args, **kwargs):
        assert (
            int(runtime.normalizer.ema_updates)
            == runtime.steps_seen - runtime.cfg.normalization_fit_steps + 1
        )
        return original_head_loss(runtime, *args, **kwargs)

    monkeypatch.setattr(DebiasRuntime, "head_loss", ordered_head_loss)
    snapshots = []
    moments = []
    for name, enabled, track in [
        ("bypass", False, False),
        ("control", False, True),
        ("zero_head", True, True),
    ]:
        with initialize_config_dir(
            config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
            version_base="1.2",
        ):
            cfg = compose(
                config_name="train_allgcl",
                overrides=[
                    f"data.data_root={data}/processed",
                    f"data.splits_file={data}/splits.json",
                    f"data.units_file={data}/units.parquet",
                    f"output_dir={tmp_path}/{name}",
                    "device=cpu",
                    "precision=fp32",
                    "num_workers=0",
                    "batch_size=2",
                    "target_steps=32",
                    "max_steps=3",
                    "data.evaluator=null",
                    "model.dim=32",
                    "model.self_heads=2",
                    "model.dim_head=16",
                    "model.t_layers=0",
                    "model.st_layers=0",
                    f"debias.enabled={str(enabled).lower()}",
                    f"debias.track_normalizer={str(track).lower()}",
                    f"debias.ledger_file={data}/ledger.json",
                    "debias.normalization_fit_steps=1",
                    "debias.head_warmup_steps=0",
                ],
            )
        trainer = ContrastiveTrainer(cfg, 0, 1)
        trainer.train()
        if track:
            runtime = trainer.debias
            assert int(runtime.normalizer.ema_updates) == 2
            moments.append(copy.deepcopy(runtime.normalizer.state_dict()))
            before = copy.deepcopy(runtime.normalizer.state_dict())
            trainer.val_epoch()
            equal_tree(before, runtime.normalizer.state_dict())
        best = json.loads((Path(cfg.output_dir) / "best.json").read_text())
        saved = torch.load(Path(cfg.output_dir) / best["ckpt"], weights_only=False)
        snapshots.append(saved)
        wandb.finish()
        if name == "control":
            cfg.max_steps = 4
            resumed = ContrastiveTrainer(cfg, 0, 1)
            equal_tree(moments[-1], resumed.debias.normalizer.state_dict())
            # Audits read one fixed snapshot and leave every moment/counter intact.
            resumed.debias.audit(resumed)
            equal_tree(moments[-1], resumed.debias.normalizer.state_dict())
            resumed.train()
            assert int(resumed.debias.normalizer.ema_updates) == 3
            wandb.finish()
    for key in (
        "model_state_dict",
        "loss_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    ):
        equal_tree(snapshots[0][key], snapshots[1][key])
        equal_tree(snapshots[0][key], snapshots[2][key])
    equal_tree(moments[0], moments[1])


def test_real_pilot_measurement_step_updates_ema_once_before_head_and_matches_control(
    synthetic_export, tmp_path, monkeypatch
):
    from src.debias import Calibration
    from src.debias_runtime import DebiasRuntime

    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    shutil.copytree(synthetic_export, data)
    source = data / "processed/euler_allgcl_2025/synthetic_train.h5"
    with h5py.File(source, "r+") as h5:
        h5["blocks/condition_class"][:] = ["control"] * 4
    audit = data / "processed/euler_allgcl_2025/synthetic_audit.h5"
    shutil.copyfile(source, audit)
    with h5py.File(audit, "r+") as h5:
        assert "id" in h5["session"].attrs
        h5["session"].attrs.modify("id", "synthetic_audit")
    frame = pd.read_parquet(data / "units.parquet")
    frame["setupid"] = 1
    frame["experimenter"] = "Experimenter 1"
    frame["split_group"] = "Experimenter_1_2026-01-01"
    audit_rows = frame[frame["split"] == "train"].copy()
    audit_rows["session_id"] = "synthetic_audit"
    pd.concat([frame, audit_rows], ignore_index=True).to_parquet(data / "units.parquet")
    splits = json.loads((data / "splits.json").read_text())
    splits["sessions"]["train"].append("synthetic_audit")
    (data / "splits.json").write_text(json.dumps(splits))
    ledger = dict(
        audit_fields=["synthetic_audit"],
        audit_windows=[
            dict(
                recording_id="euler_allgcl_2025/synthetic_audit", start=0.0, end=32.768
            )
        ],
        exposure={"split_group": {"Experimenter_1_2026-01-01": 20}},
    )
    (data / "ledger.json").write_text(json.dumps(ledger))
    monkeypatch.setattr(
        Calibration, "is_measurement", staticmethod(lambda step: 3 <= step <= 7)
    )
    # Both recipes omit validation, matching the production pilot's RNG schedule.
    monkeypatch.setattr(ContrastiveTrainer, "val_epoch", lambda self: None)
    original_measure = DebiasRuntime.measure
    original_update = DebiasRuntime.update_normalizer
    original_head = DebiasRuntime.head_loss
    events = {}

    def measured(runtime, *args, **kwargs):
        assert kwargs["update_normalizer"] is True
        runtime._test_in_measure = True
        try:
            return original_measure(runtime, *args, **kwargs)
        finally:
            runtime._test_in_measure = False

    def updated(runtime, *args, **kwargs):
        before = int(runtime.normalizer.ema_updates)
        original_update(runtime, *args, **kwargs)
        assert int(runtime.normalizer.ema_updates) == before + 1
        events.setdefault(id(runtime), []).append(
            (runtime.steps_seen + 1, getattr(runtime, "_test_in_measure", False))
        )

    def headed(runtime, *args, **kwargs):
        assert int(runtime.normalizer.ema_updates) == runtime.steps_seen  # fit=1
        return original_head(runtime, *args, **kwargs)

    monkeypatch.setattr(DebiasRuntime, "measure", measured)
    monkeypatch.setattr(DebiasRuntime, "update_normalizer", updated)
    monkeypatch.setattr(DebiasRuntime, "head_loss", headed)
    states = []
    normalizers = []
    for name, pilot in [("control", False), ("pilot", True)]:
        with initialize_config_dir(
            config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
            version_base="1.2",
        ):
            cfg = compose(
                config_name="train_allgcl",
                overrides=[
                    f"data.data_root={data}/processed",
                    f"data.splits_file={data}/splits.json",
                    f"data.units_file={data}/units.parquet",
                    f"output_dir={tmp_path}/{name}",
                    "device=cpu",
                    "precision=fp32",
                    "num_workers=0",
                    "batch_size=4",
                    "target_steps=32",
                    "max_steps=7",
                    "data.evaluator=null",
                    "model.dim=32",
                    "model.self_heads=2",
                    "model.dim_head=16",
                    "model.t_layers=0",
                    "model.st_layers=0",
                    f"debias.enabled={str(pilot).lower()}",
                    f"debias.pilot={str(pilot).lower()}",
                    "debias.track_normalizer=true",
                    f"debias.ledger_file={data}/ledger.json",
                    "debias.normalization_fit_steps=1",
                    "debias.head_warmup_steps=0",
                ],
            )
        trainer = ContrastiveTrainer(cfg, 0, 1)
        trainer.train()
        runtime = trainer.debias
        assert int(runtime.normalizer.ema_updates) == 6
        assert events[id(runtime)] == [
            (step, pilot and step >= 3) for step in range(2, 8)
        ]
        normalizers.append(copy.deepcopy(runtime.normalizer.state_dict()))
        states.append(
            torch.load(trainer.last_ckpt_path, map_location="cpu", weights_only=False)
        )
        if pilot:
            assert [p["step"] for p in runtime.calibration.points] == [3, 4, 5, 6, 7]
            points = copy.deepcopy(runtime.calibration.state_dict())
        wandb.finish()
    equal_tree(normalizers[0], normalizers[1])
    for key in (
        "model_state_dict",
        "loss_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    ):
        equal_tree(states[0][key], states[1][key])
    cfg.max_steps = 8
    resumed = ContrastiveTrainer(cfg, 0, 1)
    equal_tree(normalizers[1], resumed.debias.normalizer.state_dict())
    equal_tree(points, resumed.debias.calibration.state_dict())
    resumed.train()
    assert int(resumed.debias.normalizer.ema_updates) == 7
    wandb.finish()
