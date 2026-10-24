"""Trainer-facing contracts, including optional-head setup and resource lifetime."""

from types import SimpleNamespace
import copy

import torch
from omegaconf import OmegaConf

from src.trainers.contrastive import ContrastiveTrainer


def test_common_scaler_is_loaded_even_after_a_resumed_prefix_changes_worker_draws():
    from src.debias import FrozenNormalizer
    from src.debias_runtime import freeze_common_normalizer

    reference = FrozenNormalizer(2)
    reference.observe(torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    reference.freeze()
    resumed = FrozenNormalizer(2)
    resumed.observe(torch.tensor([[100.0, 10.0], [102.0, 12.0]]))
    freeze_common_normalizer(resumed, reference.state_dict())
    for name, value in reference.state_dict().items():
        assert torch.equal(resumed.state_dict()[name], value)


def test_head_broadcast_precedes_rank_reseed_and_has_own_optimizer(monkeypatch):
    trainer = ContrastiveTrainer.__new__(ContrastiveTrainer)
    trainer.cfg = OmegaConf.create(
        dict(
            seed=42,
            model={"ctx_duration": 1},
            views={"duration": 1},
            target_steps=None,
            ckpt={"resume": False},
            debias={"enabled": True},
        )
    )
    trainer.rank, trainer.is_distributed = 1, True
    trainer.train_loader = SimpleNamespace(dataset=SimpleNamespace())
    trainer.model, trainer.loss = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    events = []
    trainer.debias = SimpleNamespace(wrap=lambda t: events.append("head"))
    for name in (
        "_setup_model",
        "_setup_train_loader",
        "_setup_val_loader",
        "_setup_embedding_collectors",
        "_setup_evaluator",
        "_setup_clean_monitor_loader",
        "_setup_optimizer",
        "_setup_debias",
    ):
        monkeypatch.setattr(trainer, name, lambda: None)
    monkeypatch.setattr(trainer, "_apply_ckpt", lambda _: None)
    monkeypatch.setattr(trainer, "make_ddp", lambda m: (events.append("base") or m))
    monkeypatch.setattr(
        "src.trainers.contrastive.seed_everything", lambda _: events.append("reseed")
    )
    trainer.setup(None)
    assert events == ["base", "base", "head", "reseed"]


def test_pilot_never_applies_measured_alpha_and_full_run_ramps_after_common_prefix():
    from src.debias_runtime import encoder_alpha, clip_with_stats

    cfg = OmegaConf.create(
        dict(
            pilot=True,
            lambda_max=0.01,
            normalization_fit_steps=125,
            head_warmup_steps=125,
            encoder_ramp_steps=1250,
        )
    )
    assert all(encoder_alpha(cfg, step) == 0 for step in range(1, 1251))
    cfg.pilot = False
    assert encoder_alpha(cfg, 250) == 0
    assert encoder_alpha(cfg, 875) == 0.005
    assert encoder_alpha(cfg, 1500) == 0.01
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.full((3,), 100.0)
    stats = clip_with_stats([p], 1)
    assert stats["active"] == 1 and stats["scale"] < 0.01
    assert torch.linalg.vector_norm(p.grad) <= 1.000001


def test_audit_close_releases_dataset_and_model_without_cycles():
    import gc
    import weakref
    from src.debias_runtime import AuditWindows

    class Data:
        def _close_open_files(self):
            self.closed = True

    data = Data()
    model = torch.nn.Linear(2, 2)
    data_ref, model_ref = weakref.ref(data), weakref.ref(model)
    wrapper = AuditWindows(data, model.forward, [])
    wrapper.close()
    assert data.closed
    del data, model
    gc.collect()
    assert data_ref() is None and model_ref() is None


def test_disabled_epoch_is_bit_identical_to_pushed_d0b2c03():
    import subprocess
    import numpy as np
    import types

    source = subprocess.check_output(
        [
            "git",
            "show",
            "d0b2c03473af170195864cb2305fedb279b50e9d:src/trainers/contrastive.py",
        ],
        cwd=__import__("pathlib").Path(__file__).resolve().parents[1],
        text=True,
    )
    old = types.ModuleType("off_oracle")
    exec(compile(source, "<d0b2c03 oracle>", "exec"), old.__dict__)

    class Encoder(torch.nn.Sequential):
        def forward(self, bins):
            return super().forward(bins)

    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projector = torch.nn.Linear(4, 3)

        def forward(self, a, b, batch, **kwargs):
            return (self.projector(a) - self.projector(b)).square().mean()

    class Batch:
        def __init__(self):
            self.view1 = SimpleNamespace(
                data={"enc_input": {"bins": torch.arange(20.0).reshape(4, 5) / 10}},
                unit_ids=np.array(["a", "b", "c", "d"]),
            )
            self.view2 = SimpleNamespace(
                data={"enc_input": {"bins": torch.arange(20.0).reshape(4, 5) / 11}}
            )

        def to(self, device):
            return self

    class Loader(list):
        sampler = SimpleNamespace(set_epoch=lambda _: None)

    results = []
    for cls in (old.ContrastiveTrainer, ContrastiveTrainer):
        torch.manual_seed(42)
        t = cls.__new__(cls)
        t.cfg = OmegaConf.create(
            {
                "debias": {"enabled": False},
                "target_steps": 3,
                "max_steps": None,
                "grad_clip": {"enable": True, "max_norm": 1.0},
            }
        )
        t.model = Encoder(torch.nn.Linear(5, 4), torch.nn.Dropout(0.2))
        t.loss = Loss()
        t.device = torch.device("cpu")
        t.train_step = 0
        t.epoch = 0
        t.optimizer = torch.optim.AdamW(
            list(t.model.parameters()) + list(t.loss.parameters())
        )
        t.train_loader = Loader([Batch() for _ in range(3)])
        t.train_emb_collector = SimpleNamespace(
            reset=lambda: None, update=lambda *a: None
        )
        t.logger = SimpleNamespace(get_pbar=lambda data, _: data)
        for name in (
            "lr_step",
            "log_weight_stats",
            "log_grad_stats",
            "log_lr",
            "push_logs",
        ):
            setattr(t, name, lambda: None)
        from contextlib import nullcontext

        t.autocast = nullcontext
        t.train_epoch()
        results.append(
            (
                copy.deepcopy(t.model.state_dict()),
                copy.deepcopy(t.loss.state_dict()),
                copy.deepcopy(t.optimizer.state_dict()),
                torch.get_rng_state(),
            )
        )

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b

    equal(*results)


def test_enabled_cpu_training_checkpoint_and_resume(
    synthetic_export, tmp_path, monkeypatch
):
    # The trainer logs code relative to cwd. Keep that capture inside this test's
    # temporary directory instead of scanning a caller's project/evidence tree.
    monkeypatch.chdir(tmp_path)
    import json
    from pathlib import Path
    import pandas as pd
    import wandb
    from hydra import compose, initialize_config_dir

    data = tmp_path / "data"
    data.mkdir()
    import shutil
    import h5py

    shutil.copytree(synthetic_export / "processed", data / "processed")
    with h5py.File(data / "processed/euler_allgcl_2025/synthetic_train.h5", "r+") as h5:
        h5["blocks/condition_class"][:] = ["control"] * 4
    (data / "splits.json").write_bytes((synthetic_export / "splits.json").read_bytes())
    frame = pd.read_parquet(synthetic_export / "units.parquet")
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
        prior={"split_group": {"0": 1.0}},
    )
    (data / "ledger.json").write_text(json.dumps(ledger))
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
                f"output_dir={tmp_path}/run",
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
                "debias.enabled=true",
                f"debias.ledger_file={data}/ledger.json",
                "debias.normalization_fit_steps=1",
                "debias.head_warmup_steps=0",
            ],
        )
    trainer = ContrastiveTrainer(cfg, 0, 1)
    base_params = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    head_params = {
        id(p)
        for group in trainer.debias.optimizer.param_groups
        for p in group["params"]
    }
    assert not base_params & head_params
    assert head_params == {id(p) for p in trainer.debias.head.parameters()}
    assert not hasattr(trainer.debias.head, "init_recording_classifier")
    trainer.train()
    assert trainer.train_step == 3
    assert trainer.debias.optimizer.param_groups[0]["lr"] == 1e-4
    assert trainer.optimizer.param_groups[0]["lr"] != 1e-4
    best = json.loads((tmp_path / "run/best.json").read_text())
    checkpoint = torch.load(tmp_path / "run" / best["ckpt"], weights_only=False)
    assert (
        "debias_head_state_dict" in checkpoint
        and "debias_optimizer_state_dict" in checkpoint
    )
    assert trainer.debias.audit_loader is None
    from src.debias_export import audit_checkpoint

    audit = audit_checkpoint(
        tmp_path / "run" / best["ckpt"], tmp_path / "export_audit.json", "cpu"
    )
    assert audit["status"] == "audited" and audit["valid"] is True
    # Validation never calls the adversary and remains the native contrastive loss.
    original = trainer.debias.head.forward

    def forbidden(*args, **kwargs):
        raise AssertionError("Head called during contrastive validation")

    trainer.debias.head.forward = forbidden
    trainer.val_epoch()
    assert trainer.last_val_loss == best["val_loss"]
    trainer.debias.head.forward = original
    head = copy.deepcopy(trainer.debias.head.state_dict())
    moments = copy.deepcopy(trainer.debias.optimizer.state_dict())
    wandb.finish()
    cfg.max_steps = 4
    resumed = ContrastiveTrainer(cfg, 0, 1)
    assert resumed.train_step == 3
    for name, value in head.items():
        assert torch.equal(resumed.debias.head.state_dict()[name], value)
    for key, state in moments["state"].items():
        for name, value in state.items():
            assert torch.equal(
                resumed.debias.optimizer.state_dict()["state"][key][name], value
            )
    resumed.train()
    assert resumed.train_step == 4
    wandb.finish()
    # Exercise the actual library checkpoint loader onto a distinct non-CPU
    # device without allocating a GPU. Meta verifies placement; numerical state
    # equality is covered by the real CPU resume checks above.
    import warnings
    from fixtures.device_checks import assert_optimizer_devices, assert_module_device

    runtime = resumed.debias
    from src.debias_runtime import DebiasRuntime

    # Also exercise the real constructor's .to(device) calls on a meta device,
    # so a future omitted normalizer/head/prior transfer fails on CPU CI.
    meta_cfg = copy.deepcopy(resumed.cfg)
    meta_cfg.output_dir = str(tmp_path / "meta_placement")
    meta_trainer = SimpleNamespace(
        cfg=meta_cfg,
        device=torch.device("meta"),
        rank=0,
        world_size=1,
        model=SimpleNamespace(emb_dim=cfg.model.dim, tokenize=lambda _: None),
        add_checkpoint_items=lambda **kwargs: None,
    )
    meta_runtime = DebiasRuntime(meta_trainer)
    try:
        for module in (meta_runtime.head, meta_runtime.normalizer, meta_runtime.prior):
            assert_module_device(module, "meta")
    finally:
        meta_runtime.close()
    runtime.head.to("meta")
    runtime.normalizer.to("meta")
    runtime.prior.to("meta")
    runtime.optimizer = torch.optim.AdamW(
        runtime.head.parameters(), lr=1e-4, weight_decay=0
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*copying from a non-meta parameter.*meta parameter.*"
        )
        runtime.load_checkpoint(checkpoint)
    assert_optimizer_devices(runtime.optimizer)
    for module in (runtime.head, runtime.normalizer, runtime.prior):
        assert_module_device(module, "meta")
    for state in runtime.optimizer.state.values():
        assert state["step"].device.type == "cpu"
    # A blanket CPU snapshot must never conceal a misplaced live moment/buffer.
    import pytest

    param = next(runtime.head.parameters())
    runtime.optimizer.state[param]["exp_avg"] = torch.zeros(param.shape)
    with pytest.raises(AssertionError, match="exp_avg device"):
        assert_optimizer_devices(runtime.optimizer)
    runtime.normalizer.mean = torch.zeros_like(runtime.normalizer.mean, device="cpu")
    with pytest.raises(AssertionError, match="mean device"):
        assert_module_device(runtime.normalizer, "meta")


def test_panel_audit_and_noncontrol_rows_have_no_auxiliary_gradient_in_either_view():
    import numpy as np
    import pandas as pd
    from src.debias import DebiasHead, DebiasPopulation, FrozenNormalizer
    from src.debias_runtime import DebiasRuntime

    rows = []
    for sid, animal in [
        ("fit1", "a"),
        ("fit2", "b"),
        ("audit", "a"),
        ("panel", "Experimenter_2_2023-02-14"),
    ]:
        rows.append(
            dict(
                unit_id=sid,
                session_id=sid,
                split_group=animal,
                split="train",
                experimenter="Experimenter 2",
                setupid=1,
                chirp_qidx=0.8,
                bar_qidx=0.8,
                supergroup="ON",
                label_condition="C1",
                label_condition_is_drug=False,
            )
        )
    frame = pd.DataFrame(rows)
    runtime = DebiasRuntime.__new__(DebiasRuntime)
    runtime.cfg = OmegaConf.create({"pilot": True})
    runtime.population = DebiasPopulation(
        frame, "split_group", {"audit"}, {"a": 2, "b": 1}
    )
    runtime.head = DebiasHead(4, 2)
    runtime.normalizer = FrozenNormalizer(4)
    runtime.normalizer.frozen.fill_(True)
    views = [
        SimpleNamespace(
            unit_ids=frame.unit_id.to_numpy(),
            recording_ids=frame.session_id.to_numpy(),
            unit_seqlen=[1, 1, 1, 1],
            debias_control=np.array(control),
        )
        for control in ([True, False, True, True], [True, True, True, True])
    ]
    batch = SimpleNamespace(view1=views[0], view2=views[1], batch_size=4)
    ys = [torch.randn(4, 4, requires_grad=True) for _ in range(2)]
    aux, _, _ = runtime.head_loss(ys, batch, alpha=1.0)
    gradients = torch.autograd.grad(aux, ys)
    assert gradients[0][0].abs().sum() > 0 and gradients[0][1].abs().sum() == 0
    assert (gradients[1][:2].abs().sum(1) > 0).all()
    for gradient in gradients:
        assert gradient[2:].abs().sum() == 0
