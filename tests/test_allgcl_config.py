from pathlib import Path
import json
from hydra import compose, initialize_config_dir
from torch_brain.data import Dataset


def allgcl_config(export, output):
    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()), version_base="1.2"
    ):
        return compose(
            config_name="train_allgcl",
            overrides=[
                f"data.data_root={export}/processed",
                f"data.splits_file={export}/splits.json",
                f"data.units_file={export}/units.parquet",
                f"output_dir={output}",
                "device=cpu",
                "num_workers=0",
                "batch_size=2",
            ],
        )


def test_train_selection_only_train_sessions(synthetic_export, tmp_path):
    from src.allgcl import prepare_allgcl_config

    cfg = allgcl_config(synthetic_export, tmp_path / "run")
    prepare_allgcl_config(cfg)
    splits = json.loads((synthetic_export / "splits.json").read_text())["sessions"]
    for key, split in [("train_dataset", "train"), ("val_dataset", "valid")]:
        ds = Dataset(
            root=cfg.data_root, config=cfg.data[key], session_id_prefix_fn=lambda _: ""
        )
        assert sorted(ds.get_session_ids()) == sorted(splits[split])


def test_wandb_offline(synthetic_export, tmp_path):
    from src.allgcl import prepare_allgcl_config

    cfg = allgcl_config(synthetic_export, tmp_path / "run")
    cfg.wandb.mode = "online"
    prepare_allgcl_config(cfg)
    assert cfg.wandb.mode == "offline"
    assert Path(cfg.wandb.dir).resolve().is_relative_to((tmp_path / "run").resolve())
    assert Path(cfg.ckpt.dir).resolve() == (tmp_path / "run/ckpt").resolve()


def test_resume_picks_latest(synthetic_export, tmp_path):
    import torch
    import wandb
    from src.checkpoint import save_ckpt
    from src.trainers.contrastive import ContrastiveTrainer

    cfg = allgcl_config(synthetic_export, tmp_path / "run")
    cfg.model.dim = 32
    cfg.model.self_heads = 2
    cfg.model.dim_head = 16
    cfg.model.t_layers = cfg.model.st_layers = 0
    cfg.precision = "fp32"
    cfg.data.evaluator = None
    cfg.target_steps = 32
    trainer = ContrastiveTrainer(cfg, rank=0, world_size=1)
    for epoch in (1, 2, 3):
        for param in trainer.model.parameters():
            param.grad = torch.ones_like(param)
        trainer.optimizer.step()
        trainer.scheduler.step()
        save_ckpt(
            Path(cfg.ckpt.dir) / f"epoch_{epoch}.pt",
            epoch=epoch,
            train_step=epoch * 4,
            run_id=trainer.logger.run_id,
            cfg=cfg,
            model=trainer.model,
            loss=trainer.loss,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
        )
    optimizer_state = trainer.optimizer.state_dict()
    scheduler_state = trainer.scheduler.state_dict()
    wandb.finish()
    resumed = ContrastiveTrainer(cfg, rank=0, world_size=1)
    assert resumed.epoch == 4 and resumed.train_step == 12
    assert resumed.scheduler.state_dict() == scheduler_state
    for k, state in optimizer_state["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                resumed.optimizer.state_dict()["state"][k][name], value
            )
    assert (
        resumed.optimizer.param_groups[0]["lr"]
        == trainer.optimizer.param_groups[0]["lr"]
    )
    wandb.finish()


def test_cpu_train_writes_best_and_respects_step_limit(synthetic_export, tmp_path):
    import torch
    import wandb
    from src.trainers.contrastive import ContrastiveTrainer

    torch.set_num_threads(1)
    cfg = allgcl_config(synthetic_export, tmp_path / "run")
    cfg.model.dim = 32
    cfg.model.self_heads = 2
    cfg.model.dim_head = 16
    cfg.model.t_layers = cfg.model.st_layers = 0
    cfg.precision = "fp32"
    cfg.data.evaluator = None
    cfg.target_steps = 2
    trainer = ContrastiveTrainer(cfg, rank=0, world_size=1)
    assert trainer.scheduler.base_lrs == [cfg.lr.max] * len(
        trainer.optimizer.param_groups
    )
    trainer.train()
    assert trainer.train_step == 2
    best = json.loads((Path(cfg.output_dir) / "best.json").read_text())
    assert best["ckpt"] == f"ckpt/epoch_{best['epoch']}.pt"
    ckpt = torch.load(
        Path(cfg.output_dir) / best["ckpt"], map_location="cpu", weights_only=False
    )
    assert best["epoch"] == ckpt["epoch"]
    assert isinstance(best["val_loss"], float)
    assert ckpt["best_val_loss"] == best["val_loss"]
    # Simulate interruption after the atomic checkpoint but before best.json.
    (Path(cfg.output_dir) / "best.json").unlink()
    wandb.finish()
    resumed = ContrastiveTrainer(cfg, rank=0, world_size=1)
    assert resumed.train_step == 2
    assert json.loads((Path(cfg.output_dir) / "best.json").read_text()) == best
    wandb.finish()


def test_configured_evaluator_resolves_paths(synthetic_export, tmp_path):
    import numpy as np
    import pandas as pd
    import torch
    from src.allgcl import prepare_allgcl_config
    from src.utils import instantiate

    cfg = allgcl_config(synthetic_export, tmp_path)
    prepare_allgcl_config(cfg)
    evaluator = instantiate(cfg.data.evaluator, world_size=1)
    md = pd.read_parquet(synthetic_export / "units.parquet")
    x = torch.tensor(np.eye(2)[np.arange(40) % 2], dtype=torch.float32)
    metrics = evaluator(x, md[md.split == "train"].id, x, md[md.split == "valid"].id)
    assert metrics["val/group_id_balanced_accuracy"] == 1.0


def test_validation_pairs_repeat_across_passes_and_resume(synthetic_export, tmp_path):
    import hashlib
    import torch
    import wandb
    from src.trainers.contrastive import ContrastiveTrainer

    cfg = allgcl_config(synthetic_export, tmp_path / "deterministic")
    cfg.model.dim = 32
    cfg.model.self_heads = 2
    cfg.model.dim_head = 16
    cfg.model.t_layers = cfg.model.st_layers = 0
    cfg.precision = "fp32"
    cfg.data.evaluator = None
    cfg.target_steps = 2
    cfg.lr.max = 0.0  # Freeze weights: loss should depend only on these fixed weights.
    cfg.num_workers = 2  # Exercise persistent worker RNG state, as in production.

    def pairs(loader):
        result = []
        for batch in loader:
            result.append(
                tuple(
                    (
                        tuple(view.unit_ids),
                        hashlib.sha256(
                            view.data["enc_input"]["bins"].numpy().tobytes()
                        ).hexdigest(),
                    )
                    for view in (batch.view1, batch.view2)
                )
            )
        return result

    try:
        trainer = ContrastiveTrainer(cfg, rank=0, world_size=1)
        first = pairs(trainer.val_loader)
        assert first == pairs(trainer.val_loader)
        trainer.val_epoch()
        initial_loss = trainer.last_val_loss
        trainer.val_epoch()
        assert trainer.last_val_loss == initial_loss
        trainer.train()
        assert trainer.last_val_loss == initial_loss
        wandb.finish()
        resumed = ContrastiveTrainer(cfg, rank=0, world_size=1)
        assert resumed.train_step == 2
        assert first == pairs(resumed.val_loader)
        resumed.val_epoch()
        assert resumed.last_val_loss == initial_loss
    finally:
        wandb.finish()


def test_checkpoint_retention_and_relocated_legacy_resume(synthetic_export, tmp_path):
    import shutil
    import torch
    import wandb
    from src.trainers.contrastive import ContrastiveTrainer

    def config(output, max_steps):
        cfg = allgcl_config(synthetic_export, output)
        cfg.model.dim = 32
        cfg.model.self_heads = 2
        cfg.model.dim_head = 16
        cfg.model.t_layers = cfg.model.st_layers = 0
        cfg.precision = "fp32"
        cfg.data.evaluator = None
        cfg.target_steps = 32
        cfg.max_steps = max_steps
        cfg.lr.max = 0.0  # Fixed weights and pairs keep the best at epoch zero.
        return cfg

    original = tmp_path / "original"
    try:
        trainer = ContrastiveTrainer(config(original, 12), rank=0, world_size=1)
        trainer.train()
        assert trainer.train_step == 12
        best = json.loads((original / "best.json").read_text())
        assert best["epoch"] == 0
        checkpoints = list((original / "ckpt").glob("epoch_*.pt"))
        latest = max(checkpoints, key=lambda p: int(p.stem[6:]))
        assert int(latest.stem[6:]) > best["epoch"]
        assert {p.name for p in checkpoints} == {latest.name, Path(best["ckpt"]).name}
        scheduler = trainer.scheduler.state_dict()
        wandb.finish()

        # Relocate a run whose latest checkpoint contains the former absolute format.
        # Leave best.json absent, as after a kill between checkpoint/JSON publication.
        moved = tmp_path / "moved"
        shutil.copytree(original / "ckpt", moved / "ckpt")
        payload = torch.load(
            moved / "ckpt" / latest.name, map_location="cpu", weights_only=False
        )
        payload["best_record"]["ckpt"] = str(original / best["ckpt"])
        torch.save(payload, moved / "ckpt" / latest.name)
        shutil.rmtree(original / "ckpt")
        resumed = ContrastiveTrainer(config(moved, 20), rank=0, world_size=1)
        assert resumed.train_step == 12
        assert resumed.scheduler.state_dict() == scheduler
        assert json.loads((moved / "best.json").read_text()) == best
        assert (moved / best["ckpt"]).is_file()
        resumed.train()
        assert resumed.train_step == 20
        assert json.loads((moved / "best.json").read_text()) == best
        remaining = list((moved / "ckpt").glob("epoch_*.pt"))
        newest = max(remaining, key=lambda p: int(p.stem[6:]))
        assert int(newest.stem[6:]) > int(latest.stem[6:])
        assert {p.name for p in remaining} == {newest.name, Path(best["ckpt"]).name}
        saved = torch.load(newest, map_location="cpu", weights_only=False)
        assert saved["best_record"] == best
    finally:
        wandb.finish()
