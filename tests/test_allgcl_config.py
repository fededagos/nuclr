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
    ckpt = torch.load(best["ckpt"], map_location="cpu", weights_only=False)
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
