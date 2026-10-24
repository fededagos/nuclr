"""Monitoring receives clean windows and does not advance the training RNG."""

import numpy as np
import gc
import weakref
import pytest
import torch
import wandb
from test_allgcl_config import allgcl_config
from src.dataset import get_window, ViewData
from src.evaluators.collector import EmbeddingCollector
from src.trainers.contrastive import ContrastiveTrainer
from src.transforms.calcium_augmentation import CalciumTrainingViews
from torch_brain.data.dataset import DatasetIndex


def monitor_config(export, output):
    cfg = allgcl_config(export, output)
    cfg.train_transform.augmentation = "all"
    cfg.train_transform.min_units = 1000
    cfg.model.dim = 32
    cfg.model.self_heads = 2
    cfg.model.dim_head = 16
    cfg.model.t_layers = cfg.model.st_layers = 0
    cfg.precision = "fp32"
    cfg.target_steps = 2
    return cfg


@pytest.mark.parametrize("fail_training", [False, True])
def test_clean_dataset_released_without_cyclic_gc(
    synthetic_export, tmp_path, monkeypatch, fail_training
):
    import src.trainers.contrastive as contrastive

    torch.set_num_threads(1)
    init = wandb.init
    monkeypatch.setattr(wandb, "init", lambda **kw: init(**{**kw, "mode": "disabled"}))
    dataset_type = contrastive.Dataset
    refs, files = [], []

    def track_dataset(*args, **kwargs):
        dataset = dataset_type(*args, **kwargs)
        refs.append(weakref.ref(dataset))
        files.extend(dataset._open_files.values())
        return dataset

    monkeypatch.setattr(contrastive, "Dataset", track_dataset)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        trainer = ContrastiveTrainer(
            monitor_config(synthetic_export, tmp_path / "release"), rank=0, world_size=1
        )
        if fail_training:

            def fail():
                raise RuntimeError("injected training failure")

            monkeypatch.setattr(trainer, "train_epoch", fail)
            with pytest.raises(RuntimeError, match="injected training failure"):
                trainer.train()
        else:
            trainer.train()
        assert len(refs) == 1
        assert refs[0]() is None, "Clean Dataset survived train() without cyclic GC"
        assert files and all(not file.id.valid for file in files)
    finally:
        if was_enabled:
            gc.enable()
        wandb.finish()


def test_clean_monitor_reuses_setup_dataset_and_loader(
    synthetic_export, tmp_path, monkeypatch
):
    import src.trainers.contrastive as contrastive

    torch.set_num_threads(1)
    init = wandb.init
    monkeypatch.setattr(wandb, "init", lambda **kw: init(**{**kw, "mode": "disabled"}))
    dataset_type = contrastive.Dataset
    refs = []

    def track_dataset(*args, **kwargs):
        dataset = dataset_type(*args, **kwargs)
        refs.append(weakref.ref(dataset))
        return dataset

    monkeypatch.setattr(contrastive, "Dataset", track_dataset)
    trainer = ContrastiveTrainer(
        monitor_config(synthetic_export, tmp_path / "reuse"), rank=0, world_size=1
    )
    try:
        assert len(refs) == 1, "Clean Dataset must be opened once during setup"
        loader = trainer.clean_monitor_loader
        trainer.model.eval()
        trainer.evaluator = lambda **kwargs: {}
        for _ in range(2):
            trainer._run_evaluator()
            assert len(refs) == 1
            assert trainer.clean_monitor_loader is loader
            assert refs[0]() is not None
    finally:
        trainer.close()
        wandb.finish()
    assert refs[0]() is None
    trainer.close()  # Explicit teardown is idempotent, even if a caller retains the loader.


def test_monitor_train_arm_uses_clean_current_model_embeddings(
    synthetic_export, tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    init = wandb.init
    monkeypatch.setattr(wandb, "init", lambda **kw: init(**{**kw, "mode": "disabled"}))
    cfg = monitor_config(synthetic_export, tmp_path / "monitor")
    trainer = None
    try:
        trainer = ContrastiveTrainer(cfg, rank=0, world_size=1)
        trainer.train_epoch()
        trainer.model.eval()
        # Build a clean oracle directly from known fixture block boundaries.
        collector = EmbeddingCollector(
            trainer.train_unit_ids, trainer.model.emb_dim, trainer.device
        )
        ds = trainer.train_loader.dataset
        with torch.inference_mode():
            for block_start in (0, 180, 360, 540):
                index = DatasetIndex(
                    "euler_allgcl_2025/synthetic_train",
                    block_start,
                    block_start + 32.768,
                )
                clean = ViewData.collate(
                    [trainer.model.tokenize(get_window(ds, index))]
                ).to(trainer.device)
                collector.update(
                    trainer.model(**clean.data["enc_input"]), clean.unit_ids
                )
        expected, expected_ids = collector.compute()
        seen = []

        def evaluator(X_train, train_unit_ids, **kwargs):
            np.testing.assert_array_equal(train_unit_ids, expected_ids)
            torch.testing.assert_close(X_train, expected, rtol=1e-6, atol=1e-6)
            seen.append(True)
            return {}

        trainer.evaluator = evaluator

        def forbidden(*args, **kwargs):
            raise AssertionError("Augmentation used in monitoring")

        monkeypatch.setattr(CalciumTrainingViews, "__call__", forbidden)
        trainer.val_epoch()
        rng = torch.get_rng_state().clone()
        trainer._run_evaluator()
        assert torch.equal(rng, torch.get_rng_state())
        assert seen == [True, True]
    finally:
        if trainer is not None:
            trainer.close()
        wandb.finish()
