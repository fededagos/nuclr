"""Training-view perturbations preserve unit alignment and the disabled RNG stream."""

import copy
from types import SimpleNamespace

import numpy as np


def sample(values):
    values = np.asarray(values, dtype=np.float32)
    return SimpleNamespace(
        units=np.arange(values.shape[1]),
        calcium_traces=SimpleNamespace(
            df_over_f=values.copy(),
            df_over_f_norm=values.copy() / 10,
            timestamps=np.arange(len(values)) * 0.128,
        ),
    )


def test_disabled_matches_8afa03c_unit_dropout_and_rng(synthetic_export):
    import subprocess
    from pathlib import Path
    from test_pairing import pairing_dataset
    from src.transforms.calcium_augmentation import CalciumTrainingViews

    # Execute the actual frozen implementation, rather than duplicating its logic.
    namespace = {
        "__name__": "src.transforms.frozen_unit_dropout",
        "__package__": "src.transforms",
    }
    source = subprocess.check_output(
        ["git", "show", "8afa03c:src/transforms/unit_dropout.py"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    )
    exec(compile(source, "8afa03c/unit_dropout.py", "exec"), namespace)
    ds = pairing_dataset(synthetic_export, "recording")
    original = ds.get("euler_allgcl_2025/synthetic_train", -1e-9, 32.768 - 1e-9)
    for seed in range(8):
        np.random.seed(seed)
        expected = namespace["UnitDropout"](0.5)(copy.deepcopy(original))
        expected_rng = np.random.get_state()
        np.random.seed(seed)
        actual = CalciumTrainingViews(min_units=0.5)(copy.deepcopy(original))
        actual_rng = np.random.get_state()
        np.testing.assert_array_equal(actual.units.id, expected.units.id)
        for key in ("df_over_f", "df_over_f_norm", "timestamps"):
            a, b = getattr(actual.calcium_traces, key), getattr(
                expected.calcium_traces, key
            )
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
        assert actual_rng[0] == expected_rng[0]
        np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
        assert actual_rng[2:] == expected_rng[2:]


def test_jitter_is_per_unit_per_view_edge_padded_without_wrap():
    from src.transforms.calcium_augmentation import CalciumTrainingViews

    values = np.repeat(np.arange(256, dtype=np.float32)[:, None], 128, axis=1)
    transform = CalciumTrainingViews(
        min_units=1000, augmentation="jitter", jitter_frames=2
    )
    np.random.seed(42)
    a, b = transform(sample(values)), transform(sample(values))
    shifts = []
    for view in (a, b):
        observed = view.calcium_traces.df_over_f
        shift = 100 - observed[100].astype(int)
        shifts.append(shift)
        assert set(shift) == {-2, -1, 0, 1, 2}
        expected = np.take_along_axis(
            values, np.clip(np.arange(256)[:, None] - shift, 0, 255), axis=0
        )
        np.testing.assert_array_equal(observed, expected)
        np.testing.assert_allclose(view.calcium_traces.df_over_f_norm, observed / 10)
        np.testing.assert_array_equal(view.units, np.arange(128))
        np.testing.assert_array_equal(
            view.calcium_traces.timestamps, np.arange(256) * 0.128
        )
    assert not np.array_equal(shifts[0], shifts[1])


def test_gain_is_light_constant_over_time_independent_per_unit_and_view():
    from src.transforms.calcium_augmentation import CalciumTrainingViews

    transform = CalciumTrainingViews(
        min_units=1000, augmentation="scale", gain_range=(0.8, 1.2)
    )
    np.random.seed(123)
    original = sample(np.ones((256, 128)))
    a, b = transform(copy.deepcopy(original)), transform(copy.deepcopy(original))
    gain = a.calcium_traces.df_over_f[0]
    assert np.all((gain >= 0.8) & (gain <= 1.2))
    assert len(np.unique(gain)) == 128
    np.testing.assert_array_equal(
        a.calcium_traces.df_over_f, np.broadcast_to(gain, (256, 128))
    )
    np.testing.assert_allclose(
        a.calcium_traces.df_over_f_norm, a.calcium_traces.df_over_f / 10
    )
    assert not np.array_equal(gain, b.calcium_traces.df_over_f[0])
    np.testing.assert_array_equal(original.calcium_traces.df_over_f, 1)


def test_noise_sigma_is_in_baseline_sd_and_draws_are_independent():
    from src.transforms.calcium_augmentation import CalciumTrainingViews

    transform = CalciumTrainingViews(
        min_units=1000, augmentation="noise", noise_sigma=0.2
    )
    values = np.ones((256, 128), dtype=np.float32)
    np.random.seed(987)
    a, b = transform(sample(values)), transform(sample(values))
    noise = a.calcium_traces.df_over_f - values
    assert abs(float(noise.mean())) < 0.005
    assert abs(float(noise.std()) - 0.2) < 0.005
    assert abs(float(np.corrcoef(noise[:, 0], noise[:, 1])[0, 1])) < 0.2
    assert not np.array_equal(noise, b.calcium_traces.df_over_f - values)
    np.testing.assert_allclose(
        a.calcium_traces.df_over_f_norm, a.calcium_traces.df_over_f / 10, atol=2e-8
    )
    assert a.calcium_traces.df_over_f.dtype == np.float32
    assert a.calcium_traces.df_over_f_norm.dtype == np.float32


def test_all_is_the_three_independent_draws_in_order():
    from src.transforms.calcium_augmentation import CalciumTrainingViews

    values = np.arange(256 * 128, dtype=np.float32).reshape(256, 128) / 100
    np.random.seed(321)
    expected = sample(values)
    for mode in ("jitter", "scale", "noise"):
        expected = CalciumTrainingViews(min_units=1000, augmentation=mode)(expected)
    np.random.seed(321)
    actual = CalciumTrainingViews(min_units=1000, augmentation="all")(sample(values))
    for key in ("df_over_f", "df_over_f_norm"):
        np.testing.assert_array_equal(
            getattr(actual.calcium_traces, key), getattr(expected.calcium_traces, key)
        )


def test_allgcl_defaults_off_and_one_config_key_selects_each_pilot():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from src.utils import instantiate

    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()), version_base="1.2"
    ):
        cfg = compose(config_name="train_allgcl")
        assert cfg.train_transform.augmentation == "off"
        for mode in ("off", "jitter", "scale", "noise", "all"):
            cfg = compose(
                config_name="train_allgcl",
                overrides=[f"train_transform.augmentation={mode}"],
            )
            transform = instantiate(cfg.train_transform)
            assert transform.augmentation == mode


def test_off_real_training_is_bit_identical_to_frozen_unit_dropout(
    synthetic_export, tmp_path, monkeypatch
):
    import torch
    import wandb
    from test_allgcl_config import allgcl_config
    from src.trainers.contrastive import ContrastiveTrainer
    from omegaconf import OmegaConf

    torch.set_num_threads(1)
    wandb_init = wandb.init
    monkeypatch.setattr(
        wandb, "init", lambda **kw: wandb_init(**{**kw, "mode": "disabled"})
    )
    results = []
    try:
        for legacy in (True, False):
            cfg = allgcl_config(synthetic_export, tmp_path / str(legacy))
            if legacy:
                cfg.train_transform = OmegaConf.create(
                    {
                        "_target_": "src.transforms.unit_dropout.UnitDropout",
                        "min_units": 0.5,
                    }
                )
            cfg.model.dim = 32
            cfg.model.self_heads = 2
            cfg.model.dim_head = 16
            cfg.model.t_layers = cfg.model.st_layers = 0
            cfg.precision = "fp32"
            cfg.target_steps = 2
            trainer = ContrastiveTrainer(cfg, rank=0, world_size=1)
            trainer.train_epoch()
            trainer.val_epoch()
            results.append(
                (
                    copy.deepcopy(trainer.model.state_dict()),
                    copy.deepcopy(trainer.loss.state_dict()),
                    trainer.last_val_loss,
                    np.random.get_state(),
                    torch.get_rng_state().clone(),
                )
            )
            wandb.finish()
        for before, after in zip(results[0][:2], results[1][:2]):
            assert before.keys() == after.keys()
            for key in before:
                assert torch.equal(before[key], after[key]), key
        assert results[0][2] == results[1][2]
        np.testing.assert_array_equal(results[0][3][1], results[1][3][1])
        assert results[0][3][2:] == results[1][3][2:]
        assert torch.equal(results[0][4], results[1][4])
    finally:
        wandb.finish()


def test_training_views_change_but_validation_and_export_never_call_transform(
    synthetic_export, tmp_path, monkeypatch
):
    import torch
    import wandb
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from test_allgcl_config import allgcl_config
    from src.trainers.contrastive import ContrastiveTrainer
    from src.trainers.forward import ForwardPass
    from src.transforms.calcium_augmentation import CalciumTrainingViews

    torch.set_num_threads(1)
    # Offline W&B still launches a socket service, unavailable in Codex's sandbox.
    # Disable only telemetry; the real trainers/loaders/model/loss/export run.
    wandb_init = wandb.init
    monkeypatch.setattr(
        wandb, "init", lambda **kw: wandb_init(**{**kw, "mode": "disabled"})
    )
    original_call = CalciumTrainingViews.__call__
    calls = []

    def record(self, data):
        before = data.calcium_traces.df_over_f_norm.copy()
        out = original_call(self, data)
        if before.shape == out.calcium_traces.df_over_f_norm.shape:
            assert not np.array_equal(before, out.calcium_traces.df_over_f_norm)
        calls.append(out.session.id)
        return out

    monkeypatch.setattr(CalciumTrainingViews, "__call__", record)
    cfg = allgcl_config(synthetic_export, tmp_path / "train")
    cfg.train_transform.augmentation = "all"
    cfg.train_transform.min_units = 1000
    cfg.model.dim = 32
    cfg.model.self_heads = 2
    cfg.model.dim_head = 16
    cfg.model.t_layers = cfg.model.st_layers = 0
    cfg.precision = "fp32"
    cfg.target_steps = 2
    cfg.data.evaluator = None
    try:
        trainer = ContrastiveTrainer(cfg, rank=0, world_size=1)
        trainer.train_epoch()
        assert trainer.train_step == 2 and len(calls) >= 4
        assert set(calls) == {"euler_allgcl_2025/synthetic_train"}

        def forbidden(*args, **kwargs):
            raise AssertionError("Training augmentation leaked into inference")

        monkeypatch.setattr(CalciumTrainingViews, "__call__", forbidden)
        first = next(iter(trainer.val_loader))
        second = next(iter(trainer.val_loader))
        for a, b in ((first.view1, second.view1), (first.view2, second.view2)):
            torch.testing.assert_close(
                a.data["enc_input"]["bins"], b.data["enc_input"]["bins"], rtol=0, atol=0
            )
        trainer.val_epoch()
        assert np.isfinite(trainer.last_val_loss)
        wandb.finish()
        outputs = []
        for mode in ("off", "all"):
            output = tmp_path / ("export_" + mode)
            with initialize_config_dir(
                config_dir=str(Path("configs").resolve()), version_base="1.2"
            ):
                forward_cfg = compose(
                    config_name="forward_allgcl",
                    overrides=[
                        f"data.data_root={synthetic_export}/processed",
                        f"data.splits_file={synthetic_export}/splits.json",
                        f"data.units_file={synthetic_export}/units.parquet",
                        f"output_dir={output}",
                        "device=cpu",
                        "precision=fp32",
                        "num_workers=0",
                        "init_only=true",
                        "model.t_layers=0",
                        "model.st_layers=0",
                        f"train_transform.augmentation={mode}",
                    ],
                )
            ForwardPass(forward_cfg, rank=0, world_size=1).train()
            outputs.append((output / "embeddings.parquet").read_bytes())
            wandb.finish()
        assert outputs[0] == outputs[1]
    finally:
        wandb.finish()
