import json
import h5py
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.checkpoint import save_ckpt
from src.utils import Precision, instantiate
from src.trainers.forward import ForwardPass


@pytest.mark.parametrize("missing_condition", [False, True])
def test_forward_scopes(tmp_path, missing_condition):
    from fixtures.make_synthetic import make_synthetic

    synthetic_export = make_synthetic(tmp_path / "data", valid_conditions=["c1"] * 4)
    if missing_condition:
        with h5py.File(
            synthetic_export / "processed/euler_allgcl_2025/synthetic_train.h5", "r+"
        ) as f:
            values = f["calcium_traces/df_over_f_norm"][:]
            values[f["calcium_traces/timestamps"][:] >= 360.0, 0] = np.nan
            f["calcium_traces/df_over_f_norm"][:] = values
    model_cfg = OmegaConf.load("configs/model/nuclr_v2ga_ca2.yaml")
    model_cfg.ctx_duration = 32.768
    model_cfg.sampling_rate = 7.8125
    model_cfg.patch_size = 8
    model_cfg.drop_nonfinite_units = True
    model_cfg.t_layers = model_cfg.st_layers = (
        0  # CPU; full attention is checked on A100.
    )
    model = instantiate(model_cfg, precision=Precision("fp32"))
    ckpt = save_ckpt(
        tmp_path / "model.pt",
        cfg=OmegaConf.create({"model": model_cfg}),
        model=model,
        epoch=3,
        run_id="test",
    )
    output = tmp_path / "forward"
    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()), version_base="1.2"
    ):
        cfg = compose(
            config_name="forward_allgcl",
            overrides=[
                f"data.data_root={synthetic_export}/processed",
                f"data.splits_file={synthetic_export}/splits.json",
                f"data.units_file={synthetic_export}/units.parquet",
                f"output_dir={output}",
                f"ckpt.load_from={ckpt}",
                "device=cpu",
                "precision=fp32",
                "num_workers=0",
            ],
        )
    ForwardPass(cfg, rank=0, world_size=1).train()
    import wandb

    wandb.finish()
    df = pd.read_parquet(output / "embeddings.parquet")
    all_rows = df[df.scope == "all"].set_index("unit_id")
    assert len(all_rows) == 80
    assert set(df.scope) == {
        "all",
        "condition:c1",
        "condition:d",
        "stimulus:chirp",
        "stimulus:movingbar",
    }
    assert np.isfinite(np.stack(df.emb)).all()
    if missing_condition:
        assert all_rows.loc["synthetic_train_roi0", "n_windows"] == 6
        assert set(all_rows.drop(index="synthetic_train_roi0").n_windows) == {12}
        assert not (
            (df.unit_id == "synthetic_train_roi0") & (df.scope == "condition:d")
        ).any()
    else:
        assert set(all_rows.n_windows) == {12}
    assert not (
        (df.unit_id.str.startswith("synthetic_valid")) & (df.scope == "condition:d")
    ).any()
    for uid, row in all_rows.iterrows():
        stimuli = df[(df.unit_id == uid) & df.scope.str.startswith("stimulus:")]
        assert row.n_windows == stimuli.n_windows.sum()
        np.testing.assert_allclose(
            row.emb,
            np.average(np.stack(stimuli.emb), axis=0, weights=stimuli.n_windows),
            atol=1e-6,
        )
    emb_type = pq.read_schema(output / "embeddings.parquet").field("emb").type
    assert emb_type.list_size == 256 and str(emb_type.value_type) == "float"
    meta = json.loads((output / "embeddings_meta.json").read_text())
    assert meta["epoch"] == 3 and meta["ckpt"] == str(ckpt)


def test_init_only_forward_is_reproducible_without_checkpoint(
    synthetic_export, tmp_path
):
    import wandb

    results = []
    for run in range(2):
        output = tmp_path / f"init_{run}"
        with initialize_config_dir(
            config_dir=str(Path("configs").resolve()), version_base="1.2"
        ):
            cfg = compose(
                config_name="forward_allgcl",
                overrides=[
                    f"data.data_root={synthetic_export}/processed",
                    f"data.splits_file={synthetic_export}/splits.json",
                    f"data.units_file={synthetic_export}/units.parquet",
                    f"output_dir={output}",
                    "init_only=true",
                    "seed=42",
                    "device=cpu",
                    "precision=fp32",
                    "num_workers=0",
                    "model.t_layers=0",
                    "model.st_layers=0",
                ],
            )
        runner = ForwardPass(cfg, rank=0, world_size=1)
        runner.train()
        wandb.finish()
        assert cfg.ckpt.load_from is None
        assert not list(output.rglob("*.pt"))
        meta = json.loads((output / "embeddings_meta.json").read_text())
        assert meta["init_only"] is True and meta["seed"] == 42
        assert meta["ckpt"] is None and meta["epoch"] is None
        assert meta["model_config"]["ctx_duration"] == 32.768
        frame = pd.read_parquet(output / "embeddings.parquet").sort_values(
            ["unit_id", "scope"]
        )
        results.append(np.stack(frame.emb))
    np.testing.assert_array_equal(results[0], results[1])


@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None])
def test_init_only_rejects_non_boolean(synthetic_export, tmp_path, invalid):
    import wandb

    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()), version_base="1.2"
    ):
        cfg = compose(
            config_name="forward_allgcl",
            overrides=[
                f"data.data_root={synthetic_export}/processed",
                f"data.splits_file={synthetic_export}/splits.json",
                f"data.units_file={synthetic_export}/units.parquet",
                f"output_dir={tmp_path}/invalid",
                "device=cpu",
                "precision=fp32",
                "num_workers=0",
                "model.t_layers=0",
                "model.st_layers=0",
            ],
        )
    cfg.init_only = invalid
    try:
        with pytest.raises(TypeError, match="init_only.*bool"):
            ForwardPass(cfg, rank=0, world_size=1)
    finally:
        wandb.finish()


def test_stimulus_condition_scopes_preserve_existing_rows(synthetic_export, tmp_path):
    """Adding joint scopes changes no old row, count, or unnormalised mean."""
    import wandb

    results = []
    for joint in (False, True):
        scopes = ["all", "condition", "stimulus"]
        if joint:
            scopes.append("stimulus_condition")
        with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.2"):
            cfg = compose(config_name="forward_allgcl", overrides=[
                f"data.data_root={synthetic_export}/processed",
                f"data.splits_file={synthetic_export}/splits.json",
                f"data.units_file={synthetic_export}/units.parquet",
                f"output_dir={tmp_path}/joint_{joint}",
                f"scopes={json.dumps(scopes)}",
                "init_only=true", "seed=42", "device=cpu", "precision=fp32",
                "num_workers=0", "model.t_layers=0", "model.st_layers=0",
                "wandb.mode=disabled",
            ])
        try:
            ForwardPass(cfg, rank=0, world_size=1).train()
        finally:
            wandb.finish()
        results.append(pd.read_parquet(Path(cfg.output_dir) / "embeddings.parquet"))
    old, new = results
    legacy = new[~new.scope.str.startswith("stimulus_condition:")]
    pd.testing.assert_frame_equal(old.reset_index(drop=True), legacy.reset_index(drop=True))
    joint = new[new.scope.str.startswith("stimulus_condition:")]
    assert set(joint.scope) == {
        f"stimulus_condition:{s}|{c}"
        for s in ("chirp", "movingbar") for c in ("c1", "d")
    }
    for uid, rows in new.groupby("unit_id"):
        children = joint[joint.unit_id == uid]
        for kind, position in [("stimulus", 0), ("condition", 1)]:
            for row in rows[rows.scope.str.startswith(kind + ":")].itertuples():
                label = row.scope.split(":", 1)[1]
                selected = children[children.scope.map(lambda s: s.split(":", 1)[1].split("|")[position] == label)]
                assert selected.n_windows.sum() == row.n_windows
                np.testing.assert_allclose(np.average(np.stack(selected.emb), axis=0, weights=selected.n_windows), row.emb, atol=1e-6)
