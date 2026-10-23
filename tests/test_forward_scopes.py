import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.checkpoint import save_ckpt
from src.utils import Precision, instantiate
from src.trainers.forward import ForwardPass


def test_forward_scopes(tmp_path):
    from fixtures.make_synthetic import make_synthetic

    synthetic_export = make_synthetic(tmp_path / "data", valid_conditions=["c1"] * 4)
    model_cfg = OmegaConf.load("configs/model/nuclr_v2ga_ca2.yaml")
    model_cfg.ctx_duration = 32.768
    model_cfg.sampling_rate = 7.8125
    model_cfg.patch_size = 8
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
