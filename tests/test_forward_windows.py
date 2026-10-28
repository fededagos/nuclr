"""Real CPU forwards: unchanged be5b439 scopes and single-pass window output."""
from pathlib import Path
import json
import subprocess

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from hydra import compose, initialize_config_dir

from fixtures.make_synthetic import make_synthetic
from src.trainers import forward as module

BASE = "be5b439d7dee66865c63753c0a5326e6a6182e3e"


def configuration(data, output, option="absent"):
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.2"):
        overrides = [f"data.data_root={data}/processed", f"data.splits_file={data}/splits.json",
                     f"data.units_file={data}/units.parquet", f"output_dir={output}",
                     "scopes=[all,condition,stimulus,stimulus_condition]", "init_only=true",
                     "device=cpu", "precision=fp32", "num_workers=0", "seed=42",
                     "model.t_layers=0", "model.st_layers=0"]
        if option != "absent":
            overrides.append("++export_window_stimuli=" + json.dumps(option))
        return compose(config_name="forward_allgcl", overrides=overrides)


def run_forward(cls, cfg, *, fail=False, captures=None):
    import wandb
    runner = cls(cfg, rank=0, world_size=1)
    calls = []
    def hook(model, inputs, result):
        calls.append(len(result))
        if captures is not None:
            captures.append(result.detach().float().cpu().numpy().copy())
        if fail and len(calls) == 2:
            return result * float("nan")
    handle = runner.model.register_forward_hook(hook)
    try:
        runner.train()
    finally:
        handle.remove()
        wandb.finish()
    return calls


def test_windows_reuse_outputs_preserve_scope_bytes_and_order(tmp_path):
    data = make_synthetic(tmp_path / "data", block_duration=70.)
    with h5py.File(data / "processed/euler_allgcl_2025/synthetic_train.h5", "r+") as h5:
        h5["calcium_traces/df_over_f_norm"][0, 0] = np.nan
    # Actual committed implementation is the oracle, not another branch of new code.
    source = subprocess.check_output(["git", "show", BASE + ":src/trainers/forward.py"], text=True)
    namespace = dict(vars(module))
    exec(compile(source, "<be5b439-forward-oracle>", "exec"), namespace)
    cases = [(namespace["ForwardPass"], "oracle", "absent"),
             (module.ForwardPass, "off", None),
             (module.ForwardPass, "on", ["chirp", "movingbar"])]
    counts, captures = [], []
    for cls, name, option in cases:
        counts.append(run_forward(cls, configuration(data, tmp_path / name, option), captures=captures if name == "on" else None))
    assert counts[0] == counts[1] == counts[2]
    assert len(counts[2]) == 16 and counts[2][0] == 39
    assert not (tmp_path / "off/windows.parquet").exists()
    assert (tmp_path / "oracle/embeddings.parquet").read_bytes() == (tmp_path / "off/embeddings.parquet").read_bytes() == (tmp_path / "on/embeddings.parquet").read_bytes()
    windows = pd.read_parquet(tmp_path / "on/windows.parquet")
    assert len(windows) == sum(counts[2])
    assert pq.read_schema(tmp_path / "on/windows.parquet").field("emb").type.list_size == 256
    assert not (tmp_path / "on/windows.parquet.tmp").exists()
    for capture, ((recording, block, window), rows) in zip(captures, windows.groupby(["recording", "block", "window"], sort=False)):
        np.testing.assert_array_equal(np.stack(rows.emb), capture)
        assert rows.n_context.nunique() == 1 and rows.n_context.iloc[0] == len(rows)
        assert rows.start_s.nunique() == 1
        assert rows.start_s.iloc[0] == pytest.approx(block * 130. + window * 32.768)
        assert window in (0, 1)
        assert rows.stimulus.unique().tolist() == [["chirp", "movingbar"][block % 2]]
        assert rows.condition_class.unique().tolist() == ["control" if block < 2 else "drug"]
    for _, rows in windows.groupby("unit_id", sort=False):
        assert rows.start_s.is_monotonic_increasing
    scopes = pd.read_parquet(tmp_path / "on/embeddings.parquet").set_index(["unit_id", "scope"])
    for (uid, stimulus, condition), rows in windows.groupby(["unit_id", "stimulus", "condition"]):
        parent = scopes.loc[(uid, f"stimulus_condition:{stimulus}|{condition}")]
        assert len(rows) == parent.n_windows
        np.testing.assert_allclose(np.stack(rows.emb).astype(np.float64).mean(axis=0), parent.emb, atol=2e-6, rtol=2e-6)


def test_windows_filter_and_atomic_failure(tmp_path):
    data = make_synthetic(tmp_path / "data", block_duration=70.)
    cfg = configuration(data, tmp_path / "filtered", ["chirp"])
    run_forward(module.ForwardPass, cfg)
    assert set(pd.read_parquet(tmp_path / "filtered/windows.parquet").stimulus) == {"chirp"}
    assert "stimulus:movingbar" in set(pd.read_parquet(tmp_path / "filtered/embeddings.parquet").scope)
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="Non-finite embedding"):
        run_forward(module.ForwardPass, configuration(data, failed, ["chirp", "movingbar"]), fail=True)
    assert not (failed / "windows.parquet").exists()
    assert not (failed / "embeddings.parquet").exists()
