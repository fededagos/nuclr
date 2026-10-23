"""Exercise the real data/paired-window/tokenizer boundary using --allgcl-export."""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from src.allgcl import split_selection
from src.dataset import TwoViewData, TwoViewDataset
from src.samplers import RandomFixedWindowSampler
from src.utils import Precision, instantiate


@pytest.mark.parametrize("mode", ["recording", "local", "within_condition"])
def test_smoke_export_paired_tokenization(smoke_export, mode):
    data_cfg = OmegaConf.create(
        {
            "brainset": "euler_allgcl_2025",
            "splits_file": str(smoke_export / "splits.json"),
        }
    )
    ds = TwoViewDataset(
        root=smoke_export / "processed",
        config=split_selection(data_cfg, ["train", "valid"]),
        unit_id_prefix_fn=lambda _: "",
        session_id_prefix_fn=lambda _: "",
    )
    ds.set_second_view_props(
        32.768, 240.0, seed=42, pairing={"mode": mode, "max_distance": 240.0}
    )
    ds.common_transform = lambda views: views
    model_cfg = OmegaConf.load("configs/model/nuclr_v2ga_ca2.yaml")
    model_cfg.ctx_duration, model_cfg.sampling_rate, model_cfg.patch_size = (
        32.768,
        7.8125,
        8,
    )
    model_cfg.drop_nonfinite_units = True
    model = instantiate(model_cfg, precision=Precision("fp32"))
    metadata = pd.read_parquet(smoke_export / "units.parquet")
    indices = list(
        RandomFixedWindowSampler(ds.get_sampling_intervals(), 32.768, seed=42)
    )
    assert indices
    for draw in range(20):
        index = indices[draw % len(indices)]
        views = ds[index]
        sid = index.recording_id.split("/")[-1]
        ids = metadata[metadata.session_id == sid].id.to_numpy()
        for view in views:
            assert len(view.blocks) == 1
            assert view.calcium_traces.df_over_f_norm.shape == (256, len(ids))
            assert np.allclose(
                np.diff(view.calcium_traces.timestamps), 0.128, atol=1e-5
            )
            np.testing.assert_array_equal(view.units.id, ids)
            assert view.session.id == sid
        if mode == "within_condition":
            np.testing.assert_array_equal(
                views[0].blocks.condition, views[1].blocks.condition
            )
        if mode == "local":
            assert (
                abs(views[0].absolute_start - views[1].absolute_start) <= 240.0 + 1e-8
            )
        pair = TwoViewData.from_two_views([model.tokenize(view) for view in views])
        batch = TwoViewData.collate([pair])
        for view in (batch.view1, batch.view2):
            assert view.data["enc_input"]["bins"].shape == (len(view.unit_ids) * 32, 8)
            assert view.data["enc_input"]["bins"].isfinite().all()
        n_matches = len(np.intersect1d(batch.view1.unit_ids, batch.view2.unit_ids))
        assert batch.match_idx[0].shape == (2, n_matches)
