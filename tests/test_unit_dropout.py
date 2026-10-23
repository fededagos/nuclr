import h5py
import pytest
import numpy as np
from temporaldata import Data
from src.transforms.unit_dropout import UnitDropout


def test_calcium_dropout_consistent(synthetic_export):
    with h5py.File(
        synthetic_export / "processed/euler_allgcl_2025/synthetic_train.h5"
    ) as f:
        original = Data.from_hdf5(f, lazy=False).slice(0.0, 32.768)
    np.random.seed(1)
    raw = original.calcium_traces.df_over_f.copy()
    normalized = original.calcium_traces.df_over_f_norm.copy()
    ids = original.units.id.copy()
    dropped = UnitDropout(min_units=12)(original)
    assert 12 <= len(dropped.units.id) < len(ids)
    kept = np.array([np.where(ids == x)[0][0] for x in dropped.units.id])
    np.testing.assert_array_equal(dropped.calcium_traces.df_over_f, raw[:, kept])
    np.testing.assert_array_equal(
        dropped.calcium_traces.df_over_f_norm, normalized[:, kept]
    )


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_units_dropped_per_view(tmp_path, value):
    from fixtures.make_synthetic import make_synthetic

    synthetic_export = make_synthetic(tmp_path / "nonfinite")
    import torch
    from omegaconf import OmegaConf
    from torch_brain.data.dataset import DatasetIndex
    from src.dataset import TwoViewData, TwoViewDataset
    from src.utils import instantiate
    from test_time_base import calcium_model

    path = synthetic_export / "processed/euler_allgcl_2025/synthetic_train.h5"
    with h5py.File(path, "r+") as f:
        f["calcium_traces/df_over_f_norm"][0, 0] = value
    model = calcium_model(t_layers=0, st_layers=0, drop_nonfinite_units=True)
    ds = TwoViewDataset(
        root=synthetic_export / "processed",
        config=OmegaConf.create(
            [
                {
                    "selection": [
                        {
                            "brainset": "euler_allgcl_2025",
                            "sessions": ["synthetic_train"],
                        }
                    ]
                }
            ]
        ),
        unit_id_prefix_fn=lambda _: "",
    )
    ds.set_second_view_props(32.768, 240.0, 42, pairing={"mode": "recording"})
    ds.transform = model.tokenize
    ds.common_transform = TwoViewData.from_two_views
    sample = ds[DatasetIndex("euler_allgcl_2025/synthetic_train", 0.0, 32.768)]
    assert "synthetic_train_roi0" not in sample.view1.unit_ids
    assert "synthetic_train_roi0" in sample.view2.unit_ids
    batch = TwoViewData.collate([sample]).to(torch.device("cpu"))
    assert batch.view1.data["enc_input"]["bins"].isfinite().all()
    assert batch.view2.data["enc_input"]["bins"].isfinite().all()
    assert len(batch.view1.unit_ids) == 39 and len(batch.view2.unit_ids) == 40
    assert batch.match_idx[0].shape[1] == 39
    loss = instantiate(OmegaConf.load("configs/loss/sw_dcl.yaml"), model=model)
    value = loss(
        model(**batch.view1.data["enc_input"]),
        model(**batch.view2.data["enc_input"]),
        batch,
        prefix="test",
    )
    assert value.isfinite()
    value.backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
