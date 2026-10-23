import numpy as np
from omegaconf import OmegaConf
from torch_brain.data.dataset import DatasetIndex
from src.dataset import TwoViewDataset


def pairing_dataset(export, mode, distance=240.):
    ds = TwoViewDataset(root=export / 'processed', config=OmegaConf.create([
        {'selection': [{'brainset': 'euler_allgcl_2025', 'sessions': ['synthetic_train']}]}]),
        unit_id_prefix_fn=lambda _: '', session_id_prefix_fn=lambda _: '')
    ds.set_second_view_props(32.768, distance, 42,
                             pairing={'mode': mode, 'max_distance': distance})
    ds.common_transform = lambda views: views
    return ds


def test_recording_mode_crosses_conditions(synthetic_export):
    ds = pairing_dataset(synthetic_export, 'recording')
    first = DatasetIndex('euler_allgcl_2025/synthetic_train', 0., 32.768)
    starts = [ds[first][1].absolute_start for _ in range(500)]
    assert any(s >= 360 for s in starts)
    assert any(s < 120 for s in starts)


def test_within_condition_never_crosses(synthetic_export):
    ds = pairing_dataset(synthetic_export, 'within_condition')
    first = DatasetIndex('euler_allgcl_2025/synthetic_train', 180., 212.768)
    for _ in range(500):
        a, b = ds[first]
        assert np.array_equal(a.blocks.condition, b.blocks.condition)


def test_local_respects_max_distance(synthetic_export):
    ds = pairing_dataset(synthetic_export, 'local', distance=20.)
    first = DatasetIndex('euler_allgcl_2025/synthetic_train', 40., 72.768)
    assert all(abs(ds[first][1].absolute_start - first.start) <= 20. for _ in range(500))


def test_views_inside_blocks(synthetic_export):
    from src.samplers import RandomFixedWindowSampler
    for mode in ('recording', 'local', 'within_condition'):
        ds = pairing_dataset(synthetic_export, mode)
        for first in RandomFixedWindowSampler(ds.get_sampling_intervals(), 32.768, seed=5):
            for view in ds[first]:
                assert len(view.blocks) == 1
                assert view.calcium_traces.df_over_f_norm.shape == (256, 40)
                assert np.allclose(np.diff(view.calcium_traces.timestamps), 0.128)


def test_gap_view_rejected(synthetic_export):
    import pytest
    ds = pairing_dataset(synthetic_export, 'recording')
    with pytest.raises(ValueError, match='block'):
        ds[DatasetIndex('euler_allgcl_2025/synthetic_train', 110., 142.768)]
