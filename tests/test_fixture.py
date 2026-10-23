import numpy as np
from omegaconf import OmegaConf
from torch_brain.data import Dataset
from src.samplers import RandomFixedWindowSampler


def test_synthetic_schema_loads(synthetic_export):
    ds = Dataset(
        root=synthetic_export / "processed",
        config=OmegaConf.create(
            [
                {
                    "selection": [
                        {
                            "brainset": "euler_allgcl_2025",
                            "sessions": ["synthetic_train", "synthetic_valid"],
                        }
                    ]
                }
            ]
        ),
        unit_id_prefix_fn=lambda _: "",
        session_id_prefix_fn=lambda _: "",
    )
    sampler = RandomFixedWindowSampler(ds.get_sampling_intervals(), 32.768, seed=42)
    indices = list(sampler) + list(sampler)
    assert len(indices) >= 20
    for index in indices[:20]:
        data = ds.get(index.recording_id, index.start, index.end)
        assert data.calcium_traces.df_over_f_norm.shape == (256, 40)
        assert data.session.id == index.recording_id.split("/")[-1]
        assert np.all(np.diff(data.calcium_traces.timestamps) > 0)
        assert len(data.units.id) == 40
