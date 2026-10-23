import numpy as np
import pytest
import torch
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries
from src.models.nuclr_v2ga_ca2 import NuclrV2gaCa2
from src.utils import Precision


def calcium_model(**kwargs):
    defaults = dict(
        ctx_duration=32.768,
        patch_size=8,
        sampling_rate=7.8125,
        precision=Precision("fp32"),
        dim=32,
        self_heads=2,
        dim_head=16,
        atn_dropout=0.0,
        lin_dropout=0.0,
        t_layers=1,
        st_layers=1,
    )
    return NuclrV2gaCa2(**(defaults | kwargs))


def test_tokenize_uses_full_window():
    model = calcium_model()
    values = np.arange(256 * 3, dtype="float32").reshape(256, 3)
    view = Data(
        units=ArrayDict(id=np.array(["a", "b", "c"])),
        session=Data(id="field"),
        calcium_traces=IrregularTimeSeries(
            timestamps=np.arange(256) / 7.8125,
            df_over_f_norm=values,
            domain=Interval(0.0, 32.768),
        ),
        domain=Interval(0.0, 32.768),
    )
    tokens = model.tokenize(view)
    from torch_brain.data import collate

    bins = collate([tokens.data])["enc_input"]["bins"]
    assert model.num_latents == 32
    torch.testing.assert_close(bins, torch.from_numpy(values.T.reshape(-1, 8)))


def test_bad_window_raises():
    with pytest.raises(ValueError, match="multiple"):
        calcium_model(ctx_duration=30.0)


def test_bugeon_defaults_unchanged():
    # An old checkpoint/config without sampling_rate retains its 32 x 4 layout.
    with pytest.warns(UserWarning, match="Bugeon"):
        model = calcium_model(ctx_duration=30.0, patch_size=4, sampling_rate=None)
    assert model.num_latents == 32


def test_tokenize_rejects_wrong_sample_count(synthetic_export):
    import h5py

    with h5py.File(
        synthetic_export / "processed/euler_allgcl_2025/synthetic_train.h5"
    ) as f:
        data = Data.from_hdf5(f, lazy=False).slice(0.0, 30.0)
    with pytest.raises(ValueError, match="Expected 256 samples"):
        calcium_model().tokenize(data)


def test_shipped_bugeon_config_keeps_legacy_windows():
    from omegaconf import OmegaConf
    from src.utils import instantiate
    from torch_brain.data import collate

    cfg = OmegaConf.load("configs/model/nuclr_v2ga_ca2.yaml")
    cfg.ctx_duration = OmegaConf.load("configs/train_ca.yaml").views.duration
    assert cfg.ctx_duration == 30.0
    assert "sampling_rate" not in cfg
    with pytest.warns(UserWarning, match="Legacy Bugeon"):
        model = instantiate(cfg, precision=Precision("fp32"))
    assert model.sampling_rate is None
    assert model.num_latents == 32 and model.patch_size == 4
    for n_frames in (128, 129):
        values = np.arange(n_frames * 3, dtype="float32").reshape(n_frames, 3)
        view = Data(
            units=ArrayDict(id=np.array(["a", "b", "c"])),
            session=Data(id="bugeon"),
            calcium_traces=IrregularTimeSeries(
                timestamps=np.arange(n_frames) * 0.23344201114422702,
                df_over_f_norm=values,
                domain=Interval(0.0, 30.0),
            ),
            domain=Interval(0.0, 30.0),
        )
        if n_frames == 129:
            with pytest.warns(UserWarning, match="drops 1 trailing samples"):
                tokens = model.tokenize(view)
        else:
            tokens = model.tokenize(view)
        bins = collate([tokens.data])["enc_input"]["bins"]
        torch.testing.assert_close(
            bins, torch.from_numpy(values[:128].T.reshape(-1, 4))
        )


def test_bugeon_jittered_timestamps_tokenize_plain_dataset_windows(
    synthetic_export, tmp_path
):
    """Exercise the unshifted Bugeon validation Dataset.get path with measured-time jitter."""
    import h5py
    import warnings
    from brainsets.core import serialize_fn_map
    from omegaconf import OmegaConf
    from torch_brain.data import Dataset, collate
    from torch_brain.data.sampler import SequentialFixedWindowSampler
    from src.utils import instantiate

    cfg = OmegaConf.load("configs/model/nuclr_v2ga_ca2.yaml")
    cfg.ctx_duration = OmegaConf.load("configs/train_ca.yaml").views.duration
    with pytest.warns(UserWarning, match="Legacy Bugeon"):
        model = instantiate(cfg, precision=Precision("fp32"))
    period = 0.23344201114422702
    timestamps = np.arange(0.0, 1800.0, period)
    timestamps += np.random.default_rng(42).normal(0.0, 0.0005, len(timestamps))
    timestamps -= timestamps[0]
    assert (np.diff(timestamps) > 0).all()
    values = np.arange(len(timestamps) * 40, dtype="float32").reshape(-1, 40)
    with h5py.File(
        synthetic_export / "processed/euler_allgcl_2025/synthetic_valid.h5"
    ) as f:
        data = Data.from_hdf5(f, lazy=False)
    domain = Interval(0.0, float(timestamps[-1] + period))
    # Bugeon has domain intervals, not All-GCL stimulus blocks.
    data = Data(
        domain=domain,
        brainset=data.brainset,
        session=data.session,
        units=data.units,
        calcium_traces=IrregularTimeSeries(
            timestamps=timestamps,
            df_over_f=values,
            df_over_f_norm=values,
            domain=domain,
        ),
    )
    root = tmp_path / "processed"
    path = root / "euler_allgcl_2025/synthetic_valid.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)
    ds = Dataset(
        root=root,
        config=OmegaConf.create(
            [
                {
                    "selection": [
                        {
                            "brainset": "euler_allgcl_2025",
                            "sessions": ["synthetic_valid"],
                        }
                    ]
                }
            ]
        ),
        unit_id_prefix_fn=lambda _: "",
    )
    counts = set()
    for index in SequentialFixedWindowSampler(
        sampling_intervals=ds.get_sampling_intervals(),
        window_length=30.0,
        drop_short=True,
    ):
        view = ds.get(index.recording_id, index.start, index.end)
        n_frames = len(view.calcium_traces.timestamps)
        counts.add(n_frames)
        assert n_frames in (128, 129)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tokens = model.tokenize(view)
        assert any("drops 1 trailing samples" in str(w.message) for w in caught) == (
            n_frames == 129
        )
        bins = collate([tokens.data])["enc_input"]["bins"]
        torch.testing.assert_close(
            bins,
            torch.from_numpy(view.calcium_traces.df_over_f_norm[:128].T.reshape(-1, 4)),
        )
    assert counts == {128, 129}
