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


def test_shipped_bugeon_config_is_exact():
    from omegaconf import OmegaConf
    from src.utils import instantiate

    cfg = OmegaConf.load("configs/model/nuclr_v2ga_ca2.yaml")
    cfg.ctx_duration = OmegaConf.load("configs/train_ca.yaml").views.duration
    model = instantiate(cfg, precision=Precision("fp32"))
    assert model.num_latents == 32 and model.patch_size == 4
