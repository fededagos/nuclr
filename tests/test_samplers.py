import numpy as np
import pytest
from temporaldata import Interval
from src.samplers import DistributedSamplerWrapper, RandomFixedWindowSampler


@pytest.mark.parametrize("distributed", [False, True])
def test_requested_epoch_survives_cached_length(distributed):
    def make():
        sampler = RandomFixedWindowSampler(
            {"field": Interval(np.array([0.0, 200.0]), np.array([120.0, 320.0]))},
            32.768,
            seed=42,
        )
        return DistributedSamplerWrapper(sampler) if distributed else sampler

    cached = make()
    assert len(cached) > 0
    cached.set_epoch(4)
    fresh = make()
    fresh.set_epoch(4)
    assert list(cached) == list(fresh)


def test_empty_recording_does_not_abort_nonempty_epoch():
    sampler = RandomFixedWindowSampler(
        {"long": Interval(0.0, 20.0), "short": Interval(100.0, 101.1)},
        window_length=1.0,
        seed=42,
    )
    indices = list(sampler)
    assert indices and all(index.recording_id == "long" for index in indices)


def test_whole_empty_epoch_raises_clearly():
    sampler = RandomFixedWindowSampler(
        {"short": Interval(0.0, 1.1)}, window_length=1.0, seed=42
    )
    with pytest.raises(ValueError, match="epoch.*empty|empty.*epoch"):
        list(sampler)
