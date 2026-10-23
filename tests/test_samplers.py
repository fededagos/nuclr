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
