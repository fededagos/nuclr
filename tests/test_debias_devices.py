"""Exercise mixed CPU/CUDA snapshot placement on CPU-only allocations."""

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from fixtures.device_checks import optimizer_snapshot, assert_optimizer_devices


@pytest.mark.parametrize("index", [0, 1])
def test_snapshot_accepts_host_step_and_cuda_moments_without_moving_live_state(index):
    with FakeTensorMode():
        state = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.ones(3, device=f"cuda:{index}"),
            "exp_avg_sq": torch.ones(3, device=f"cuda:{index}"),
        }
        devices = {key: tensor.device for key, tensor in state.items()}
        optimizer = SimpleNamespace(state={object(): state})
        snapshot = optimizer_snapshot([optimizer])
        assert snapshot.device.type == "cpu" and snapshot.shape == (7,)
        assert {key: tensor.device for key, tensor in state.items()} == devices


def test_meta_adamw_restore_moves_moments_but_keeps_host_counter():
    # Actual AdamW initialization/load logic on distinct devices, without CUDA.
    cpu = torch.nn.Parameter(torch.ones(3))
    original = torch.optim.AdamW([cpu], lr=1e-4, foreach=False)
    cpu.grad = torch.ones_like(cpu)
    original.step()
    parameter = torch.nn.Parameter(torch.ones(3, device="meta"))
    restored = torch.optim.AdamW([parameter], lr=1e-4, foreach=False)
    restored.load_state_dict(original.state_dict())
    assert restored.state[parameter]["step"].device.type == "cpu"
    assert_optimizer_devices(restored)
    parameter.grad = torch.ones_like(parameter)
    restored.step()
    assert_optimizer_devices(restored)
    restored.state[parameter]["exp_avg"] = torch.ones(3)
    with pytest.raises(AssertionError, match="exp_avg device"):
        assert_optimizer_devices(restored)
