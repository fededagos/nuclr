"""Test diagnostics must preserve AdamW's legitimate heterogeneous state devices."""

import torch


def optimizer_snapshot(optimizers):
    return torch.cat(
        [
            state[key].detach().flatten().cpu()
            for optimizer in optimizers
            for state in optimizer.state.values()
            for key in ("step", "exp_avg", "exp_avg_sq")
        ]
    )


def assert_optimizer_devices(optimizer):
    """Moments follow their parameter; a noncapturable/nonfused step may be CPU."""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                assert (
                    state[key].device == parameter.device
                ), f"{key} device: {state[key].device} != {parameter.device}"
            step_device = state["step"].device
            if group.get("capturable") or group.get("fused"):
                assert step_device == parameter.device
            else:
                assert step_device.type == "cpu" or step_device == parameter.device


def assert_module_device(module, device):
    device = torch.device(device)
    for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
        assert tensor.device == device, f"{name} device: {tensor.device} != {device}"
