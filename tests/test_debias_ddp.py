"""Same-item 1-process vs 2-rank objective, gradients, clipped updates and resume.

CPU/Gloo by default; ADV_TEST_DEVICE=cuda switches the identical test to 2 GPUs.
Tolerances fixed before GPU execution: loss atol=1e-6; gradients rtol=1e-5,
atol=1e-6; updates/moments rtol=1e-5, atol=1e-7 (small fp32 toy networks).
"""

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.dataset import ViewData, TwoViewData
from src.debias import (
    DebiasHead,
    FrozenNormalizer,
    HeadPrior,
    auxiliary_loss,
    row_coefficients,
)
from src.losses.sw_contrastive import SampleWiseContrastiveLoss
from fixtures.device_checks import (
    optimizer_snapshot,
    assert_optimizer_devices,
    assert_module_device,
)


def batch_for(items, device):
    lengths = [3, 5, 4]
    views = []
    for v in range(2):
        entries = []
        for i in items:
            n = lengths[i]
            vd = ViewData(
                {}, np.array([f"{i}:{j}" for j in range(n)]), np.array([str(i)] * n)
            )
            entries.append(vd)
        views.append(ViewData.collate(entries))
    batch = TwoViewData(*views, [torch.arange(lengths[i]).repeat(2, 1) for i in items])
    batch._batch_size = len(items)
    batch._collated = True
    batch.to(device)
    indices = torch.cat(
        [torch.arange(sum(lengths[:i]), sum(lengths[: i + 1])) for i in items]
    ).to(device)
    x = torch.arange(60, device=device, dtype=torch.float32).reshape(12, 5) / 40
    return batch, x[indices], x[indices] * 0.9 + 0.1


def run_steps(rank, world, rendezvous, output, empty_rank, conditioned):
    use_cuda = os.environ.get("ADV_TEST_DEVICE") == "cuda"
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if world == 2:
        dist.init_process_group(
            "nccl" if use_cuda else "gloo",
            init_method=f"file://{rendezvous}",
            rank=rank,
            world_size=world,
        )
    torch.manual_seed(42)
    encoder = torch.nn.Linear(5, 4).to(device)
    encoder.emb_dim = 4
    loss = SampleWiseContrastiveLoss(
        0.2, True, encoder, scale_same_view=True, full_denom=True
    ).to(device)
    num_classes = 2 if conditioned else 3
    head = DebiasHead(4, num_classes, conditioned=conditioned, seed=42).to(device)
    if world == 2:
        kwargs = dict(device_ids=[rank]) if use_cuda else {}
        encoder, loss, head = [DDP(m, **kwargs) for m in (encoder, loss, head)]
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(loss.parameters()), lr=2.5e-4
    )
    head_optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0)
    prior = HeadPrior(num_classes, conditioned).to(device)
    normalizer = FrozenNormalizer(4).to(device)
    # Both topologies fit the same two rows, sharded across ranks.
    normalization_rows = torch.tensor([[-1.0] * 4, [1.0] * 4], device=device)
    normalizer.observe(normalization_rows[rank::world])
    normalizer.freeze()
    items = ([0, 1] if rank == 0 else [2]) if world == 2 else [0, 1, 2]
    snapshots = []
    for step in range(3):
        batch, x1, x2 = batch_for(items, device)
        if world == 2 and step == 1:
            raw_encoder, raw_loss, raw_head = [m.module for m in (encoder, loss, head)]
            measure_y = [raw_encoder(x) for x in (x1, x2)]
            measure_base = raw_loss(*measure_y, batch, prefix="measure")
            labels = torch.arange(len(x1), device=device) % num_classes
            measure_aux = auxiliary_loss(
                raw_head,
                measure_y[0],
                labels,
                torch.ones(len(x1), device=device),
                len(items),
                alpha=1.0,
                experimenter=torch.zeros_like(labels),
            )
            torch.autograd.grad(
                measure_base, tuple(raw_encoder.parameters()), retain_graph=True
            )
            torch.autograd.grad(measure_aux, tuple(raw_encoder.parameters()))
            del measure_y, measure_base, measure_aux
        ys = [encoder(x) for x in (x1, x2)]
        base = loss(*ys, batch, prefix="train")
        selected, labels, item_ids, view_ids, corrections, experimenters = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for v, y in enumerate(ys):
            offset = 0
            for local, global_id in enumerate(items):
                n = [3, 5, 4][global_id]
                if not (empty_rank and global_id == 2):
                    selected.append(y[offset : offset + n])
                    labels += [global_id % num_classes] * n
                    experimenters += [global_id] * n
                    item_ids += [local] * n
                    view_ids += [v] * n
                    corrections += [
                        [17.0, 0.4, 2.0][global_id]
                    ] * n  # global cap active
                offset += n
        # A graph-connected empty tensor still reaches the wrapped head on rank 1.
        features = torch.cat(selected) if selected else ys[0][:0]
        tensor = lambda a, dtype: torch.tensor(a, dtype=dtype, device=device)
        weights, info = row_coefficients(
            tensor(item_ids, torch.long),
            tensor(view_ids, torch.long),
            tensor(corrections, torch.float32),
            len(items),
        )
        prior.observe(
            tensor(labels, torch.long),
            tensor(experimenters, torch.long),
            weights / info["global_base_items"],
        )
        aux = auxiliary_loss(
            head,
            normalizer(features),
            tensor(labels, torch.long),
            weights,
            len(items),
            alpha=0.01,
            experimenter=tensor(experimenters, torch.long),
        )
        optimizer.zero_grad()
        head_optimizer.zero_grad()
        (base + aux).backward()
        modules = [encoder, loss, head]
        gradients = (
            torch.cat([p.grad.flatten() for m in modules for p in m.parameters()])
            .detach()
            .cpu()
        )
        for m in modules:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1e-4)  # deliberately active
        optimizer.step()
        head_optimizer.step()
        for opt in (optimizer, head_optimizer):
            assert_optimizer_devices(opt)
        states = [copy.deepcopy(o.state_dict()) for o in (optimizer, head_optimizer)]
        # Checkpoint/resume the separate optimizers, including their moments.
        if step == 1:
            optimizer.load_state_dict(states[0])
            head_optimizer.load_state_dict(states[1])
            for module in (normalizer, prior):
                module.load_state_dict(
                    {
                        name: buffer.detach().cpu().clone()
                        for name, buffer in module.state_dict().items()
                    }
                )
        for opt in (optimizer, head_optimizer):
            assert_optimizer_devices(opt)
        for module in (*modules, normalizer, prior):
            assert_module_device(module, device)
        parameters = torch.cat(
            [p.detach().flatten() for m in modules for p in m.parameters()]
        ).cpu()
        moments = optimizer_snapshot((optimizer, head_optimizer))
        snapshots.append(
            dict(
                loss=float((base + aux).detach()),
                gradients=gradients,
                parameters=parameters,
                moments=moments,
                prior_counts=prior.counts.detach().cpu().clone(),
                normalizer_buffers=torch.cat(
                    [
                        buffer.detach().cpu().double().flatten()
                        for buffer in normalizer.buffers()
                    ]
                ),
            )
        )
    torch.save(snapshots, Path(output) / f"{world}-{rank}.pt")
    if world == 2:
        dist.destroy_process_group()


@pytest.mark.parametrize("empty_rank", [False, True])
@pytest.mark.parametrize("conditioned", [False, True])
def test_ddp_weighted_objective_head_sync_clip_and_resume(
    tmp_path, empty_rank, conditioned
):
    mp.spawn(
        run_steps,
        args=(2, str(tmp_path / "rdzv"), str(tmp_path), empty_rank, conditioned),
        nprocs=2,
        join=True,
    )
    run_steps(0, 1, None, str(tmp_path), empty_rank, conditioned)
    reference = torch.load(tmp_path / "1-0.pt", weights_only=False)
    actual = [torch.load(tmp_path / f"2-{r}.pt", weights_only=False) for r in range(2)]
    for step, expected in enumerate(reference):
        for rank in range(2):
            assert torch.equal(
                actual[rank][step]["normalizer_buffers"], expected["normalizer_buffers"]
            )
            torch.testing.assert_close(
                torch.tensor(actual[rank][step]["loss"]),
                torch.tensor(expected["loss"]),
                rtol=0,
                atol=1e-6,
            )
            for key, atol in [
                ("gradients", 1e-6),
                ("parameters", 1e-7),
                ("moments", 1e-7),
                ("prior_counts", 1e-7),
            ]:
                torch.testing.assert_close(
                    actual[rank][step][key], expected[key], rtol=1e-5, atol=atol
                )
                assert torch.equal(actual[0][step][key], actual[1][step][key])
