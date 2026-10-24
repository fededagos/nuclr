"""Compare the actual lambda-selector measurement on SP and unequal Gloo shards."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.debias import DebiasHead, auxiliary_loss, row_coefficients
from src.debias_runtime import DebiasRuntime
from src.losses.sw_contrastive import SampleWiseContrastiveLoss
from test_debias_ddp import batch_for


class FixedRowsMeasurement(DebiasRuntime):
    """Use the applied weighted-CE objective; keep measure() completely unchanged."""

    def head_loss(self, ys, batch, alpha, head=None, update_prior=False):
        features, labels, experiments, corrections, items, views = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for v, (y, view) in enumerate(zip(ys, (batch.view1, batch.view2))):
            start = 0
            for local, length in enumerate(view.unit_seqlen):
                global_item = int(view.unit_ids[start].split(":")[0])
                if not (self.empty and global_item == 2):
                    features.append(y[start : start + length])
                    labels.extend([global_item % self.classes] * length)
                    experiments.extend([global_item] * length)
                    corrections.extend([[17.0, 0.4, 2.0][global_item]] * length)
                    items.extend([local] * length)
                    views.extend([v] * length)
                start += length
        x = torch.cat(features) if features else ys[0][:0]
        tensor = lambda a, dtype: torch.tensor(a, dtype=dtype)
        weights, info = row_coefficients(
            tensor(items, torch.long),
            tensor(views, torch.long),
            tensor(corrections, torch.float32),
            batch.batch_size,
        )
        aux = auxiliary_loss(
            head or self.head,
            x,
            tensor(labels, torch.long),
            weights,
            batch.batch_size,
            alpha,
            tensor(experiments, torch.long),
        )
        return aux, info, []

    def _log(self, trainer, values):
        self.measured_ratio = values["unit_alpha_gradient_ratio"]


def measure_worker(rank, world, rendezvous, output, conditioned, empty):
    if world == 2:
        dist.init_process_group(
            "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
        )
    try:
        torch.manual_seed(42)
        model = torch.nn.Linear(5, 4)
        model.emb_dim = 4
        loss = SampleWiseContrastiveLoss(
            0.2, True, model, full_denom=True, scale_same_view=True
        )
        head = DebiasHead(4, 2 if conditioned else 3, conditioned=conditioned)
        if world == 2:
            model, loss, head = [DDP(m) for m in (model, loss, head)]
        items = ([0, 1] if rank == 0 else [2]) if world == 2 else [0, 1, 2]
        batch, x1, x2 = batch_for(items, torch.device("cpu"))
        batch.view1.data = {"enc_input": {"input": x1}}
        batch.view2.data = {"enc_input": {"input": x2}}
        runtime = FixedRowsMeasurement.__new__(FixedRowsMeasurement)
        runtime.cfg = SimpleNamespace(pilot=False)
        runtime.head = head
        runtime.device = torch.device("cpu")
        runtime.classes = 2 if conditioned else 3
        runtime.empty = empty
        trainer = SimpleNamespace(
            model=model,
            loss=loss,
            device=torch.device("cpu"),
            is_distributed=world == 2,
            world_size=world,
            autocast=nullcontext,
        )
        runtime.measure(trainer, batch, 275)
        Path(output, f"{world}-{rank}.txt").write_text(repr(runtime.measured_ratio))
    finally:
        if world == 2:
            dist.destroy_process_group()


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_actual_measure_global_ratio_equals_single_process(
    tmp_path, conditioned, empty
):
    mp.spawn(
        measure_worker,
        args=(2, str(tmp_path / "rdzv"), str(tmp_path), conditioned, empty),
        nprocs=2,
        join=True,
    )
    measure_worker(0, 1, None, str(tmp_path), conditioned, empty)
    reference = float((tmp_path / "1-0.txt").read_text())
    ranks = [float((tmp_path / f"2-{rank}.txt").read_text()) for rank in (0, 1)]
    assert reference > 0 and ranks[0] == ranks[1]
    for actual in ranks:
        torch.testing.assert_close(
            torch.tensor(actual), torch.tensor(reference), rtol=1e-5, atol=1e-6
        )
