"""SPEC §8 19:25: detached, globally reduced running mean and SD, momentum .01."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def initialized():
    from src.debias import EMANormalizer

    normalizer = EMANormalizer(2, momentum=0.01)
    normalizer.observe(torch.tensor([[0.0, 1.0], [2.0, 5.0]]))
    normalizer.freeze()
    return normalizer


def test_ema_mean_and_sd_update_from_prefix_is_detached_and_checkpointed(tmp_path):
    norm = initialized()
    rows = torch.tensor([[2.0, 2.0], [6.0, 10.0]], requires_grad=True)
    norm.update(rows)
    torch.testing.assert_close(norm.mean, torch.tensor([1.03, 3.03]), rtol=0, atol=1e-6)
    torch.testing.assert_close(norm.std, torch.tensor([1.01, 2.02]), rtol=0, atol=1e-6)
    assert int(norm.ema_updates) == 1
    assert all(not b.requires_grad for b in norm.buffers())
    torch.save(norm.state_dict(), tmp_path / "ema.pt")
    restored = initialized()
    restored.load_state_dict(torch.load(tmp_path / "ema.pt", weights_only=False))
    for model in (norm, restored):
        model.update(torch.tensor([[5.0, 3.0], [9.0, 9.0]]))
    assert all(
        torch.equal(v, restored.state_dict()[k]) for k, v in norm.state_dict().items()
    )
    norm(rows).sum().backward()
    torch.testing.assert_close(rows.grad, 1 / norm.std.expand_as(rows))


def test_ema_drift_is_only_diagnostic_but_invalid_scales_fail():
    norm = initialized()
    bad = torch.tensor([[100.0, 101.0], [102.0, 103.0]])
    for _ in range(3):
        before = {k: v.clone() for k, v in norm.state_dict().items()}
        audit = norm.audit(bad)
        assert all(torch.equal(before[k], v) for k, v in norm.state_dict().items())
        assert audit["valid"] is True and audit["drift_diagnostic_only"] is True
        assert audit["drift_within_v1_threshold"] is False
    with pytest.raises(RuntimeError, match="Nonfinite|zero"):
        norm.update(torch.tensor([[float("nan"), 1.0], [2.0, 3.0]]))
    norm.std[0] = 0
    with pytest.raises(RuntimeError, match="Nonfinite|zero"):
        norm(bad)


def ema_worker(rank, world, rendezvous, output, empty):
    norm = initialized()
    if world > 1:
        dist.init_process_group(
            "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
        )
    try:
        rows = torch.tensor([[1.0, 4.0], [5.0, 2.0], [3.0, 8.0]])
        if world > 1:
            rows = (
                rows[:0]
                if empty and rank == 1
                else (rows if empty else (rows[:2] if rank == 0 else rows[2:]))
            )
        norm.update(rows)
        torch.save(norm.state_dict(), Path(output) / f"{world}-{rank}.pt")
    finally:
        if world > 1:
            dist.destroy_process_group()


@pytest.mark.parametrize("empty", [False, True])
def test_ema_ddp_global_moments_match_single_process(tmp_path, empty):
    mp.spawn(
        ema_worker,
        args=(2, str(tmp_path / "rendezvous"), str(tmp_path), empty),
        nprocs=2,
        join=True,
    )
    ema_worker(0, 1, None, str(tmp_path), empty)
    reference = torch.load(tmp_path / "1-0.pt", weights_only=False)
    for rank in (0, 1):
        actual = torch.load(tmp_path / f"2-{rank}.pt", weights_only=False)
        for key in reference:
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)


def test_empty_global_population_keeps_moments_and_records_skip():
    norm = initialized()
    mean = norm.mean.clone()
    std = norm.std.clone()
    norm.update(torch.empty(0, 2))
    assert torch.equal(norm.mean, mean) and torch.equal(norm.std, std)
    assert int(norm.ema_updates) == 0 and int(norm.ema_empty_steps) == 1
    assert norm.audit(torch.tensor([[0.0, 1.0], [2.0, 5.0]]))["ema_empty_steps"] == 1


def test_matched_population_in_controls_targets_and_pilots_excludes_panel_not_q2():
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from src.debias import DebiasPopulation
    from src.debias_runtime import DebiasRuntime

    rows = []
    for sid, e, date, setup in [
        ("fit1", 2, "2020-01-01", 1),
        ("fit3", 2, "2020-01-02", 3),
        ("panel", 2, "2023-02-14", 1),
        ("single", 1, "2020-01-03", 1),
        ("audit", 2, "2020-01-01", 1),
    ]:
        rows.append(
            dict(
                unit_id=sid,
                session_id=sid,
                split="train",
                split_group=f"Experimenter_{e}_{date}",
                experimenter=f"Experimenter {e}",
                setupid=setup,
                chirp_qidx=0.8,
                bar_qidx=0.8,
                supergroup="ON",
                label_condition="C1",
                label_condition_is_drug=False,
            )
        )
    frame = pd.DataFrame(rows)
    exposure = {name: 2 for name in frame.split_group.unique() if "2023" not in name}
    common = DebiasPopulation(frame, "split_group", {"audit"}, exposure)
    ids = np.array([r["session_id"] for r in rows])
    view = SimpleNamespace(
        unit_ids=ids,
        recording_ids=ids,
        unit_seqlen=[1] * 5,
        debias_control=np.ones(5, dtype=bool),
    )
    batch = SimpleNamespace(view1=view, view2=view, batch_size=5)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1000.0, 1000.0], [5.0, 6.0], [7.0, 8.0]])
    snapshots = []
    for enabled, pilot, target in [
        (False, False, "split_group"),
        (True, False, "split_group"),
        (True, False, "setupid"),
        (True, True, "split_group"),
        (True, True, "setupid"),
    ]:
        runtime = DebiasRuntime.__new__(DebiasRuntime)
        runtime.cfg = SimpleNamespace(enabled=enabled, pilot=pilot, target=target)
        runtime.common_population = common
        runtime.population = DebiasPopulation(frame, target, {"audit"}, exposure)
        runtime.normalizer = initialized()
        runtime.update_normalizer([y, y], batch)
        snapshots.append(runtime.normalizer.state_dict())
        if pilot:
            selected = runtime.rows([y], [view], 5)[0]
            assert not bool(
                (selected[:, 0] == 7).any()
            )  # Q2 excluded only from head fitting.
    for actual in snapshots[1:]:
        for key, value in snapshots[0].items():
            assert torch.equal(value, actual[key])
    # The panel outlier is excluded; E1 singleton and the Q2 field stay in common moments.
    torch.testing.assert_close(
        snapshots[0]["mean"], torch.tensor([1.03, 3.02]), rtol=0, atol=1e-6
    )


def test_consumers_initialize_from_exact_common_prefix_before_ema_updates():
    from src.debias import EMANormalizer
    from src.debias_runtime import freeze_common_normalizer

    reference = initialized()
    consumer = EMANormalizer(2)
    consumer.observe(torch.tensor([[100.0, 200.0], [102.0, 204.0]]))
    freeze_common_normalizer(consumer, reference.state_dict())
    assert all(
        torch.equal(v, consumer.state_dict()[k])
        for k, v in reference.state_dict().items()
    )
    consumer.update(torch.tensor([[2.0, 2.0], [6.0, 10.0]]))
    torch.testing.assert_close(
        consumer.mean, torch.tensor([1.03, 3.03]), rtol=0, atol=1e-6
    )
