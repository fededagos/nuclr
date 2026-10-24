"""CPU regressions for rank streams without changing the rank-0 oracle."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.trainers.contrastive import ContrastiveTrainer, train_loader_randomness
from src.utils import seed_everything


class RandomDraws(Dataset):
    def __init__(self):
        self.np_rng = np.random.default_rng(42)

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.tensor(
            [
                torch.rand(()).item(),
                np.random.random(),
                random.random(),
                self.np_rng.random(),
            ],
            dtype=torch.float64,
        )


def draws(rank, legacy=False):
    seed_everything(42)
    loader = DataLoader(
        RandomDraws(),
        batch_size=2,
        num_workers=2,
        multiprocessing_context="fork",
        **({} if legacy else train_loader_randomness(42, rank)),
    )
    return torch.cat(list(loader))


def test_rank_zero_worker_draws_are_bit_identical_to_legacy():
    assert train_loader_randomness(42, 0) == {}
    assert torch.equal(draws(0), draws(0, legacy=True))


def test_nonzero_rank_workers_are_distinct_and_reproducible():
    rank0, rank1 = draws(0), draws(1)
    assert torch.equal(rank1, draws(1))
    assert not torch.eq(rank0, rank1).any()
    # The explicit default_rng also differs between workers within rank 1.
    assert not torch.equal(rank1[:2, 3], rank1[2:4, 3])


@pytest.mark.parametrize("rank,distributed", [(0, False), (0, True), (1, True)])
def test_setup_seeds_only_nonzero_rank_after_both_broadcasts(
    monkeypatch, rank, distributed
):
    seed_everything(42)
    trainer = ContrastiveTrainer.__new__(ContrastiveTrainer)
    trainer.cfg = OmegaConf.create(
        {
            "seed": 42,
            "model": {"ctx_duration": 1.0},
            "views": {"duration": 1.0},
            "target_steps": None,
            "ckpt": {"resume": False},
        }
    )
    trainer.rank, trainer.is_distributed = rank, distributed
    trainer.train_loader = SimpleNamespace(dataset=RandomDraws())
    trainer.model, trainer.loss = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    for name in (
        "_setup_model",
        "_setup_train_loader",
        "_setup_val_loader",
        "_setup_embedding_collectors",
        "_setup_evaluator",
        "_setup_clean_monitor_loader",
        "_setup_optimizer",
    ):
        monkeypatch.setattr(trainer, name, lambda: None)
    monkeypatch.setattr(trainer, "_apply_ckpt", lambda _: None)
    broadcasts = []

    def broadcast(module):
        torch.rand(3)  # Preserve any pre-existing consumption before the new hook.
        broadcasts.append(torch.get_rng_state().clone())
        return module

    monkeypatch.setattr(trainer, "make_ddp", broadcast)
    trainer.setup(None)
    assert len(broadcasts) == 2
    if rank == 0:
        assert torch.equal(torch.get_rng_state(), broadcasts[-1])
        assert (
            trainer.train_loader.dataset.np_rng.bit_generator.state
            == np.random.default_rng(42).bit_generator.state
        )
    else:
        expected = torch.Generator().manual_seed(43)
        assert torch.equal(torch.get_rng_state(), expected.get_state())
        assert (
            trainer.train_loader.dataset.np_rng.bit_generator.state
            == np.random.default_rng(43).bit_generator.state
        )
        assert np.random.random() == np.random.RandomState(43).random_sample()
        assert random.random() == random.Random(43).random()


def test_rank_loader_generator_does_not_consume_global_rng():
    seed_everything(42)
    before = torch.get_rng_state().clone()
    options = train_loader_randomness(42, 1)
    assert options["generator"].initial_seed() == 43
    assert torch.equal(before, torch.get_rng_state())


def test_window_sampler_seed_and_strided_partition_are_unchanged():
    from temporaldata import Interval
    from src.samplers import RandomFixedWindowSampler, DistributedSamplerWrapper

    intervals = {"field": Interval(0.0, 30.0)}
    seed_everything(42)
    expected = list(RandomFixedWindowSampler(intervals, 1.0, seed=42))
    shards = []
    for rank in (0, 1):
        seed_everything(42 + rank)
        shards.append(
            list(
                DistributedSamplerWrapper(
                    RandomFixedWindowSampler(intervals, 1.0, seed=42),
                    num_replicas=2,
                    rank=rank,
                )
            )
        )
    assert [index for pair in zip(*shards) for index in pair] == expected[
        : len(shards[0]) * 2
    ]
