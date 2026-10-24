"""Compare against the actual approved loss source, not a reimplementation."""

from pathlib import Path
import subprocess
import types
from types import SimpleNamespace
import math
import pytest
import torch
import torch.nn.functional as F
from src.losses.sw_contrastive import SampleWiseContrastiveLoss


def old_loss_class():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            "8afa03c4325df12b2b147d2a8345466fdb624747:src/losses/sw_contrastive.py",
        ],
        text=True,
    )
    module = types.ModuleType("src.losses._approved_8afa_loss")
    module.__package__ = "src.losses"
    exec(
        compile(source, "8afa03c:src/losses/sw_contrastive.py", "exec"), module.__dict__
    )
    return module.SampleWiseContrastiveLoss


@pytest.mark.parametrize("dcl", [False, True])
@pytest.mark.parametrize("full_denom", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_default_false_is_bit_identical_to_8afa03c(dcl, full_denom, explicit):
    torch.manual_seed(42)
    old = old_loss_class()(
        tau=0.2, dcl=dcl, model=SimpleNamespace(emb_dim=7), full_denom=full_denom
    )
    kwargs = {"scale_same_view": False} if explicit else {}
    new = SampleWiseContrastiveLoss(
        tau=0.2,
        dcl=dcl,
        model=SimpleNamespace(emb_dim=7),
        full_denom=full_denom,
        **kwargs
    )
    new.load_state_dict(old.state_dict())
    x = torch.randn(7, 7, requires_grad=True)
    y = torch.randn(7, 7, requires_grad=True)
    a = x.detach().clone().requires_grad_()
    b = y.detach().clone().requires_grad_()
    batch = SimpleNamespace(
        collated=True,
        batch_size=2,
        device=torch.device("cpu"),
        view1=SimpleNamespace(unit_seqlen=[3, 4]),
        view2=SimpleNamespace(unit_seqlen=[4, 3]),
        match_idx=[
            torch.tensor([[0, 1, 2], [0, 2, 3]]),
            torch.tensor([[0, 2], [1, 2]]),
        ],
    )
    expected = old(x, y, batch, "train")
    actual = new(a, b, batch, "train")
    assert torch.equal(actual, expected)
    expected.backward()
    actual.backward()
    assert torch.equal(x.grad, a.grad) and torch.equal(y.grad, b.grad)
    for p, q in zip(old.parameters(), new.parameters()):
        assert torch.equal(p.grad, q.grad)


@pytest.mark.parametrize("dcl", [False, True])
def test_scaled_full_denom_matches_all_logits_over_tau(dcl):
    torch.manual_seed(7)
    x = F.normalize(torch.randn(3, 5), dim=-1).requires_grad_()
    y = F.normalize(torch.randn(4, 5), dim=-1).requires_grad_()
    match = torch.tensor([[0, 1, 2], [0, 2, 3]])
    tau = 0.2
    loss = SampleWiseContrastiveLoss(
        tau=tau,
        dcl=dcl,
        model=SimpleNamespace(emb_dim=5),
        projector=False,
        full_denom=True,
        scale_same_view=True,
    )
    numerator, denominator = loss.std_contrastive_loss(x, y, match)
    cross = x @ y.T / tau
    expected_numerator = 1 / tau - cross[match[0], match[1]].mean()
    if dcl:
        cross = cross.clone()
        cross[match[0], match[1]] = float("-inf")
    a = (x @ x.T / tau).fill_diagonal_(float("-inf"))
    b = (y @ y.T / tau).fill_diagonal_(float("-inf"))
    p = torch.cat([cross, a], 1)[match[0]]
    q = torch.cat([cross.T, b], 1)[match[1]]
    expected_denominator = (
        (p.logsumexp(1) - math.log(p.shape[1])).mean()
        + (q.logsumexp(1) - math.log(q.shape[1])).mean()
    ) / 2
    torch.testing.assert_close(numerator, expected_numerator, rtol=0, atol=0)
    torch.testing.assert_close(denominator, expected_denominator, rtol=0, atol=0)
    (numerator + denominator).backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
    old = old_loss_class()(
        tau=tau,
        dcl=dcl,
        model=SimpleNamespace(emb_dim=5),
        projector=False,
        full_denom=True,
    )
    assert not torch.equal(denominator, old.std_contrastive_loss(x, y, match)[1])


def test_scaled_flag_leaves_cross_only_branch_unchanged():
    torch.manual_seed(9)
    x = F.normalize(torch.randn(3, 5), dim=-1)
    y = F.normalize(torch.randn(3, 5), dim=-1)
    match = torch.tensor([[0, 1, 2], [0, 1, 2]])
    old = old_loss_class()(
        tau=0.2,
        dcl=True,
        model=SimpleNamespace(emb_dim=5),
        projector=False,
        full_denom=False,
    )
    new = SampleWiseContrastiveLoss(
        tau=0.2,
        dcl=True,
        model=SimpleNamespace(emb_dim=5),
        projector=False,
        full_denom=False,
        scale_same_view=True,
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            old.std_contrastive_loss(x, y, match), new.std_contrastive_loss(x, y, match)
        )
    )


def test_allgcl_config_uses_corrected_loss_with_legacy_opt_out():
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.2"):
        allgcl = compose(config_name="train_allgcl")
        legacy = compose(config_name="train_ca")
        control = compose(
            config_name="train_allgcl", overrides=["loss.scale_same_view=false"]
        )
    assert allgcl.loss.scale_same_view is True
    assert legacy.loss.scale_same_view is False
    assert control.loss.scale_same_view is False
    loss = SampleWiseContrastiveLoss(
        tau=0.2, dcl=True, model=SimpleNamespace(emb_dim=5), projector=False
    )
    assert loss.scale_same_view is False
