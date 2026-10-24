import copy

import torch

from src.debias import DebiasHead, auxiliary_loss


def test_lambda_zero_preserves_encoder_projector_updates_and_moments():
    torch.manual_seed(42)
    encoder = torch.nn.Linear(5, 4)
    projector = torch.nn.Linear(4, 3)
    control = copy.deepcopy(torch.nn.Sequential(encoder, projector))
    before = torch.get_rng_state().clone()
    head = DebiasHead(4, 3, conditioned=False, seed=42)
    assert torch.equal(before, torch.get_rng_state())
    optim = torch.optim.AdamW(list(encoder.parameters()) + list(projector.parameters()))
    reference = torch.optim.AdamW(control.parameters())
    head_optim = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0)
    for _ in range(3):
        x = torch.randn(7, 5)
        y = encoder(x)
        base = projector(y).square().mean()
        aux = auxiliary_loss(head, y, torch.arange(7) % 3, torch.ones(7), 7, alpha=0)
        optim.zero_grad()
        head_optim.zero_grad()
        (base + aux).backward()
        reference.zero_grad()
        control(x).square().mean().backward()
        for module in (encoder, projector, control[0], control[1]):
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        assert torch.nn.utils.clip_grad_norm_(head.parameters(), 1e-6) > 1e-6
        optim.step()
        reference.step()
        head_optim.step()
        for actual, expected in zip(
            list(encoder.parameters()) + list(projector.parameters()),
            control.parameters(),
        ):
            assert torch.equal(actual, expected)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                assert torch.equal(
                    optim.state[actual][key], reference.state[expected][key]
                )


def test_population_excludes_panel_audit_and_single_setup_experimenters():
    import pandas as pd
    from src.debias import DebiasPopulation

    rows = []
    for e, date, setup, sid in [
        (2, "2020-01-01", 1, "fit1"),
        (2, "2020-01-02", 3, "fit3"),
        (2, "2023-02-14", 1, "panel"),
        (1, "2020-01-03", 1, "single"),
        (2, "2020-01-01", 1, "audit"),
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
    a = DebiasPopulation(
        frame,
        "split_group",
        {"audit"},
        {
            "Experimenter_2_2020-01-01": 2,
            "Experimenter_2_2020-01-02": 1,
            "Experimenter_1_2020-01-03": 1,
        },
    )
    b = DebiasPopulation(
        frame,
        "setupid",
        {"audit"},
        {"Experimenter_2_2020-01-01": 2, "Experimenter_2_2020-01-02": 1},
    )
    assert len(a.class_map) == 3
    assert len(b.class_map) == 2
    assert a.lookup(["panel", "audit"], ["panel", "audit"], pilot=True)[
        "eligible"
    ].tolist() == [False, False]
    assert a.lookup(["audit"], ["audit"], pilot=False)["eligible"].tolist() == [True]
    assert b.lookup(["single"], ["single"], pilot=False)["eligible"].tolist() == [False]
    # Type *values* never enter the head's metadata or weights.
    frame["supergroup"] = "a different known label"
    other = DebiasPopulation(frame, "setupid", {"audit"}, b.exposure)
    assert b.state_dict() == other.state_dict()


def test_global_row_cap_loses_mass_without_renormalizing():
    from src.debias import row_coefficients

    # Two views of one item, with one eligible cell each and enormous exposure correction.
    weights, info = row_coefficients(
        torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([100.0, 100.0]), 2
    )
    assert weights.tolist() == [5.0, 5.0]
    assert info["lost_mass"] == 90


def test_normalizer_freezes_checkpoint_roundtrip_and_rejects_two_bad_audits():
    from src.debias import FrozenNormalizer
    import pytest

    normalizer = FrozenNormalizer(2)
    normalizer.observe(torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    normalizer.freeze()
    assert torch.equal(normalizer(torch.tensor([[1.0, 2.0]])), torch.zeros(1, 2))
    restored = FrozenNormalizer(2)
    restored.load_state_dict(normalizer.state_dict())
    assert torch.equal(restored.std, normalizer.std)
    bad = torch.tensor([[100.0, 101.0], [102.0, 103.0]])
    assert not normalizer.audit(bad)["valid"]
    with pytest.raises(RuntimeError, match="two consecutive"):
        normalizer.audit(bad)


def test_calibration_uses_only_prescribed_steps_and_frozen_audit_ce_gate():
    from src.debias import Calibration
    import pytest

    c = Calibration()
    assert [s for s in range(1, 1251) if c.is_measurement(s)] == list(
        range(275, 1251, 25)
    )
    for step in range(275, 1251, 25):
        c.record(step, ratio=1000.0, audit_ce=0.94, prior_ce=1.0)
    assert c.finish()["lambda_max"] == 0.0001
    c.points[-1]["audit_ce"] = 1.5
    with pytest.raises(RuntimeError, match="audit CE"):
        c.finish()


def test_current_condition_survives_row_filtering_and_collation():
    from src.dataset import ViewData
    from src.debias import attach_condition
    import numpy as np

    views = []
    for ids, control in [(["u3", "u1"], True), (["u2"], False)]:
        view = ViewData(
            {"enc_input": {"unit_seqlen": len(ids)}},
            np.array(ids),
            np.array(["s"] * len(ids)),
        )
        attach_condition(view, control)
        views.append(view)
    batch = ViewData.collate(views)
    assert batch.unit_ids.tolist() == ["u3", "u1", "u2"]
    assert batch.debias_control.tolist() == [True, True, False]


def test_audit_draw_never_excludes_a_single_field_animal_or_last_fit_field():
    from src.debias import audit_fields

    groups = {
        "single": "a",
        **{f"b{i}": "b" for i in range(3)},
        **{f"c{i}": "c" for i in range(6)},
    }
    chosen = audit_fields(groups, field_animals=groups, total_fields=20)
    assert len(chosen) == 2
    assert "single" not in chosen
    assert set(groups) - set(chosen)
    for animal in set(groups.values()):
        assert any(groups[f] == animal and f not in chosen for f in groups)


def test_head_weighted_priors_are_ce_optimal_and_conditioned_on_e():
    from src.debias import HeadPrior

    prior = HeadPrior(2, conditioned=True)
    # Weights already contain actual hierarchical means, exposure and row cap.
    prior.observe(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([1, 1, 1, 6]),
        torch.tensor([0.1, 0.2, 0.7, 1.0]),
    )
    torch.testing.assert_close(
        prior.probabilities(torch.tensor([0, 1, 1]), torch.tensor([1, 1, 6])),
        torch.tensor([0.3, 0.7, 1.0]),
    )
    state = prior.state_dict()
    restored = HeadPrior(2, conditioned=True)
    restored.load_state_dict(state)
    assert torch.equal(restored.counts, prior.counts)


def test_cell_share_within_animal_does_not_macro_balance_fields():
    import pandas as pd
    from src.debias import DebiasPopulation

    frame = pd.DataFrame(
        [
            dict(
                unit_id=str(i),
                session_id="small" if i == 0 else "large",
                split="train",
                split_group="animal",
                experimenter="Experimenter 1",
                setupid=1,
                chirp_qidx=0.8,
                bar_qidx=0.8,
                supergroup="ON",
                label_condition="C1",
                label_condition_is_drug=False,
            )
            for i in range(11)
        ]
    )
    p = DebiasPopulation(
        frame,
        "split_group",
        set(),
        {
            "animal": 2.0,
            "__base_items__": 2.0,
            "__field_exposure__": {"small": 1.0, "large": 1.0},
        },
    )
    rows = p.lookup(frame.unit_id, frame.session_id)
    assert abs(rows["correction"][0] - rows["correction"][1] / 10) < 1e-12


def test_only_last_five_audits_require_supported_observed_priors():
    from src.debias import Calibration
    import pytest

    c = Calibration()
    for step in range(275, 1251, 25):
        c.record(step, 1000.0, 0.94, None if step < 1150 else 1.0)
    assert c.finish()["accepted"] is True
    with pytest.raises(RuntimeError, match="prior"):
        Calibration().record(1150, 1000.0, 0.94, None)
