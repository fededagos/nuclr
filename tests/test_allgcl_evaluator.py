import numpy as np
import pandas as pd
import torch


def test_evaluator_never_reads_test(synthetic_export, tmp_path):
    from src.evaluators.allgcl import AllGCLEvaluator

    md = pd.read_parquet(synthetic_export / "units.parquet")
    test = md.iloc[:4].copy()
    test["id"] = test["unit_id"] = [f"test_roi{i}" for i in range(4)]
    test["split"] = "test"
    test["group_id"] = 999
    path = tmp_path / "units.parquet"
    pd.concat([md, test]).to_parquet(path, index=False)
    evaluator = AllGCLEvaluator(path, n_units_subsample=80, seed=0)
    train = md[md.split == "train"]
    valid = md[md.split == "valid"]
    x = torch.tensor(np.eye(2)[train.group_id.to_numpy() - 1], dtype=torch.float32)
    clean = evaluator(x, train.id.to_numpy(), x, valid.id.to_numpy())
    contaminated = evaluator(
        torch.cat([x, torch.ones(4, 2) * 999]),
        np.r_[train.id.to_numpy(), test.id.to_numpy()],
        torch.cat([x, torch.ones(4, 2) * 999]),
        np.r_[valid.id.to_numpy(), test.id.to_numpy()],
    )
    assert clean == contaminated
    assert clean["val/group_id_balanced_accuracy"] == 1.0
    assert clean["val/supergroup_balanced_accuracy"] == 1.0
    assert clean["train/n_units"] == 40 and clean["val/n_units"] == 40


def test_evaluator_handles_missing_labels(synthetic_export, tmp_path):
    from src.evaluators.allgcl import AllGCLEvaluator

    md = pd.read_parquet(synthetic_export / "units.parquet")
    md = md.drop(columns=["group_id", "supergroup"])
    path = tmp_path / "unlabelled.parquet"
    md.to_parquet(path, index=False)
    evaluator = AllGCLEvaluator(path, seed=7)
    x = torch.tensor(np.random.default_rng(1).normal(size=(40, 8)), dtype=torch.float32)
    metrics = evaluator(x, md[md.split == "train"].id, x, md[md.split == "valid"].id)
    assert np.isfinite(metrics["train/rankme"]) and np.isfinite(metrics["val/rankme"])
    assert not any("balanced_accuracy" in key for key in metrics)


def test_evaluator_quality_thresholds(synthetic_export, tmp_path):
    from src.evaluators.allgcl import AllGCLEvaluator

    md = pd.read_parquet(synthetic_export / "units.parquet")
    md["chirp_qidx"] = 0.35
    md["bar_qidx"] = 0.6
    path = tmp_path / "low_quality.parquet"
    md.to_parquet(path, index=False)
    evaluator = AllGCLEvaluator(path)
    x = torch.randn(40, 8)
    metrics = evaluator(x, md[md.split == "train"].id, x, md[md.split == "valid"].id)
    assert not any("balanced_accuracy" in key for key in metrics)
    assert np.isfinite(metrics["val/rankme"])


def test_rankme_matches_covariance_entropy_for_few_units():
    from src.evaluators.rankme import rankme

    torch.set_num_threads(1)
    x = torch.tensor(
        np.random.default_rng(2).normal(size=(40, 256)), dtype=torch.float32
    )
    normalized = torch.nn.functional.normalize(x, dim=-1)
    eigenvalues = torch.linalg.eigvalsh(torch.cov(normalized.T))
    p = eigenvalues[eigenvalues > 0] / eigenvalues.sum()
    expected = torch.exp(-(p * p.log()).sum()).item()
    assert np.isclose(rankme(x), expected, rtol=1e-5)
