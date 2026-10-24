"""End-to-end default-OFF oracle against the actual complete pushed source."""

import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import torch

BASE = "d0b2c03473af170195864cb2305fedb279b50e9d"
SOURCE = Path(__file__).resolve().parents[1]


def test_default_off_complete_pipeline_is_bit_identical_to_d0b2c03(
    synthetic_export, tmp_path
):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    archive = subprocess.check_output(
        ["git", "archive", BASE, "src", "configs", "train.py"], cwd=SOURCE
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        assert all(
            not Path(m.name).is_absolute() and ".." not in Path(m.name).parts
            for m in tar.getmembers()
        )
        tar.extractall(old)
    for name in ("src", "configs"):
        shutil.copytree(
            SOURCE / name, new / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    shutil.copyfile(SOURCE / "train.py", new / "train.py")
    checkpoints = []
    for tree in (old, new):
        output = tree / "output"
        command = [
            sys.executable,
            str(tree / "train.py"),
            "--config-name",
            "train_allgcl",
            f"data.data_root={synthetic_export}/processed",
            f"data.splits_file={synthetic_export}/splits.json",
            f"data.units_file={synthetic_export}/units.parquet",
            f"output_dir={output}",
            "device=cpu",
            "precision=fp32",
            "num_workers=0",
            "batch_size=2",
            "target_steps=3",
            "model.dim=32",
            "model.self_heads=2",
            "model.dim_head=16",
            "model.t_layers=0",
            "model.st_layers=0",
            "data.evaluator=null",
        ]
        result = subprocess.run(
            command,
            cwd=tree,
            env=dict(
                os.environ,
                PYTHONPATH=str(tree),
                CUDA_VISIBLE_DEVICES="",
                PYTHONNOUSERSITE="1",
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            ),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        checkpoints.append(
            torch.load(
                output / "ckpt/epoch_0.pt", map_location="cpu", weights_only=False
            )
        )

    def compare(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y)
        else:
            assert a == b

    for key in (
        "model_state_dict",
        "loss_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    ):
        compare(checkpoints[0][key], checkpoints[1][key])
