"""Audit a frozen head scaler on registered train windows without modifying training."""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.debias_runtime import DebiasRuntime
from src.utils import Precision, instantiate


def audit_checkpoint(checkpoint, output, device=None):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(saved["cfg"])
    if not (
        cfg.get("debias", {}).get("enabled", False)
        or cfg.get("debias", {}).get("track_normalizer", False)
    ):
        result = dict(status="bypass_no_fitted_normalizer")
    else:
        from contextlib import nullcontext

        device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        cfg.output_dir = str(Path(output).parent / "_normalizer_audit")
        model = instantiate(cfg.model, precision=Precision(cfg.precision)).to(device)
        model.load_state_dict(saved["model_state_dict"])
        precision = Precision(cfg.precision)
        trainer = SimpleNamespace(
            cfg=cfg,
            model=model,
            rank=0,
            world_size=1,
            is_distributed=False,
            device=device,
            add_checkpoint_items=lambda **kw: None,
            autocast=lambda: (
                torch.autocast("cuda", dtype=precision.dtype)
                if device.type == "cuda" and precision.dtype != torch.float32
                else nullcontext()
            ),
        )
        runtime = DebiasRuntime(trainer)
        try:
            runtime.load_checkpoint(saved)
            result = dict(
                status="audited",
                epoch=saved["epoch"],
                train_step=saved["train_step"],
                checkpoint=str(Path(checkpoint).resolve()),
                **runtime.audit(trainer)
            )
        finally:
            runtime.close()
    Path(output).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit_checkpoint(args.checkpoint, args.output)
