"""Runtime All-GCL selection and run configuration shared by train and forward."""

import json
from pathlib import Path
from omegaconf import OmegaConf, open_dict


def read_splits(splits_file):
    sessions = json.loads(Path(splits_file).read_text())["sessions"]
    flattened = [sid for split in ("train", "valid", "test") for sid in sessions[split]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Session leakage or duplicate session in splits.json")
    return sessions


def split_selection(data_cfg, splits):
    sessions = read_splits(data_cfg.splits_file)
    selected = []
    for split in splits:
        if split not in sessions:
            raise ValueError(f"Unknown split {split}")
        selected.extend(sessions[split])
    if not selected:
        raise ValueError(f"Empty session selection for {splits}")
    return OmegaConf.create(
        [{"selection": [{"brainset": data_cfg.brainset, "sessions": selected}]}]
    )


def prepare_allgcl_config(cfg):
    if cfg.data.get("brainset") != "euler_allgcl_2025":
        return
    with open_dict(cfg):
        cfg.data.train_dataset = split_selection(cfg.data, [cfg.data.train_split])
        cfg.data.val_dataset = split_selection(cfg.data, [cfg.data.val_split])
        cfg.output_dir = str(Path(cfg.output_dir).expanduser().resolve())
        cfg.ckpt.dir = str(Path(cfg.output_dir) / "ckpt")
        cfg.wandb.dir = str(Path(cfg.output_dir) / "wandb")
        cfg.wandb.mode = "offline"
        if cfg.get("resume") == "auto" and cfg.ckpt.load_from is None:
            checkpoints = list(Path(cfg.ckpt.dir).glob("epoch_*.pt"))
            checkpoints = [
                p for p in checkpoints if p.stem.removeprefix("epoch_").isdigit()
            ]
            if checkpoints:
                cfg.ckpt.load_from = str(
                    max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
                )
                cfg.ckpt.resume = True
