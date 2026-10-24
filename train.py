import hydra
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch
import torch.multiprocessing as mp

from src.utils import cleanup_ddp, get_open_port
from src.logger import get_cli_logger
from src.trainers.trainer import Trainer
from src.utils import instantiate


def train(rank: int, world_size: int, cfg: DictConfig):

    if cfg.get("device", "auto") != "cpu" and torch.cuda.is_available():
        torch.cuda.set_device(rank)  # avoid putting things on GPU0 by default

    trainer: Trainer = instantiate(
        cfg.trainer,
        cfg=cfg,
        rank=rank,
        world_size=world_size,
    )
    trainer.train()
    cleanup_ddp()


def launch_from_torchrun(cfg: DictConfig) -> bool:
    """Use torchrun's single-node ranks without spawning a second process tree."""
    if "LOCAL_RANK" not in os.environ:
        return False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if rank != local_rank:
        raise ValueError("This entry point currently supports single-node torchrun")
    cfg.ddp.master_addr = os.environ["MASTER_ADDR"]
    cfg.ddp.master_port = int(os.environ["MASTER_PORT"])
    train(rank, world_size, cfg)
    return True


@hydra.main(version_base="1.2", config_path="./configs", config_name="train_ca")
def main(cfg: DictConfig):
    hydra_choices = HydraConfig.get().runtime.choices
    with open_dict(cfg):
        cfg.hydra_choices = hydra_choices

    if launch_from_torchrun(cfg):
        return

    num_gpus = 0 if cfg.get("device", "auto") == "cpu" else torch.cuda.device_count()

    if num_gpus > 1 or cfg.ddp.force:
        if cfg.ddp.master_port is None:
            cfg.ddp.master_port = get_open_port()
            get_cli_logger().info(f"DDP MASTER_PORT = {cfg.ddp.master_port}")

    mp.set_start_method("spawn")
    for rank in range(1, num_gpus):
        mp.Process(target=train, args=(rank, num_gpus, cfg)).start()

    train(0, max(1, num_gpus), cfg)


if __name__ == "__main__":
    main()
