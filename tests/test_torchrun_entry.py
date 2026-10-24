import importlib.util
from pathlib import Path
from omegaconf import OmegaConf


def test_torchrun_uses_external_rank_and_rendezvous(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "nuclr_train_entry", Path(__file__).resolve().parents[1] / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for k, v in {
        "LOCAL_RANK": "1",
        "RANK": "1",
        "WORLD_SIZE": "2",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29517",
    }.items():
        monkeypatch.setenv(k, v)
    calls = []
    monkeypatch.setattr(
        module,
        "train",
        lambda rank, world, cfg: calls.append(
            (rank, world, cfg.ddp.master_addr, cfg.ddp.master_port)
        ),
    )
    cfg = OmegaConf.create({"ddp": {"master_addr": "localhost", "master_port": None}})
    assert module.launch_from_torchrun(cfg)
    assert calls == [(1, 2, "127.0.0.1", 29517)]


def test_normal_entry_stays_on_original_path(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "nuclr_train_entry_plain", Path(__file__).resolve().parents[1] / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(
        module,
        "train",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not launch")),
    )
    assert module.launch_from_torchrun(OmegaConf.create({})) is False
