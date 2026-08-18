from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch


def create_logger(log_path: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger("G4RegFormer")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(stream)
    if rank == 0:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
    return logger


def unwrap_model(model):
    model = model.module if hasattr(model, "module") else model
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def save_model(
    run_dir: Path,
    epoch: int,
    model,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    args,
) -> Path:
    checkpoint_path = run_dir / f"checkpoint-{epoch}.pth"
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scaler": scaler.state_dict(),
            "args": args,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_model(
    checkpoint_path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1


def append_epoch_log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
