from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from pretrain import model as pretrain_models
from .dataset import (
    CRERegGeneDataset, collate_genes, split_gene_indices, target_scales,
)
from .engine import evaluate, train_epoch
from .model import CRERegGeneModel


def parse_args():
    parser = argparse.ArgumentParser("CRE-to-gene Pol II regression")
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/preprocessed/downstream_data/CRERegGene/CRE_500kb_Reg_Gene_mm10/train_data/cre_reg_gene_dataset.h5"),
    )
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("downstream/CRERegGene/output"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--target-peak-quantile", type=float, default=0.995)
    parser.add_argument("--nonzero-weight", type=float, default=2.0)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_logger(path: Path):
    logger = logging.getLogger("CRERegGene")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_pretrained(path: Path, feature_names: list[str]):
    checkpoint = torch.load(path, map_location="cpu")
    saved = checkpoint.get("args", {})
    model_name = saved.get("model", "mae_bin_large") if isinstance(saved, dict) else saved.model
    model = pretrain_models.__dict__[model_name](feature_names=feature_names)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed = {
        "input_norm.weight", "input_norm.bias",
        "decoder_input_norm.weight", "decoder_input_norm.bias",
    }
    if set(missing) - allowed or unexpected:
        raise RuntimeError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    return model, model_name


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = args.output_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    logger = make_logger(run / "run.log")
    normalization = args.normalization or args.pretrain_checkpoint.parent / "feature_normalization.npz"
    norm = np.load(normalization)
    feature_names = norm["feature_names"].astype(str).tolist()
    peak_method = (
        str(norm["peak_normalization"].item())
        if "peak_normalization" in norm
        else "log_quantile"
    )
    with h5py.File(args.data, "r") as h5:
        stored = h5["feature_names"][:].astype(str).tolist()
        if stored != feature_names:
            raise RuntimeError("Gene features do not match pretraining feature order")
        splits = split_gene_indices(len(h5["gene_id"]), args.valid_ratio, args.test_ratio, args.seed)
        target_mask, target_scale = target_scales(
            h5["polii_targets"], splits["train"], args.target_peak_quantile
        )
    np.savez(run / "split_indices.npz", **splits)
    np.savez(run / "target_normalization.npz", peak_mask=target_mask, peak_scales=target_scale)
    with (run / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, default=str, indent=2)

    def dataset(indices):
        return CRERegGeneDataset(
            args.data, indices, norm["peak_mask"], norm["peak_scales"], peak_method,
            target_mask, target_scale,
        )

    loaders = {
        name: DataLoader(
            dataset(indices), batch_size=args.batch_size, shuffle=name == "train",
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_genes, drop_last=name == "train",
        )
        for name, indices in splits.items()
    }
    encoder, model_name = load_pretrained(args.pretrain_checkpoint, feature_names)
    model = CRERegGeneModel(encoder, depth=args.depth)
    if args.freeze_encoder:
        for parameter in model.region_encoder.parameters():
            parameter.requires_grad = False
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    best = float("inf")
    history = []
    logger.info("model=%s train=%d validation=%d test=%d", model_name, *(len(splits[x]) for x in ("train", "validation", "test")))
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, loaders["train"], optimizer, device, args.nonzero_weight)
        metrics, _, _, _ = evaluate(model, loaders["validation"], device, args.nonzero_weight)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"validation_{k}": v for k, v in metrics.items()}}
        history.append(row)
        logger.info("epoch=%d train=%.6f validation=%.6f rmse=%.6f mae=%.6f pearson=%.6f", epoch, train_loss, metrics["loss"], metrics["rmse"], metrics["mae"], metrics["pearson"])
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)}
        torch.save(checkpoint, run / "checkpoint-last.pth")
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save(checkpoint, run / "checkpoint-best.pth")
    with (run / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    best_state = torch.load(run / "checkpoint-best.pth", map_location=device)
    model.load_state_dict(best_state["model"])
    metrics, prediction, target, indices = evaluate(model, loaders["test"], device, args.nonzero_weight)
    with (run / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    np.savez_compressed(run / "test_predictions.npz", gene_index=indices, prediction=prediction, target=target)


if __name__ == "__main__":
    main()
