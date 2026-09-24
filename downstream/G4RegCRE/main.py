from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from pretrain import model as pretrain_models
from .dataset import G4ToCREDataset, split_cre_indices, target_peak_metadata
from .engine import evaluate, train_epoch
from .model import G4ToCREModel


def parse_args():
    parser = argparse.ArgumentParser("G4-to-CRE perturbation prediction")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("downstream/G4RegCRE/output"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-g4-per-cre", type=int, default=256,
                        help="0 uses every G4; positive values sample implicit proximal pairs while retaining observed relations first")
    parser.add_argument("--target-peak-quantile", type=float, default=0.995)
    parser.add_argument("--nonzero-weight", type=float, default=2.0)
    parser.add_argument("--fusion-depth", type=int, default=2)
    parser.add_argument("--freeze-encoders", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def logger_for(path):
    logger = logging.getLogger("g4_to_cre")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_pretrained(path: Path, feature_names: list[str]):
    checkpoint = torch.load(path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    model_name = saved_args.get("model", "mae_bin_large") if isinstance(saved_args, dict) else getattr(saved_args, "model", "mae_bin_large")
    model = pretrain_models.__dict__[model_name](feature_names=feature_names)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {"input_norm.weight", "input_norm.bias", "decoder_input_norm.weight", "decoder_input_norm.bias"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Incompatible checkpoint; missing={missing}, unexpected={unexpected}")
    return model, model_name


def make_dataset(args, indices, input_meta, target_meta, seed):
    return G4ToCREDataset(
        args.data, indices, input_meta[0], input_meta[1], input_meta[2],
        target_meta[0], target_meta[1], args.max_g4_per_cre, seed,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = args.output_dir.resolve() / timestamp
    run.mkdir(parents=True)
    logger = logger_for(run / "run.log")
    args.data, args.pretrain_checkpoint = args.data.resolve(), args.pretrain_checkpoint.resolve()
    args.normalization = (args.normalization or args.pretrain_checkpoint.parent / "feature_normalization.npz").resolve()
    with (run / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, default=str, indent=2)

    norm = np.load(args.normalization)
    feature_names = norm["feature_names"].astype(str).tolist()
    peak_method = str(norm["peak_normalization"].item()) if "peak_normalization" in norm else "log_quantile"
    with h5py.File(args.data, "r") as h5:
        stored = h5["pretrain_feature_names"][:].astype(str).tolist()
        if stored != feature_names:
            raise RuntimeError("G4-to-CRE features do not match the pretraining feature order")
        cre_count = len(h5["cre/id"])
        target_names = h5["target_feature_names"][:].astype(str).tolist()
        g4_observed = torch.from_numpy(h5["g4/observed_features"][:].astype(bool))
        cre_observed = torch.from_numpy(h5["cre/observed_features"][:].astype(bool))
        splits = split_cre_indices(cre_count, args.valid_ratio, args.test_ratio, args.seed)
        target_peak_mask, target_peak_scales = target_peak_metadata(
            target_names, h5["cre/perturbed_targets"], splits["train"], args.target_peak_quantile
        )
    np.savez(run / "split_indices.npz", **splits)
    np.savez(run / "target_normalization.npz", peak_mask=target_peak_mask,
             peak_scales=target_peak_scales, peak_normalization=peak_method,
             feature_names=np.asarray(target_names))

    input_meta = (norm["peak_mask"].astype(bool), norm["peak_scales"].astype(np.float32), peak_method)
    target_meta = (target_peak_mask, target_peak_scales)
    datasets = {name: make_dataset(args, rows, input_meta, target_meta, args.seed) for name, rows in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=name == "train", num_workers=args.num_workers,
                         pin_memory=True, drop_last=name == "train")
        for name, dataset in datasets.items()
    }

    encoder, model_name = load_pretrained(args.pretrain_checkpoint, feature_names)
    model = G4ToCREModel(encoder, len(target_names), fusion_depth=args.fusion_depth)
    if args.freeze_encoders:
        for parameter in list(model.g4_encoder.parameters()) + list(model.cre_encoder.parameters()):
            parameter.requires_grad = False
    device = torch.device(args.device)
    model.to(device)
    g4_observed, cre_observed = g4_observed.to(device), cre_observed.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    logger.info("model=%s train=%d validation=%d test=%d", model_name,
                len(splits["train"]), len(splits["validation"]), len(splits["test"]))
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, loaders["train"], optimizer, device,
                                 g4_observed, cre_observed, args.nonzero_weight)
        metrics, _, _, _ = evaluate(model, loaders["validation"], device,
                                    g4_observed, cre_observed, args.nonzero_weight)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"validation_{k}": v for k, v in metrics.items()}}
        history.append(row)
        logger.info("epoch=%d train_loss=%.6f validation_loss=%.6f rmse=%.6f mae=%.6f nonzero_mae=%.6f",
                    epoch, train_loss, metrics["loss"], metrics["rmse"], metrics["mae"], metrics["nonzero_mae"])
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "args": vars(args), "feature_names": feature_names,
                 "target_names": target_names}
        torch.save(state, run / "checkpoint-last.pth")
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save(state, run / "checkpoint-best.pth")

    with (run / "history.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0], delimiter="\t")
        writer.writeheader(); writer.writerows(history)
    best_state = torch.load(run / "checkpoint-best.pth", map_location=device)
    model.load_state_dict(best_state["model"])
    metrics, prediction, target, indices = evaluate(
        model, loaders["test"], device, g4_observed, cre_observed, args.nonzero_weight
    )
    with (run / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    np.savez_compressed(run / "test_predictions.npz", cre_index=indices,
                        prediction=prediction, target=target,
                        target_names=np.asarray(target_names))
    logger.info("test %s", json.dumps(metrics))


if __name__ == "__main__":
    main()
