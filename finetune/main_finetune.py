from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from finetune.dataset import (
    PolIIPeakDataset,
    read_target,
    select_input_and_target,
    stratified_splits,
    target_scale,
)
from finetune.engine import evaluate, regression_metrics, train_one_epoch
from finetune import model as models_polii
from pretrain.utils import create_logger, load_model, save_model


DEFAULT_DATA = Path(
    "/mnt/afan/G4RegFormer/data/preprocessed/bin_1000bp/pretrain_1000bp/"
    "mm10_1000_multimodal_chrall_feature_matrixs.h5"
)
DEFAULT_FINETUNE = Path(
    "/mnt/afan/G4RegFormer/pretrain/output/"
    "20260730-234159-mae_bin_large/checkpoint-19.pth"
)


class WeightedMSELoss(nn.Module):
    def __init__(self, positive_weight=1.0):
        super().__init__()
        self.positive_weight = positive_weight

    def forward(self, prediction, target):
        weights = torch.where(
            target > 0,
            torch.full_like(target, self.positive_weight),
            torch.ones_like(target),
        )
        return ((prediction - target).pow(2) * weights).mean()


def get_args_parser():
    parser = argparse.ArgumentParser(
        "G4RegFormer fine-tuning for Pol II S5P peak regression",
        add_help=False,
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--eval_batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)

    parser.add_argument("--data_path", default=DEFAULT_DATA, type=Path)
    parser.add_argument("--columns_path", default=None, type=Path)
    parser.add_argument(
        "--output_dir", default=Path("finetune/output"), type=Path
    )
    parser.add_argument("--finetune", default=DEFAULT_FINETUNE, type=Path)
    parser.add_argument("--normalization_path", default=None, type=Path)
    parser.add_argument("--scratch", action="store_true")

    parser.add_argument(
        "--model", default="polii_peak_large", type=str, metavar="MODEL"
    )
    parser.add_argument("--target_name", default="PolIIS5P_peak")
    parser.add_argument("--target_quantile", default=0.995, type=float)
    parser.add_argument("--valid_ratio", default=0.1, type=float)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--positive_weight", default=1.0, type=float)

    parser.add_argument("--clip_grad", default=1.0, type=float)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--blr", default=1.0e-3, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--encoder_lr_scale", default=0.1, type=float)
    parser.add_argument("--schedule_interval", default=1, type=int)
    parser.add_argument("--early_stop", default=10, type=int)
    parser.add_argument(
        "--save_metric",
        default="loss",
        choices=["loss", "rmse", "mae", "pearson", "spearman", "r2"],
    )

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default=None, type=Path)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--max_train_samples", default=None, type=int)
    parser.add_argument("--max_validation_samples", default=None, type=int)
    parser.add_argument("--max_test_samples", default=None, type=int)
    return parser


def infer_columns_path(data_path):
    stem = data_path.name.removesuffix("_feature_matrixs.h5")
    path = data_path.parent / f"{stem}_feature_columns.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def limit_rows(rows, limit, seed):
    if limit is None or limit >= len(rows):
        return rows
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(rows, size=limit, replace=False))


def load_pretrained(model, checkpoint_path, logger):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    compatible = {
        key: value
        for key, value in checkpoint_model.items()
        if key in state_dict and value.shape == state_dict[key].shape
    }
    message = model.load_state_dict(compatible, strict=False)
    logger.info("Load pre-trained checkpoint from: %s", checkpoint_path)
    logger.info(message)
    logger.info("Missing keys: %s", message.missing_keys)
    logger.info(
        "Ignored pre-training-only keys: %d",
        len(checkpoint_model) - len(compatible),
    )


def write_split_summary(path, splits, target):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["split", "rows", "positive", "positive_ratio"])
        for name, rows in splits.items():
            positive = int(np.count_nonzero(target[rows] > 0))
            writer.writerow(
                [name, len(rows), positive, f"{positive / len(rows):.6f}"]
            )


def save_scatter(path, labels, predictions, seed):
    generator = np.random.default_rng(seed)
    count = min(50000, len(labels))
    selected = generator.choice(len(labels), count, replace=False)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.hexbin(
        labels[selected],
        predictions[selected],
        gridsize=70,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    axis.plot([0, 1], [0, 1], color="red", linewidth=1)
    axis.set_xlabel("Observed Pol II S5P peak")
    axis.set_ylabel("Predicted Pol II S5P peak")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    initialization = "scratch" if args.scratch else "pretrained"
    args.log_dir = (
        args.output_dir
        / "polii_peak"
        / f"{start_time}-{args.model}-{initialization}"
    )
    args.log_dir.mkdir(parents=True, exist_ok=False)
    logger = create_logger(args.log_dir / "run.log", rank=0)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    args.data_path = args.data_path.resolve()
    args.columns_path = (
        args.columns_path.resolve()
        if args.columns_path
        else infer_columns_path(args.data_path)
    )
    args.normalization_path = (
        args.normalization_path.resolve()
        if args.normalization_path
        else args.finetune.resolve().parent / "feature_normalization.npz"
    )

    logger.info("Running on %s", start_time)
    logger.info("job dir: %s", PROJECT_ROOT)
    logger.info("saving to %s", args.log_dir)

    input_indices, input_names, target_index = select_input_and_target(
        args.columns_path, args.target_name
    )
    normalization = np.load(args.normalization_path)
    normalized_names = normalization["feature_names"].astype(str).tolist()
    if input_names != normalized_names:
        raise RuntimeError(
            "Fine-tuning input features do not match pre-training features"
        )
    peak_mask = normalization["peak_mask"]
    peak_scales = normalization["peak_scales"]

    with h5py.File(args.data_path, "r") as handle:
        row_count = int(handle["matrix"].shape[0])
    target = read_target(args.data_path, target_index)
    splits = stratified_splits(
        target, args.valid_ratio, args.test_ratio, args.seed
    )
    split_limits = {
        "train": args.max_train_samples,
        "validation": args.max_validation_samples,
        "test": args.max_test_samples,
    }
    for offset, (name, limit) in enumerate(split_limits.items()):
        splits[name] = limit_rows(splits[name], limit, args.seed + offset)

    peak_scale = target_scale(
        target, splits["train"], args.target_quantile
    )
    np.savez(
        args.log_dir / "split_indices.npz",
        **splits,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )
    np.savez(
        args.log_dir / "target_normalization.npz",
        target_name=args.target_name,
        source_index=target_index,
        quantile=args.target_quantile,
        peak_scale=peak_scale,
    )
    write_split_summary(args.log_dir / "split_summary.tsv", splits, target)

    logger.info(
        "Data matrix: %s bins, %d input features, target=%s (column %d)",
        f"{row_count:,}",
        len(input_names),
        args.target_name,
        target_index,
    )
    logger.info(
        "Split: train=%s, validation=%s, test=%s",
        f"{len(splits['train']):,}",
        f"{len(splits['validation']):,}",
        f"{len(splits['test']):,}",
    )
    logger.info(
        "Target normalization: log1p / log1p(Q%.3f=%.6f), clipped to [0, 1]",
        args.target_quantile,
        peak_scale,
    )

    datasets = {
        name: PolIIPeakDataset(
            args.data_path,
            rows,
            input_indices,
            peak_mask,
            peak_scales,
            target_index,
            peak_scale,
        )
        for name, rows in splits.items()
    }
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_mem,
        "persistent_workers": args.num_workers > 0,
    }
    data_loader_train = DataLoader(
        datasets["train"],
        sampler=RandomSampler(datasets["train"]),
        batch_size=args.batch_size,
        drop_last=False,
        **loader_options,
    )
    data_loader_val = DataLoader(
        datasets["validation"],
        sampler=SequentialSampler(datasets["validation"]),
        batch_size=args.eval_batch_size,
        drop_last=False,
        **loader_options,
    )
    data_loader_test = DataLoader(
        datasets["test"],
        sampler=SequentialSampler(datasets["test"]),
        batch_size=args.eval_batch_size,
        drop_last=False,
        **loader_options,
    )

    device = torch.device(args.device)
    model = models_polii.__dict__[args.model](
        feature_count=len(input_names)
    )
    if not args.scratch:
        load_pretrained(model, args.finetune, logger)
    model.to(device)
    model_without_ddp = model

    head_params, backbone_params = [], []
    for name, parameter in model.named_parameters():
        if name.startswith("regression_decoder"):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    effective_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * effective_batch_size / 256
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": args.lr * args.encoder_lr_scale,
            },
            {"params": head_params, "lr": args.lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = WeightedMSELoss(args.positive_weight)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.schedule_interval, gamma=0.9
    )

    if args.resume:
        args.start_epoch = load_model(
            args.resume, model, optimizer, loss_scaler
        )
        logger.info("Resume fine-tuning from %s", args.resume)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info("Model = %s", model_without_ddp)
    logger.info("number of params (M): %.2f", parameter_count / 1.0e6)
    logger.info("{}".format(args).replace(", ", ",\n"))
    logger.info("base lr: %.2e", args.lr * 256 / effective_batch_size)
    logger.info("actual lr: %.2e", args.lr)
    logger.info("encoder lr: %.2e", args.lr * args.encoder_lr_scale)
    logger.info("effective batch size: %d", effective_batch_size)
    with (args.log_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    if args.eval:
        _, _, _, test_stats = evaluate(
            data_loader_test,
            model,
            device,
            criterion,
            return_raw=True,
            args=args,
        )
        logger.info(test_stats)
        return

    logger.info("Start training for %d epochs", args.epochs)
    training_start = time.time()
    lower_is_better = args.save_metric in {"loss", "rmse", "mae"}
    best_score = float("inf") if lower_is_better else -float("inf")
    best_epoch = -1
    best_state = None
    patience = 0

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            logger,
            args.clip_grad,
            log_writer=log_writer,
            args=args,
        )
        logger.info(train_stats)
        log_writer.flush()

        val_stats = evaluate(
            data_loader_val,
            model,
            device,
            criterion,
            log_writer=log_writer,
            epoch=epoch,
            args=args,
        )
        logger.info(
            "| end of epoch %3d | time: %5.2fs | valid loss %.4f | "
            "RMSE %.4f | MAE %.4f | R2 %.4f | Pearson %.4f | Spearman %.4f",
            epoch,
            time.time() - epoch_start,
            val_stats["loss"],
            val_stats["rmse"],
            val_stats["mae"],
            val_stats["r2"],
            val_stats["pearson"],
            val_stats["spearman"],
        )

        score = val_stats[args.save_metric]
        improved = score < best_score if lower_is_better else score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model_without_ddp.state_dict())
            patience = 0
            checkpoint = save_model(
                args.log_dir,
                epoch,
                model,
                optimizer,
                loss_scaler,
                args,
            )
            torch.save(best_state, args.log_dir / "best_model.pth")
            logger.info(
                "Best model with val %s %.6f, saved to %s",
                args.save_metric,
                score,
                checkpoint,
            )
        else:
            patience += 1
            if patience >= args.early_stop:
                logger.info("Early stop at epoch %d", epoch)
                break
        scheduler.step()

    if best_state is None:
        raise RuntimeError("No valid model was produced")
    model_without_ddp.load_state_dict(best_state)
    predictions, labels, indices, test_stats = evaluate(
        data_loader_test,
        model,
        device,
        criterion,
        return_raw=True,
        args=args,
    )
    raw_predictions = np.expm1(
        predictions * np.log1p(peak_scale)
    )
    raw_labels = target[indices]
    raw_stats = {
        f"raw_{name}": value
        for name, value in regression_metrics(
            raw_labels, raw_predictions
        ).items()
    }
    final_stats = {**test_stats, **raw_stats}
    np.savez(
        args.log_dir / "test_predictions.npz",
        matrix_index=indices,
        prediction=predictions,
        target=labels,
        raw_prediction=raw_predictions,
        raw_target=raw_labels,
    )
    with (args.log_dir / "test_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"best_epoch": best_epoch, **final_stats},
            handle,
            indent=2,
        )
    save_scatter(
        args.log_dir / "test_prediction_scatter.png",
        labels,
        predictions,
        args.seed,
    )
    logger.info("Test metrics: %s", final_stats)
    logger.info("Best epoch: %d", best_epoch)
    logger.info(
        "Training time %s",
        datetime.timedelta(seconds=int(time.time() - training_start)),
    )
    log_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "G4RegFormer Pol II S5P peak fine-tuning",
        parents=[get_args_parser()],
    )
    main(parser.parse_args())
