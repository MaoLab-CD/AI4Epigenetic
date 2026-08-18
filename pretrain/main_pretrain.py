from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

from dataset import (
    BlockShuffleSampler,
    H5BinDataset,
    compute_peak_scales,
    random_splits,
    read_feature_selection,
)
from engine import evaluate, train_one_epoch
import model as models_g4regformer
from utils import create_logger, load_model, save_model, unwrap_model


DEFAULT_DATA = Path(
    "/mnt/afan/G4RegFormer/data/preprocessed/bin_1000bp/pretrain_1000bp/"
    "mm10_1000_multimodal_chrall_feature_matrixs.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("G4RegFormer masked-reconstruction pretraining")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--columns", type=Path, default=None)
    parser.add_argument(
        "--output_dir", "--output-dir", dest="output_dir",
        type=Path, default=Path("pretrain/output")
    )
    parser.add_argument("--model", default="mae_bin_large", type=str)
    parser.add_argument(
        "--mask_ratio", "--mask-ratio", dest="mask_ratio",
        type=float, default=0.4
    )
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument(
        "--exclude_feature", "--exclude-feature", dest="exclude_feature",
        action="append", default=["PolIIS5P"]
    )
    parser.add_argument(
        "--valid_ratio", "--valid-ratio", dest="valid_ratio",
        type=float, default=0.03
    )
    parser.add_argument(
        "--peak_quantile", "--peak-quantile", dest="peak_quantile",
        type=float, default=0.995
    )
    parser.add_argument(
        "--batch_size", "--batch-size", dest="batch_size",
        type=int, default=64
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--accum_iter", "--accum-iter", dest="accum_iter",
        type=int, default=1
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", "--base-lr", dest="blr", type=float, default=1e-4)
    parser.add_argument(
        "--min_lr", "--min-lr", dest="min_lr", type=float, default=1e-6
    )
    parser.add_argument(
        "--warmup_epochs", "--warmup-epochs", dest="warmup_epochs",
        type=float, default=2.0
    )
    parser.add_argument(
        "--weight_decay", "--weight-decay", dest="weight_decay",
        type=float, default=0.05
    )
    parser.add_argument(
        "--clip_grad", "--clip-grad", dest="clip_grad",
        type=float, default=1.0
    )
    parser.add_argument(
        "--amp_dtype", "--amp-dtype", dest="amp_dtype",
        choices=["bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument(
        "--num_workers", "--num-workers", dest="num_workers",
        type=int, default=4
    )
    parser.add_argument(
        "--h5_block_rows", "--h5-block-rows", dest="h5_block_rows",
        type=int, default=2048
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--start_epoch", "--start-epoch", dest="start_epoch",
        type=int, default=0
    )
    parser.add_argument(
        "--save_freq", "--save-freq", dest="save_freq",
        type=int, default=1
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--print_freq", "--print-freq", dest="print_freq",
        type=int, default=100
    )
    parser.add_argument(
        "--max_train_samples", "--max-train-samples",
        dest="max_train_samples", type=int, default=None
    )
    parser.add_argument(
        "--max_validation_samples", "--max-validation-samples",
        dest="max_validation_samples", type=int, default=None
    )
    return parser.parse_args()


def infer_companion_path(data: Path, suffix: str) -> Path:
    stem = data.name.removesuffix("_feature_matrixs.h5")
    candidate = data.parent / f"{stem}_{suffix}"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def initialize_distributed(device_name: str) -> tuple[int, int, int, torch.device]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device(device_name)
    return rank, world_size, local_rank, device


def write_feature_table(path: Path, selection) -> None:
    selected = {
        int(index): (model_index, name)
        for model_index, (index, name) in enumerate(
            zip(selection.source_indices, selection.names)
        )
    }
    excluded = {
        int(index): name
        for index, name in zip(
            selection.excluded_source_indices,
            selection.excluded_names,
        )
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["model_index", "source_index", "name", "status"])
        for source_index in sorted(selected.keys() | excluded.keys()):
            if source_index in selected:
                model_index, name = selected[source_index]
                writer.writerow([model_index, source_index, name, "selected"])
            else:
                writer.writerow(["-", source_index, excluded[source_index], "excluded"])


def main(args: argparse.Namespace) -> None:
    if not 0.0 < args.mask_ratio < 1.0:
        raise ValueError("--mask-ratio must be between 0 and 1")
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be between 0 and 1")
    if not 0.0 < args.peak_quantile <= 1.0:
        raise ValueError("--peak-quantile must be in (0, 1]")
    if args.save_freq <= 0:
        raise ValueError("--save-freq must be positive")
    rank, world_size, local_rank, device = initialize_distributed(args.device)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.data = args.data.resolve()
    args.columns = (
        args.columns.resolve()
        if args.columns
        else infer_companion_path(args.data, "feature_columns.tsv")
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") if rank == 0 else ""
    if world_size > 1:
        shared = [timestamp]
        dist.broadcast_object_list(shared, src=0)
        timestamp = shared[0]
    run_dir = args.output_dir.resolve() / f"{timestamp}-{args.model}"
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()
    args.log_dir = str(run_dir)
    args.rank = rank
    args.world_size = world_size
    args.distributed = world_size > 1
    logger = create_logger(run_dir / "run.log", rank)
    logger.info(f"Running on {timestamp}")
    logger.info(f"job dir: {Path(__file__).resolve().parent.parent}")
    logger.info("{}".format(args).replace(", ", ",\n"))

    selection = read_feature_selection(
        args.columns,
        tuple(dict.fromkeys(args.exclude_feature)),
    )
    if len(selection.names) != 288 or len(selection.excluded_names) != 35:
        raise RuntimeError(
            "Expected 288 selected and 35 PolIIS5P-related excluded features, "
            f"found {len(selection.names)} selected and "
            f"{len(selection.excluded_names)} excluded"
        )
    with h5py.File(args.data, "r") as handle:
        matrix_rows = int(handle["matrix"].shape[0])
    splits = random_splits(matrix_rows, args.valid_ratio, args.seed)
    limits = {
        "train": args.max_train_samples,
        "validation": args.max_validation_samples,
    }
    for name, limit in limits.items():
        if limit is not None:
            splits[name] = splits[name][:limit]

    normalization_path = run_dir / "feature_normalization.npz"
    if rank == 0:
        peak_mask, peak_scales = compute_peak_scales(
            args.data,
            splits["train"],
            selection.source_indices,
            selection.names,
            args.peak_quantile,
        )
        np.savez(
            normalization_path,
            peak_mask=peak_mask,
            peak_scales=peak_scales,
            peak_quantile=args.peak_quantile,
            source_indices=selection.source_indices,
            feature_names=np.asarray(selection.names),
        )
        np.savez(
            run_dir / "split_indices.npz",
            train=splits["train"],
            validation=splits["validation"],
            seed=args.seed,
            valid_ratio=args.valid_ratio,
        )
        write_feature_table(run_dir / "features.tsv", selection)
        with (run_dir / "split_summary.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["split", "rows", "fraction", "method"])
            writer.writerow(
                [
                    "train",
                    len(splits["train"]),
                    f"{len(splits['train']) / matrix_rows:.6f}",
                    f"random_seed_{args.seed}",
                ]
            )
            writer.writerow(
                [
                    "validation",
                    len(splits["validation"]),
                    f"{len(splits['validation']) / matrix_rows:.6f}",
                    f"random_seed_{args.seed}",
                ]
            )
        logger.info(
            f"Data matrix: {matrix_rows:,} bins, "
            f"{len(selection.names)} selected features, "
            f"{len(selection.excluded_names)} excluded features"
        )
        logger.info(
            f"Random split: train={len(splits['train']):,}, "
            f"validation={len(splits['validation']):,}, "
            f"valid_ratio={args.valid_ratio}, seed={args.seed}"
        )
        logger.info(
            f"Peak normalization: log1p + non-zero "
            f"quantile={args.peak_quantile}, clipped to [0, 1]; "
            f"ratio features unchanged"
        )
    if world_size > 1:
        dist.barrier()
    normalization = np.load(normalization_path)
    peak_mask = normalization["peak_mask"]
    peak_scales = normalization["peak_scales"]

    datasets = {
        name: H5BinDataset(
            args.data,
            indices,
            selection.source_indices,
            peak_mask,
            peak_scales,
        )
        for name, indices in splits.items()
    }
    train_sampler = BlockShuffleSampler(
        len(datasets["train"]),
        block_size=args.h5_block_rows,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    evaluation_samplers = {
        name: (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
            if world_size > 1
            else SequentialSampler(dataset)
        )
        for name, dataset in datasets.items()
        if name != "train"
    }
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        datasets["train"],
        sampler=train_sampler,
        drop_last=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        sampler=evaluation_samplers["validation"],
        drop_last=False,
        **loader_options,
    )
    model = models_g4regformer.__dict__[args.model](
        feature_count=len(selection.names),
        norm_pix_loss=args.norm_pix_loss,
    ).to(device)
    if args.compile:
        model = torch.compile(model)
    raw_model = unwrap_model(model)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    effective_batch = args.batch_size * args.accum_iter * world_size
    args.lr = args.lr or args.blr * effective_batch / 256
    param_groups = optim_factory.add_weight_decay(
        raw_model, args.weight_decay
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and args.amp_dtype == "float16"
    )
    args.start_epoch = max(0, args.start_epoch)
    if args.resume:
        args.start_epoch = load_model(
            args.resume,
            model,
            optimizer,
            scaler,
        )
        logger.info(f"Resume checkpoint {args.resume}")
        logger.info(f"Resume training at epoch {args.start_epoch}")

    if rank == 0:
        parameters = sum(parameter.numel() for parameter in raw_model.parameters())
        logger.info(f"Model = {raw_model}")
        logger.info(f"number of params: {parameters:,}")
        logger.info(f"base lr: {args.lr * 256 / effective_batch:.2e}")
        logger.info(f"actual lr: {args.lr:.2e}")
        logger.info(f"accumulate grad iterations: {args.accum_iter}")
        logger.info(f"effective batch size: {effective_batch}")
        with (run_dir / "args.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, default=str)

    logger.info(f"Start training for {args.epochs} epochs")
    training_started = time.time()
    tensorboard = SummaryWriter(log_dir=run_dir) if rank == 0 else None
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            scaler,
            logger,
            log_writer=tensorboard,
            args=args,
        )
        if rank == 0:
            if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
                checkpoint_path = save_model(
                    run_dir,
                    epoch,
                    model,
                    optimizer,
                    scaler,
                    args,
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            log_stats = {
                **{f"train_{key}": value for key, value in train_metrics.items()},
                "epoch": epoch,
            }
            logger.info(log_stats)
            tensorboard.flush()

        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            epoch,
            log_writer=tensorboard,
            args=args,
        )
        if rank == 0:
            validation_log_stats = {
                **{
                    f"val_{key}": value
                    for key, value in validation_metrics.items()
                },
                "epoch": epoch,
            }
            logger.info(validation_log_stats)
            tensorboard.flush()

    total_time = time.time() - training_started
    if rank == 0:
        tensorboard.close()
        logger.info(
            f"Training time {datetime.timedelta(seconds=int(total_time))}"
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
