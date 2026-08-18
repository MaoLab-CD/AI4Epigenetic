# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Adapted from scMETH/engine_pretrain.py and the MAE implementation.

import datetime
import math
import sys
import time
from typing import Iterable

import torch
import torch.distributed as dist


def adjust_learning_rate(optimizer, progress, args):
    if progress < args.warmup_epochs:
        lr = args.lr * progress / max(args.warmup_epochs, 1.0e-8)
    else:
        progress = (progress - args.warmup_epochs) / max(
            args.epochs - args.warmup_epochs, 1.0e-8
        )
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def all_reduce_sums(values):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values)
    return values


def masked_mae(values, pred, mask):
    error = (pred.float() - values.float()).abs()
    return (error * mask).sum() / mask.sum()


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    logger,
    log_writer=None,
    args=None,
):
    model.train(True)
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    loss_sum = 0.0
    mae_sum = 0.0
    sample_count = 0
    start_time = time.time()
    end = time.time()
    data_time_sum = 0.0
    amp_dtype = (
        torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    )

    if log_writer is not None:
        logger.info("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(data_loader):
        data_time_sum += time.time() - end

        if data_iter_step % accum_iter == 0:
            lr = adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args,
            )

        samples = samples.to(device, non_blocking=True)

        if not torch.isfinite(samples).all():
            logger.warning(
                f"non-finite input at epoch={epoch}, iter={data_iter_step}"
            )
            continue

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            _, loss_mask, pred, mask = model(
                samples, mask_ratio=args.mask_ratio
            )
            mae_mask = masked_mae(samples, pred, mask)

        loss_value = loss_mask.item()
        mae_value = mae_mask.item()

        if not math.isfinite(loss_value):
            logger.info(
                f"Loss is {loss_value}, stopping training at "
                f"epoch={epoch}, iter={data_iter_step}"
            )
            sys.exit(1)

        loss = loss_mask / accum_iter
        loss_scaler.scale(loss).backward()
        if (data_iter_step + 1) % accum_iter == 0:
            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_grad
            )
            loss_scaler.step(optimizer)
            loss_scaler.update()
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        batch_size = samples.shape[0]
        loss_sum += loss_value * batch_size
        mae_sum += mae_value * batch_size
        sample_count += batch_size
        lr = optimizer.param_groups[0]["lr"]

        loss_value_reduce = all_reduce_sums(
            torch.tensor(loss_value, device=device)
        ).item()
        mae_value_reduce = all_reduce_sums(
            torch.tensor(mae_value, device=device)
        ).item()
        if dist.is_available() and dist.is_initialized():
            loss_value_reduce /= dist.get_world_size()
            mae_value_reduce /= dist.get_world_size()

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000
            )
            log_writer.add_scalar(
                "train_loss", loss_value_reduce, epoch_1000x
            )
            log_writer.add_scalar(
                "train_mae", mae_value_reduce, epoch_1000x
            )
            log_writer.add_scalar("lr", lr, epoch_1000x)

        if (
            data_iter_step % args.print_freq == 0
            or data_iter_step + 1 == len(data_loader)
        ):
            elapsed = time.time() - start_time
            time_per_iter = elapsed / (data_iter_step + 1)
            eta = datetime.timedelta(
                seconds=int(time_per_iter * (len(data_loader) - data_iter_step - 1))
            )
            max_memory = (
                torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
                if device.type == "cuda"
                else 0.0
            )
            logger.info(
                f"{header}  [{data_iter_step}/{len(data_loader)}]  "
                f"eta: {eta}  lr: {lr:.3e}  "
                f"loss: {loss_sum / sample_count:.6f}  "
                f"mae: {mae_sum / sample_count:.6f}  "
                f"time: {time_per_iter:.4f}  "
                f"data: {data_time_sum / (data_iter_step + 1):.4f}  "
                f"max mem: {max_memory:.0f}"
            )
        end = time.time()

    totals = all_reduce_sums(
        torch.tensor(
            [loss_sum, mae_sum, sample_count],
            device=device,
            dtype=torch.float64,
        )
    )
    stats = {
        "loss": float(totals[0] / totals[2]),
        "mae": float(totals[1] / totals[2]),
        "lr": optimizer.param_groups[0]["lr"],
    }
    logger.info("Averaged stats: {}".format(stats))
    return stats


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    log_writer=None,
    args=None,
):
    model.eval()
    loss_sum = 0.0
    mae_sum = 0.0
    sample_count = 0
    amp_dtype = (
        torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    )

    for samples in data_loader:
        samples = samples.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            _, loss_mask, pred, mask = model(
                samples, mask_ratio=args.mask_ratio
            )
            mae_mask = masked_mae(samples, pred, mask)

        batch_size = samples.shape[0]
        loss_sum += loss_mask.item() * batch_size
        mae_sum += mae_mask.item() * batch_size
        sample_count += batch_size

        if device.type == "cuda":
            torch.cuda.synchronize()

    totals = all_reduce_sums(
        torch.tensor(
            [loss_sum, mae_sum, sample_count],
            device=device,
            dtype=torch.float64,
        )
    )
    stats = {
        "loss": float(totals[0] / totals[2]),
        "mae": float(totals[1] / totals[2]),
    }

    if log_writer is not None:
        log_writer.add_scalar("val_loss", stats["loss"], epoch)
        log_writer.add_scalar("val_mae", stats["mae"], epoch)

    return stats
