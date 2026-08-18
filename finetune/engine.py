# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
import math
import sys
from typing import Iterable

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


def regression_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    mse = float(np.mean((predictions - labels) ** 2))
    mae = float(np.mean(np.abs(predictions - labels)))
    pearson = (
        float(pearsonr(labels, predictions).statistic)
        if np.std(labels) > 0 and np.std(predictions) > 0
        else 0.0
    )
    spearman = (
        float(spearmanr(labels, predictions).statistic)
        if np.std(labels) > 0 and np.std(predictions) > 0
        else 0.0
    )
    return {
        "loss": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "r2": float(r2_score(labels, predictions)),
    }


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    logger,
    max_norm: float = 0,
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

    if log_writer is not None:
        logger.info("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batch_data in enumerate(data_loader):
        values = batch_data["values"].to(device, non_blocking=True)
        targets = batch_data["target"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(values)
            loss_reg = criterion(outputs, targets)

        loss_value = loss_reg.item()
        mae_value = torch.abs(outputs - targets).mean().item()
        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss_reg / accum_iter
        loss_scaler.scale(loss).backward()
        if (data_iter_step + 1) % accum_iter == 0:
            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            loss_scaler.step(optimizer)
            loss_scaler.update()
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        batch_size = values.shape[0]
        loss_sum += loss_value * batch_size
        mae_sum += mae_value * batch_size
        sample_count += batch_size
        min_lr = min(group["lr"] for group in optimizer.param_groups)
        max_lr = max(group["lr"] for group in optimizer.param_groups)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000
            )
            log_writer.add_scalar("loss", loss_value, epoch_1000x)
            log_writer.add_scalar("mae", mae_value, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)
            log_writer.add_scalar("min_lr", min_lr, epoch_1000x)

        if (
            data_iter_step % args.print_freq == 0
            or data_iter_step + 1 == len(data_loader)
        ):
            logger.info(
                f"{header}  [{data_iter_step}/{len(data_loader)}]  "
                f"lr: {max_lr:.6e}  min_lr: {min_lr:.6e}  "
                f"loss: {loss_sum / sample_count:.6f}  "
                f"mae: {mae_sum / sample_count:.6f}"
            )

    stats = {
        "loss": loss_sum / sample_count,
        "mae": mae_sum / sample_count,
        "lr": max(group["lr"] for group in optimizer.param_groups),
        "min_lr": min(group["lr"] for group in optimizer.param_groups),
    }
    logger.info("Averaged stats: {}".format(stats))
    return stats


@torch.no_grad()
def evaluate(
    data_loader,
    model,
    device,
    criterion,
    log_writer=None,
    epoch=None,
    return_raw=False,
    args=None,
):
    model.eval()
    predictions, labels, indices = [], [], []

    for batch_data in data_loader:
        values = batch_data["values"].to(device, non_blocking=True)
        targets = batch_data["target"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(values)
            criterion(outputs, targets)

        predictions.append(outputs.float().cpu().numpy())
        labels.append(targets.float().cpu().numpy())
        indices.append(batch_data["index"].numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    indices = np.concatenate(indices)
    stats = regression_metrics(labels, predictions)

    if log_writer is not None and epoch is not None:
        for name, value in stats.items():
            log_writer.add_scalar(f"val_{name}", value, epoch)

    if return_raw:
        return predictions, labels, indices, stats
    return stats
