from __future__ import annotations

import numpy as np
import torch

from .model import regression_loss


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_epoch(model, loader, optimizer, device, nonzero_weight):
    model.train()
    total = count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["features"], batch["role"], batch["valid"])
        loss = regression_loss(prediction, batch["target"], nonzero_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(prediction)
        count += len(prediction)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device, nonzero_weight):
    model.eval()
    predictions, targets, indices = [], [], []
    total = count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        prediction = model(batch["features"], batch["role"], batch["valid"])
        loss = regression_loss(prediction, batch["target"], nonzero_weight)
        total += loss.item() * len(prediction)
        count += len(prediction)
        predictions.append(prediction.cpu().numpy())
        targets.append(batch["target"].cpu().numpy())
        indices.append(batch["gene_index"].cpu().numpy())
    pred = np.concatenate(predictions) if predictions else np.empty((0, 4))
    true = np.concatenate(targets) if targets else np.empty((0, 4))
    mse = float(np.mean((pred - true) ** 2)) if pred.size else float("nan")
    mae = float(np.mean(np.abs(pred - true))) if pred.size else float("nan")
    correlations = []
    for column in range(true.shape[1]):
        if np.std(true[:, column]) > 0 and np.std(pred[:, column]) > 0:
            correlations.append(np.corrcoef(true[:, column], pred[:, column])[0, 1])
    return {
        "loss": total / max(count, 1), "mse": mse,
        "rmse": float(np.sqrt(mse)), "mae": mae,
        "pearson": float(np.mean(correlations)) if correlations else float("nan"),
    }, pred, true, np.concatenate(indices) if indices else np.empty(0, dtype=int)
