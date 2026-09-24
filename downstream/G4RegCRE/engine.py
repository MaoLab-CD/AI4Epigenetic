from __future__ import annotations

import numpy as np
import torch

from .model import reconstruction_loss


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward_batch(model, batch, g4_observed, cre_observed):
    return model(
        batch["g4"], batch["cre"], batch["relation"], batch["distance"],
        batch["cis"], batch["hic"], batch["perturbation"],
        g4_observed, cre_observed,
    )


def train_epoch(model, loader, optimizer, device, g4_observed, cre_observed, nonzero_weight):
    model.train()
    total = count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_batch(model, batch, g4_observed, cre_observed)
        loss = reconstruction_loss(prediction, batch["target"], nonzero_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(prediction)
        count += len(prediction)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device, g4_observed, cre_observed, nonzero_weight):
    model.eval()
    predictions, targets, indices = [], [], []
    total = count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        prediction = forward_batch(model, batch, g4_observed, cre_observed)
        loss = reconstruction_loss(prediction, batch["target"], nonzero_weight)
        total += loss.item() * len(prediction)
        count += len(prediction)
        predictions.append(prediction.cpu().numpy())
        targets.append(batch["target"].cpu().numpy())
        indices.append(batch["cre_index"].cpu().numpy())
    pred = np.concatenate(predictions) if predictions else np.empty((0, 0))
    true = np.concatenate(targets) if targets else np.empty((0, 0))
    mse = float(np.mean((pred - true) ** 2)) if pred.size else float("nan")
    mae = float(np.mean(np.abs(pred - true))) if pred.size else float("nan")
    nonzero = true > 0
    nonzero_mae = float(np.mean(np.abs(pred[nonzero] - true[nonzero]))) if nonzero.any() else float("nan")
    return {
        "loss": total / max(count, 1), "mse": mse,
        "rmse": float(np.sqrt(mse)), "mae": mae, "nonzero_mae": nonzero_mae,
    }, pred, true, np.concatenate(indices) if indices else np.empty(0, dtype=int)
