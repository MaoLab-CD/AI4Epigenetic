from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.dataset import H5BinDataset, read_feature_selection
from pretrain import model as models_pretrain


DEFAULT_DATA = Path(
    "/mnt/afan/G4RegFormer/data/preprocessed/pretrain_data/bin_1000bp/pretrain_input/"
    "mm10_1000_feature_matrix.h5"
)
DEFAULT_CHECKPOINT = Path(
    "/mnt/afan/G4RegFormer/pretrain/output/"
    "20260730-234159-mae_bin_large/checkpoint-19.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize masked reconstruction of selected genomic bins."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--columns", type=Path, default=None)
    parser.add_argument("--regions", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument(
        "--bin-indices",
        type=str,
        default=None,
        help="Comma-separated HDF5 row indices; random rows are used if omitted.",
    )
    parser.add_argument("--num-bins", type=int, default=6)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument(
        "--mask-strategy", choices=["random", "modality"], default="random"
    )
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pretrain/visualization/visualization_output")
    )
    return parser.parse_args()


def infer_companion(data_path: Path, suffix: str) -> Path:
    stem = data_path.name.removesuffix("_feature_matrix.h5")
    path = data_path.parent / f"{stem.rsplit('_feature', 1)[0]}_{suffix}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def choose_indices(text: str | None, row_count: int, count: int, seed: int):
    if text:
        indices = np.asarray(
            [int(value.strip()) for value in text.split(",") if value.strip()],
            dtype=np.int64,
        )
    else:
        if count <= 0 or count > row_count:
            raise ValueError("--num-bins must be between 1 and the matrix row count")
        generator = np.random.default_rng(seed)
        indices = np.sort(generator.choice(row_count, count, replace=False))
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= row_count):
        raise ValueError("One or more bin indices are outside the HDF5 matrix")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("--bin-indices contains duplicate values")
    return indices


def read_regions(path: Path, selected_indices: np.ndarray):
    wanted = set(selected_indices.tolist())
    regions = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            index = int(row["index"])
            if index in wanted:
                regions[index] = row
                if len(regions) == len(wanted):
                    break
    missing = wanted.difference(regions)
    if missing:
        raise RuntimeError(f"Region metadata missing for matrix rows: {sorted(missing)}")
    return regions


def load_checkpoint(path: Path, feature_names: list[str], device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint_args = checkpoint.get("args")
    model_name = (
        checkpoint_args.model
        if checkpoint_args is not None and hasattr(checkpoint_args, "model")
        else None
    )
    if not model_name or model_name not in models_pretrain.__dict__:
        raise RuntimeError("Checkpoint does not contain a valid pre-training model name")
    typed = "modality_encoder.weight" in checkpoint["model"]
    model_kwargs = (
        {"feature_names": feature_names}
        if typed
        else {"feature_count": len(feature_names)}
    )
    model = models_pretrain.__dict__[model_name](**model_kwargs)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, model_name, int(checkpoint.get("epoch", -1))


def inverse_normalize(values, peak_mask, peak_scales, peak_normalization):
    restored = np.asarray(values, dtype=np.float64).copy()
    if peak_normalization == "log_quantile":
        restored[:, peak_mask] = np.expm1(
            restored[:, peak_mask] * np.log1p(peak_scales[peak_mask])
        )
    elif peak_normalization == "linear_quantile":
        restored[:, peak_mask] *= peak_scales[peak_mask]
    else:
        raise ValueError(f"Unknown peak normalization: {peak_normalization}")
    return restored


def write_values(
    path,
    selected_indices,
    regions,
    feature_names,
    actual,
    prediction,
    actual_raw,
    prediction_raw,
    mask,
):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "matrix_index",
                "chrom",
                "start",
                "end",
                "bin_name",
                "feature_index",
                "feature_name",
                "masked",
                "actual_normalized",
                "predicted_normalized",
                "absolute_error",
                "actual_original",
                "predicted_original",
            ]
        )
        for row_index, matrix_index in enumerate(selected_indices):
            region = regions[int(matrix_index)]
            for feature_index, feature_name in enumerate(feature_names):
                writer.writerow(
                    [
                        matrix_index,
                        region["chrom"],
                        region["start"],
                        region["end"],
                        region["name"],
                        feature_index,
                        feature_name,
                        int(mask[row_index, feature_index]),
                        f"{actual[row_index, feature_index]:.6f}",
                        f"{prediction[row_index, feature_index]:.6f}",
                        f"{abs(actual[row_index, feature_index] - prediction[row_index, feature_index]):.6f}",
                        f"{actual_raw[row_index, feature_index]:.6f}",
                        f"{prediction_raw[row_index, feature_index]:.6f}",
                    ]
                )


def plot_heatmaps(path, actual, prediction, mask, labels):
    reconstructed = actual.copy()
    reconstructed[mask] = prediction[mask]
    error = np.full_like(actual, np.nan)
    error[mask] = np.abs(prediction[mask] - actual[mask])
    value_min = min(0.0, float(reconstructed.min()))
    value_max = max(1.0, float(reconstructed.max()))

    figure, axes = plt.subplots(3, 1, figsize=(16, 2.2 * len(labels) + 4))
    panels = [
        (actual, "Observed normalized values", "viridis", value_min, value_max),
        (
            reconstructed,
            "Reconstructed input (predictions at masked positions)",
            "viridis",
            value_min,
            value_max,
        ),
        (error, "Absolute error at masked positions", "magma", 0.0, None),
    ]
    for axis, (matrix, title, cmap, minimum, maximum) in zip(axes, panels):
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=minimum,
            vmax=maximum,
        )
        axis.set_title(title)
        axis.set_ylabel("Genomic bin")
        axis.set_yticks(np.arange(len(labels)), labels=labels)
        figure.colorbar(image, ax=axis, fraction=0.015, pad=0.01)
    axes[-1].set_xlabel("Feature index in the 288-dimensional model input")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_scatter(path, actual, prediction, mask):
    observed = actual[mask]
    predicted = prediction[mask]
    lower = min(-0.02, float(predicted.min()) - 0.02)
    upper = max(1.02, float(predicted.max()) + 0.02)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(observed, predicted, s=13, alpha=0.45, edgecolors="none")
    axis.plot([lower, upper], [lower, upper], color="red", linewidth=1)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Observed normalized value")
    axis.set_ylabel("Predicted normalized value")
    axis.set_title("Masked-feature reconstruction")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_bin_comparisons(
    path, actual, prediction, mask, feature_names, labels, top_features
):
    columns = 2
    rows = math.ceil(len(labels) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, max(4.5, 4.2 * rows)),
        squeeze=False,
    )
    for bin_index, (axis, label) in enumerate(zip(axes.flat, labels)):
        masked = np.flatnonzero(mask[bin_index])
        errors = np.abs(prediction[bin_index, masked] - actual[bin_index, masked])
        ranked = masked[np.argsort(errors)]
        chosen = ranked[-min(top_features, len(ranked)) :]
        positions = np.arange(len(chosen))
        axis.barh(
            positions - 0.18,
            actual[bin_index, chosen],
            height=0.36,
            label="Observed",
        )
        axis.barh(
            positions + 0.18,
            prediction[bin_index, chosen],
            height=0.36,
            label="Predicted",
        )
        axis.set_yticks(positions, labels=[feature_names[i] for i in chosen])
        axis.set_xlim(0, 1)
        axis.set_title(label)
        axis.set_xlabel("Normalized value")
        axis.legend(loc="lower right")
    for axis in axes.flat[len(labels) :]:
        axis.set_visible(False)
    figure.suptitle("Masked features with the largest reconstruction errors")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main():
    args = parse_args()
    if not 0 < args.mask_ratio < 1:
        raise ValueError("--mask-ratio must be between 0 and 1")
    if args.top_features <= 0:
        raise ValueError("--top-features must be positive")

    args.data = args.data.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.columns = (
        args.columns.resolve()
        if args.columns
        else infer_companion(args.data, "feature_columns.tsv")
    )
    args.regions = (
        args.regions.resolve()
        if args.regions
        else infer_companion(args.data, "regions.tsv")
    )
    args.normalization = (
        args.normalization.resolve()
        if args.normalization
        else args.checkpoint.parent / "feature_normalization.npz"
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    selection = read_feature_selection(args.columns, ("PolIIS5P",))
    normalization = np.load(args.normalization)
    feature_names = normalization["feature_names"].astype(str).tolist()
    if feature_names != selection.names:
        raise RuntimeError("Feature order differs from the pre-training run")
    peak_mask = normalization["peak_mask"].astype(bool)
    peak_scales = normalization["peak_scales"].astype(np.float32)
    peak_normalization = str(
        normalization["peak_normalization"].item()
        if "peak_normalization" in normalization
        else "log_quantile"
    )

    with h5py.File(args.data, "r") as handle:
        row_count = int(handle["matrix"].shape[0])
    selected_indices = choose_indices(
        args.bin_indices, row_count, args.num_bins, args.seed
    )
    regions = read_regions(args.regions, selected_indices)

    dataset = H5BinDataset(
        args.data,
        selected_indices,
        selection.source_indices,
        peak_mask,
        peak_scales,
        peak_normalization,
    )
    values = torch.stack([dataset[index] for index in range(len(dataset))])
    model, model_name, checkpoint_epoch = load_checkpoint(
        args.checkpoint, feature_names, device
    )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        _, loss, prediction, mask = model(
            values.to(device),
            mask_ratio=args.mask_ratio,
            mask_strategy=args.mask_strategy,
        )

    actual = values.numpy()
    prediction = prediction.float().cpu().numpy()
    mask = mask.bool().cpu().numpy()
    actual_raw = inverse_normalize(
        actual, peak_mask, peak_scales, peak_normalization
    )
    prediction_raw = inverse_normalize(
        prediction, peak_mask, peak_scales, peak_normalization
    )
    absolute_error = np.abs(prediction - actual)
    per_bin_mse = np.asarray(
        [np.mean((prediction[i, mask[i]] - actual[i, mask[i]]) ** 2) for i in range(len(actual))]
    )
    per_bin_mae = np.asarray(
        [np.mean(absolute_error[i, mask[i]]) for i in range(len(actual))]
    )

    run_name = f"epoch_{checkpoint_epoch}_seed_{args.seed}"
    output_dir = args.output_dir.resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{regions[int(index)]['chrom']}:{regions[int(index)]['start']}-{regions[int(index)]['end']}"
        for index in selected_indices
    ]

    write_values(
        output_dir / "reconstruction_values.tsv",
        selected_indices,
        regions,
        feature_names,
        actual,
        prediction,
        actual_raw,
        prediction_raw,
        mask,
    )
    with (output_dir / "bin_metrics.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["matrix_index", "region", "masked_features", "mse", "rmse", "mae"]
        )
        for index, label, count, mse, mae in zip(
            selected_indices, labels, mask.sum(axis=1), per_bin_mse, per_bin_mae
        ):
            writer.writerow([index, label, count, mse, math.sqrt(mse), mae])

    summary = {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "model": model_name,
        "device": str(device),
        "seed": args.seed,
        "mask_ratio": args.mask_ratio,
        "mask_strategy": args.mask_strategy,
        "matrix_indices": selected_indices.tolist(),
        "masked_feature_count": int(mask.sum()),
        "masked_mse": float(np.mean((prediction[mask] - actual[mask]) ** 2)),
        "masked_rmse": float(np.sqrt(np.mean((prediction[mask] - actual[mask]) ** 2))),
        "masked_mae": float(np.mean(absolute_error[mask])),
        "model_loss": float(loss),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    np.savez(
        output_dir / "reconstruction_arrays.npz",
        matrix_indices=selected_indices,
        actual=actual,
        prediction=prediction,
        mask=mask,
        feature_names=np.asarray(feature_names),
    )
    plot_heatmaps(
        output_dir / "reconstruction_heatmap.png",
        actual,
        prediction,
        mask,
        labels,
    )
    plot_scatter(
        output_dir / "masked_reconstruction_scatter.png",
        actual,
        prediction,
        mask,
    )
    plot_bin_comparisons(
        output_dir / "bin_feature_comparison.png",
        actual,
        prediction,
        mask,
        feature_names,
        labels,
        args.top_features,
    )

    print(f"Model: {model_name}, checkpoint epoch: {checkpoint_epoch}")
    print(f"Selected matrix rows: {selected_indices.tolist()}")
    print(
        f"Masked reconstruction: MSE={summary['masked_mse']:.6f}, "
        f"RMSE={summary['masked_rmse']:.6f}, MAE={summary['masked_mae']:.6f}"
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
