#!/usr/bin/env python3
"""Build a compact mm10 bin-level multimodal peak table."""

from __future__ import annotations

import argparse
import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_BIN_SIZE = 1000
DEFAULT_H5_CHUNK_ROWS = 2048
EXPECTED_TRAINING_FEATURES = 323

MM10_CHROM_SIZES = {
    "chr1": 195_471_971,
    "chr2": 182_113_224,
    "chr3": 160_039_680,
    "chr4": 156_508_116,
    "chr5": 151_834_684,
    "chr6": 149_736_546,
    "chr7": 145_441_459,
    "chr8": 129_401_213,
    "chr9": 124_595_110,
    "chr10": 130_694_993,
    "chr11": 122_082_543,
    "chr12": 120_129_022,
    "chr13": 120_421_639,
    "chr14": 124_902_244,
    "chr15": 104_043_685,
    "chr16": 98_207_768,
    "chr17": 94_987_271,
    "chr18": 90_702_639,
    "chr19": 61_431_566,
    "chrX": 171_031_299,
    "chrY": 91_744_698,
}

MAIN_CHROMS = list(MM10_CHROM_SIZES)

INPUT_FILES = [
    ("G4", "CPC_G4Seq.tsv", None),
    ("ATAC", "ATAC/CPC_ATACpeak.tsv", None),
    ("NFR", "ATAC/CPC_Diff_NFR.tsv", "V6.5_CMDiff_D5_ATACseq"),
    ("Nucleosome", "ATAC/CPC_Diff_Nucleosome.tsv", "V6.5_CMDiff_D5_ATACseq"),
    ("CTCF", "TF_Histone/V6.5_CMD5_Cutag_CTCF.tsv", None),
    ("Gata4", "TF_Histone/V6.5_CMD5_Cutag_Gata4.tsv", None),
    ("Gata6", "TF_Histone/V6.5_CMD5_Cutag_Gata6.tsv", None),
    ("H3K27ac", "TF_Histone/V6.5_CMD5_Cutag_H3K27ac.tsv", None),
    ("H3K27me3", "TF_Histone/V6.5_CMD5_Cutag_H3K27me3.tsv", None),
    ("H3K4me1", "TF_Histone/V6.5_CMD5_Cutag_H3K4me1.tsv", None),
    ("H3K4me3", "TF_Histone/V6.5_CMD5_Cutag_H3K4me3.tsv", None),
    ("Hand2", "TF_Histone/V6.5_CMD5_Cutag_Hand2.tsv", None),
    ("Isl1", "TF_Histone/V6.5_CMD5_Cutag_Isl1.tsv", None),
    ("Nkx2_5", "TF_Histone/V6.5_CMD5_Cutag_Nkx2.5.tsv", None),
    ("Tbx5", "TF_Histone/V6.5_CMD5_Cutag_Tbx5.tsv", None),
    ("Rloop", "CPC_RloopSeq.tsv", None),
    ("PolIIS5P", "TF_Histone/V6.5_CMD5_Cutag_PolII_S5P.tsv", None),
]


@dataclass
class Modality:
    name: str
    path: Path
    experiment_columns: list[str]
    scaling_factors: list[float]
    peaks: pd.DataFrame


@dataclass
class BinModality:
    peak: np.ndarray
    self_ratio: np.ndarray
    bin_ratio: np.ndarray
    intervals: np.ndarray
    covered_bases: np.ndarray
    merged_intervals: list[tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a full-length-bin mm10 multimodal peak table."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_multimodal_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/preprocessed"))
    parser.add_argument(
        "--bin-size",
        type=int,
        default=DEFAULT_BIN_SIZE,
        help=f"Bin size in base pairs (default: {DEFAULT_BIN_SIZE}).",
    )
    parser.add_argument(
        "--chrom-sizes",
        type=Path,
        default=None,
        help="Optional chrom.sizes file; only chr1-chr19, chrX and chrY are used.",
    )
    parser.add_argument(
        "--scaling-factors",
        type=Path,
        default=None,
        help="Scaling-factor file; defaults to the *SF*.txt file under data-dir.",
    )
    parser.add_argument(
        "--h5-chunk-rows",
        type=int,
        default=DEFAULT_H5_CHUNK_ROWS,
        help=f"Rows per HDF5 chunk (default: {DEFAULT_H5_CHUNK_ROWS}).",
    )
    return parser.parse_args()


def read_scaling_factors(path: Path) -> dict[str, float]:
    factors: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ">" not in line:
                raise ValueError(f"Invalid scaling-factor line: {line}")
            sample, factor_text = line.split(">", 1)
            factors[sample.strip()] = float(factor_text.strip())
    return factors


def find_scaling_factor_file(data_dir: Path, requested: Path | None) -> Path:
    path = requested.resolve() if requested else data_dir / "SF汇总.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_chrom_sizes(path: Path | None) -> dict[str, int]:
    if path is None:
        return dict(MM10_CHROM_SIZES)

    observed: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, size = line.split()[:2]
            observed[chrom] = int(size)
    return {chrom: observed[chrom] for chrom in MAIN_CHROMS if chrom in observed}


def load_modality(
    name: str,
    path: Path,
    factor_prefix: str | None,
    scaling_lookup: dict[str, float],
    allowed_chroms: set[str],
) -> Modality:
    frame = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    if frame.shape[1] < 4:
        raise ValueError(f"{path} must contain coordinates and at least one experiment column")

    chrom_col, start_col, end_col = frame.columns[:3]
    experiment_columns = [str(column) for column in frame.columns[3:]]
    factors: list[float] = []
    missing: list[str] = []
    for column in experiment_columns:
        sample = column.removesuffix("_reads")
        if factor_prefix:
            sample = f"{factor_prefix}_rep{sample.rsplit('_rep', 1)[1]}"
        if sample not in scaling_lookup:
            missing.append(column)
        else:
            factors.append(scaling_lookup[sample])
    if missing:
        raise ValueError(f"No scaling factor for {path.name}: {', '.join(missing)}")

    experiment_values = (
        frame[experiment_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    peak_values = (experiment_values * np.asarray(factors)).mean(axis=1)
    peaks = pd.DataFrame(
        {
            "chrom": frame[chrom_col].astype(str).str.strip(),
            "start": pd.to_numeric(frame[start_col], errors="coerce"),
            "end": pd.to_numeric(frame[end_col], errors="coerce"),
            "value": peak_values,
        }
    ).dropna(subset=["start", "end"])
    peaks["start"] = peaks["start"].astype(np.int64)
    peaks["end"] = peaks["end"].astype(np.int64)
    peaks = peaks[
        peaks["chrom"].isin(allowed_chroms)
        & (peaks["start"] >= 0)
        & (peaks["end"] > peaks["start"])
    ]
    peaks = (
        peaks.groupby(["chrom", "start", "end"], as_index=False, sort=False)["value"]
        .max()
        .sort_values(["chrom", "start", "end"])
        .reset_index(drop=True)
    )
    return Modality(name, path, experiment_columns, factors, peaks)


def calculate_peak_in_bin(
    peak_start: int,
    peak_end: int,
    bin_start: int,
    bin_end: int,
) -> tuple[int, float]:
    overlap = max(0, min(peak_end, bin_end) - max(peak_start, bin_start))
    ratio = overlap / (peak_end - peak_start) if overlap else 0.0
    return overlap, ratio


def merge_intervals(
    intervals: list[tuple[int, int]],
    region_end: int,
) -> list[tuple[int, int]]:
    clipped = sorted(
        (max(0, start), min(end, region_end))
        for start, end in intervals
        if min(end, region_end) > max(0, start)
    )
    if not clipped:
        return []

    merged = [clipped[0]]
    for start, end in clipped[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def covered_bases_per_bin(
    intervals: list[tuple[int, int]],
    bin_size: int,
    bin_count: int,
) -> np.ndarray:
    covered = np.zeros(bin_count, dtype=np.int32)
    for start, end in intervals:
        first_bin = start // bin_size
        last_bin = min((end - 1) // bin_size, bin_count - 1)
        for bin_index in range(first_bin, last_bin + 1):
            bin_start = bin_index * bin_size
            covered[bin_index] += max(
                0,
                min(end, bin_start + bin_size) - max(start, bin_start),
            )
    return covered


def calculate_modality_for_bins(
    modality: Modality,
    chrom: str,
    bin_size: int,
    bin_count: int,
) -> BinModality:
    peak_values = np.zeros(bin_count, dtype=np.float32)
    self_ratios = np.zeros(bin_count, dtype=np.float32)
    interval_text = np.full(bin_count, "-", dtype=object)
    interval_lists: dict[int, list[str]] = {}
    region_end = bin_count * bin_size
    chrom_peaks = modality.peaks.loc[
        modality.peaks["chrom"] == chrom,
        ["start", "end", "value"],
    ]

    raw_intervals: list[tuple[int, int]] = []
    for start, end, value in chrom_peaks.itertuples(index=False, name=None):
        start, end = int(start), int(end)
        clipped_start = max(0, start)
        clipped_end = min(end, region_end)
        if clipped_end <= clipped_start:
            continue
        raw_intervals.append((start, end))
        first_bin = clipped_start // bin_size
        last_bin = (clipped_end - 1) // bin_size
        original_interval = f"{start}-{end}"
        for bin_index in range(first_bin, last_bin + 1):
            bin_start = bin_index * bin_size
            _, self_ratio = calculate_peak_in_bin(
                start,
                end,
                bin_start,
                bin_start + bin_size,
            )
            if value > peak_values[bin_index] or (
                value == peak_values[bin_index]
                and self_ratio > self_ratios[bin_index]
            ):
                peak_values[bin_index] = value
                self_ratios[bin_index] = self_ratio
            interval_lists.setdefault(bin_index, []).append(original_interval)

    for bin_index, values in interval_lists.items():
        interval_text[bin_index] = ";".join(values)

    merged = merge_intervals(raw_intervals, region_end)
    covered_bases = covered_bases_per_bin(merged, bin_size, bin_count)
    return BinModality(
        peak=peak_values,
        self_ratio=self_ratios,
        bin_ratio=covered_bases.astype(np.float32) / float(bin_size),
        intervals=interval_text,
        covered_bases=covered_bases,
        merged_intervals=merged,
    )


def intersect_intervals(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    shared: list[tuple[int, int]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            shared.append((start, end))
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return shared


def calculate_overlapping_ratios(
    left: BinModality,
    right: BinModality,
    bin_size: int,
    bin_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    shared = intersect_intervals(left.merged_intervals, right.merged_intervals)
    shared_bases = covered_bases_per_bin(shared, bin_size, bin_count)
    right_overlapping_left = np.divide(
        shared_bases,
        left.covered_bases,
        out=np.zeros(bin_count, dtype=np.float32),
        where=left.covered_bases > 0,
    )
    left_overlapping_right = np.divide(
        shared_bases,
        right.covered_bases,
        out=np.zeros(bin_count, dtype=np.float32),
        where=right.covered_bases > 0,
    )
    return right_overlapping_left, left_overlapping_right


def build_chromosome_bins(
    chrom: str,
    chrom_size: int,
    bin_size: int,
    modalities: list[Modality],
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    bin_count = chrom_size // bin_size
    starts = np.arange(bin_count, dtype=np.int64) * bin_size
    results = {
        modality.name: calculate_modality_for_bins(
            modality,
            chrom,
            bin_size,
            bin_count,
        )
        for modality in modalities
    }

    pair_columns: dict[str, np.ndarray] = {}
    for left_index, left_modality in enumerate(modalities):
        for right_modality in modalities[left_index + 1 :]:
            right_over_left, left_over_right = calculate_overlapping_ratios(
                results[left_modality.name],
                results[right_modality.name],
                bin_size,
                bin_count,
            )
            pair_columns[
                f"{right_modality.name}_overlapping_{left_modality.name}"
            ] = right_over_left
            pair_columns[
                f"{left_modality.name}_overlapping_{right_modality.name}"
            ] = left_over_right

    data: dict[str, object] = {
        "chrom": np.full(bin_count, chrom, dtype=object),
        "bin_start": starts,
        "bin_end": starts + bin_size,
        "bin_id": np.asarray(
            [f"{chrom}:{start}-{start + bin_size}" for start in starts],
            dtype=object,
        ),
    }
    ordered_feature_columns: list[str] = []
    numeric_columns: list[str] = []
    for modality in modalities:
        name = modality.name
        group = {
            f"{name}_peak": results[name].peak,
            f"{name}_self_ratio": results[name].self_ratio,
            f"{name}_bin_ratio": results[name].bin_ratio,
        }
        for other in modalities:
            if other.name != name:
                group[f"{other.name}_overlapping_{name}"] = pair_columns[
                    f"{other.name}_overlapping_{name}"
                ]
        group[f"{name}_iv"] = results[name].intervals
        data.update(group)
        ordered_feature_columns.extend(group)
        numeric_columns.extend(column for column in group if not column.endswith("_iv"))

    for column in numeric_columns:
        data[column] = np.round(np.asarray(data[column], dtype=np.float32), 2)
    numeric = np.column_stack([data[column] for column in numeric_columns])
    zero_count = (numeric == 0).sum(axis=1).astype(np.int16)
    nonzero_count = (numeric.shape[1] - zero_count).astype(np.int16)
    ratio_columns = [
        column
        for column in numeric_columns
        if column.endswith("_ratio") or "_overlapping_" in column
    ]
    invalid_ratio_count = sum(
        int(((np.asarray(data[column]) < 0) | (np.asarray(data[column]) > 1.000001)).sum())
        for column in ratio_columns
    )
    if invalid_ratio_count:
        raise RuntimeError(f"{chrom}: found {invalid_ratio_count} ratio values outside [0, 1]")
    data["zero_count"] = zero_count
    data["nonzero_count"] = nonzero_count

    columns = [
        "chrom",
        "bin_start",
        "bin_end",
        "bin_id",
        "zero_count",
        "nonzero_count",
        *ordered_feature_columns,
    ]
    frame = pd.DataFrame(data)[columns]
    summary = {
        "chrom": chrom,
        "total_bins": bin_count,
        "all_zero_bins": int((nonzero_count == 0).sum()),
        "non_all_zero_bins": int((nonzero_count > 0).sum()),
        "max_ratio": max(
            (float(np.asarray(data[column]).max(initial=0.0)) for column in ratio_columns),
            default=0.0,
        ),
        "invalid_ratio_count": invalid_ratio_count,
    }
    return frame, summary


def write_manifest(
    path: Path,
    modalities: list[Modality],
) -> None:
    rows = [
        {
            "modality": modality.name,
            "source_file": str(modality.path),
            "experiment_columns": ",".join(modality.experiment_columns),
            "scaling_factors": ",".join(map(str, modality.scaling_factors)),
            "peak_rows": len(modality.peaks),
        }
        for modality in modalities
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def training_feature_names(modalities: list[Modality]) -> list[str]:
    columns: list[str] = []
    for modality in modalities:
        name = modality.name
        columns.extend(
            [
                f"{name}_peak",
                f"{name}_self_ratio",
                f"{name}_bin_ratio",
                *[
                    f"{other.name}_overlapping_{name}"
                    for other in modalities
                    if other.name != name
                ],
            ]
        )
    return columns


def create_research_outputs(
    chromosome_nonzero: dict[str, np.ndarray],
    modality_names: list[str],
    modality_counts: np.ndarray,
    joint_counts: np.ndarray,
    total_bins: int,
    output_dir: Path,
    window_bins: int = 1000,
) -> None:
    """Draw regional density and genome-wide modality co-occurrence heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chromosome_windows = [
        np.round(
            np.asarray(
                [
                    values[start : start + window_bins].mean()
                    for start in range(0, len(values), window_bins)
                ],
                dtype=np.float32,
            ),
            2,
        )
        for values in chromosome_nonzero.values()
    ]
    width = max(map(len, chromosome_windows))
    density = np.full((len(chromosome_windows), width), np.nan, dtype=np.float32)
    for row, values in enumerate(chromosome_windows):
        density[row, : len(values)] = values

    visible_values = density[np.isfinite(density)]
    color_max = max(float(np.percentile(visible_values, 99)), 0.01)
    reds = plt.colormaps["Reds"].copy()
    reds.set_bad("white")
    plt.figure(figsize=(16, 8))
    image = plt.imshow(
        density,
        aspect="auto",
        interpolation="nearest",
        cmap=reds,
        vmin=0,
        vmax=color_max,
    )
    plt.colorbar(image, label="Fraction of non-all-zero bins")
    plt.yticks(range(len(chromosome_nonzero)), chromosome_nonzero)
    plt.xlabel(f"Chromosome position ({window_bins} bins per column)")
    plt.ylabel("Chromosome")
    plt.tight_layout()
    plt.savefig(output_dir / "chromosome_nonzero_heatmap.png", dpi=220)
    plt.close()

    presence_probability = (modality_counts + 0.5) / (total_bins + 1.0)
    joint_probability = (joint_counts + 0.5) / (total_bins + 1.0)
    enrichment = np.log2(
        joint_probability / np.outer(presence_probability, presence_probability)
    )
    np.fill_diagonal(enrichment, 0.0)
    enrichment = np.round(enrichment, 2)
    pd.DataFrame(
        enrichment,
        index=modality_names,
        columns=modality_names,
    ).to_csv(
        output_dir / "modality_cooccurrence_log2_enrichment.tsv",
        sep="\t",
        float_format="%.2f",
    )

    off_diagonal = enrichment[~np.eye(len(modality_names), dtype=bool)]
    limit = max(float(np.percentile(np.abs(off_diagonal), 95)), 0.25)
    plt.figure(figsize=(11, 9))
    image = plt.imshow(
        enrichment,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    plt.colorbar(image, label="log2 co-occurrence enrichment")
    plt.xticks(range(len(modality_names)), modality_names, rotation=45, ha="right")
    plt.yticks(range(len(modality_names)), modality_names)
    plt.tight_layout()
    plt.savefig(output_dir / "modality_cooccurrence_enrichment_heatmap.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be a positive integer")
    if args.h5_chunk_rows <= 0:
        raise ValueError("--h5-chunk-rows must be a positive integer")

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve() / f"bin_{args.bin_size}bp"
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_dir = output_dir / f"pretrain_{args.bin_size}bp"
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    chrom_sizes = read_chrom_sizes(args.chrom_sizes)
    scaling_file = find_scaling_factor_file(data_dir, args.scaling_factors)
    scaling_lookup = read_scaling_factors(scaling_file)
    modalities = [
        load_modality(
            name,
            data_dir / relative_path,
            factor_prefix,
            scaling_lookup,
            set(chrom_sizes),
        )
        for name, relative_path, factor_prefix in INPUT_FILES
    ]
    names = [modality.name for modality in modalities]

    main_table = output_dir / f"mm10_{args.bin_size}_multimodal_chrall_all_bins.tsv.gz"
    nonzero_table = output_dir / f"mm10_{args.bin_size}_multimodal_chrall_nonallzero_bins.tsv.gz"
    bin_index_table = pretrain_dir / f"mm10_{args.bin_size}_multimodal_chrall_region.tsv"
    feature_matrix_file = pretrain_dir / f"mm10_{args.bin_size}_multimodal_chrall_feature_matrixs.h5"
    feature_column_table = pretrain_dir / f"mm10_{args.bin_size}_multimodal_chrall_feature_columns.tsv"
    validation_rows: list[dict[str, int | float]] = []
    chromosome_nonzero: dict[str, np.ndarray] = {}
    modality_counts = np.zeros(len(modalities), dtype=np.int64)
    joint_counts = np.zeros((len(modalities), len(modalities)), dtype=np.int64)
    total_bins_for_plots = 0
    training_columns = training_feature_names(modalities)
    if len(training_columns) != EXPECTED_TRAINING_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAINING_FEATURES} training features, "
            f"found {len(training_columns)}"
        )
    pd.DataFrame(
        {"index": np.arange(len(training_columns)), "name": training_columns}
    ).to_csv(feature_column_table, sep="\t", index=False)
    training_row_count = 0
    first_main = True
    first_nonzero = True
    with gzip.open(
        main_table,
        "wt",
        newline="",
        compresslevel=1,
    ) as main_handle, gzip.open(
        nonzero_table,
        "wt",
        newline="",
        compresslevel=1,
    ) as nonzero_handle, bin_index_table.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as bin_handle, h5py.File(feature_matrix_file, "w") as h5_handle:
        bin_handle.write("chrom\tstart\tend\tname\tindex\n")
        feature_dataset = h5_handle.create_dataset(
            "matrix",
            shape=(0, len(training_columns)),
            maxshape=(None, len(training_columns)),
            chunks=(args.h5_chunk_rows, len(training_columns)),
            dtype=np.float32,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        feature_dataset.attrs.update(
            bin_size=args.bin_size,
            feature_count=len(training_columns),
            columns_file=feature_column_table.name,
        )
        for chrom, chrom_size in chrom_sizes.items():
            frame, summary = build_chromosome_bins(
                chrom,
                chrom_size,
                args.bin_size,
                modalities,
            )
            validation_rows.append(summary)
            chromosome_nonzero[chrom] = frame["nonzero_count"].to_numpy() > 0
            presence = np.column_stack(
                [
                    frame[f"{modality.name}_bin_ratio"].to_numpy() > 0
                    for modality in modalities
                ]
            ).astype(np.int64)
            modality_counts += presence.sum(axis=0)
            joint_counts += presence.T @ presence
            total_bins_for_plots += len(frame)
            frame.to_csv(
                main_handle,
                sep="\t",
                index=False,
                header=first_main,
                quoting=csv.QUOTE_MINIMAL,
            )
            first_main = False

            nonzero = frame.loc[frame["nonzero_count"] > 0]
            if not nonzero.empty:
                nonzero.to_csv(
                    nonzero_handle,
                    sep="\t",
                    index=False,
                    header=first_nonzero,
                    quoting=csv.QUOTE_MINIMAL,
                )
                first_nonzero = False
                first_index = training_row_count
                block = nonzero[training_columns].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )
                training_row_count += len(block)
                feature_dataset.resize(training_row_count, axis=0)
                feature_dataset[first_index:training_row_count] = block

                bins = nonzero[
                    ["chrom", "bin_start", "bin_end", "bin_id"]
                ].rename(
                    columns={"bin_start": "start", "bin_end": "end", "bin_id": "name"}
                )
                bins["index"] = np.arange(
                    first_index,
                    training_row_count,
                    dtype=np.int64,
                )
                bins.to_csv(
                    bin_handle,
                    sep="\t",
                    index=False,
                    header=False,
                    quoting=csv.QUOTE_MINIMAL,
                )
            print(
                f"{chrom}: {len(frame):,} full bins, "
                f"{len(nonzero):,} bins with at least one non-zero value"
            )
        feature_dataset.attrs["row_count"] = training_row_count

    validation = pd.DataFrame(validation_rows)
    validation.loc[len(validation)] = {
        "chrom": "all_chromosomes",
        "total_bins": int(validation["total_bins"].sum()),
        "all_zero_bins": int(validation["all_zero_bins"].sum()),
        "non_all_zero_bins": int(validation["non_all_zero_bins"].sum()),
        "max_ratio": float(validation["max_ratio"].max()),
        "invalid_ratio_count": int(validation["invalid_ratio_count"].sum()),
    }
    validation["non_all_zero_fraction"] = (
        validation["non_all_zero_bins"] / validation["total_bins"]
    ).round(2)
    validation.to_csv(
        output_dir / "validation_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.2f",
    )
    write_manifest(output_dir / "column_manifest.tsv", modalities)
    create_research_outputs(
        chromosome_nonzero,
        names,
        modality_counts,
        joint_counts,
        total_bins_for_plots,
        output_dir,
    )

    print(f"Main table: {main_table}")
    print(f"Non-all-zero table: {nonzero_table}")
    print(f"Validation summary: {output_dir / 'validation_summary.tsv'}")
    print(f"Column manifest: {output_dir / 'column_manifest.tsv'}")
    print(f"Pretraining bins: {bin_index_table}")
    print(f"Pretraining features: {feature_matrix_file}")
    print(f"Feature columns: {feature_column_table}")
    print(f"Chromosome heatmap: {output_dir / 'chromosome_nonzero_heatmap.png'}")
    print(
        "Co-occurrence enrichment: "
        f"{output_dir / 'modality_cooccurrence_enrichment_heatmap.png'}"
    )


if __name__ == "__main__":
    main()
