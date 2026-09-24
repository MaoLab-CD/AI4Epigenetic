#!/usr/bin/env python3
"""Build the mm10 bin-level multimodal pretraining dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_BIN_SIZE = 1000
DEFAULT_H5_CHUNK_ROWS = 2048
EXPECTED_TRAINING_FEATURES = 288

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


@dataclass(frozen=True)
class InputSpec:
    name: str
    relative_path: str
    factor_prefix: str | None = None


INPUT_SPECS = (
    InputSpec("G4", "CPC_G4Seq.tsv"),
    InputSpec("ATAC", "ATAC/CPC_ATACpeak.tsv"),
    InputSpec("NFR", "ATAC/CPC_Diff_NFR.tsv", "V6.5_CMDiff_D5_ATACseq"),
    InputSpec(
        "Nucleosome",
        "ATAC/CPC_Diff_Nucleosome.tsv",
        "V6.5_CMDiff_D5_ATACseq",
    ),
    InputSpec("CTCF", "TF_Histone/V6.5_CMD5_Cutag_CTCF.tsv"),
    InputSpec("Gata4", "TF_Histone/V6.5_CMD5_Cutag_Gata4.tsv"),
    InputSpec("Gata6", "TF_Histone/V6.5_CMD5_Cutag_Gata6.tsv"),
    InputSpec("H3K27ac", "TF_Histone/V6.5_CMD5_Cutag_H3K27ac.tsv"),
    InputSpec("H3K27me3", "TF_Histone/V6.5_CMD5_Cutag_H3K27me3.tsv"),
    InputSpec("H3K4me1", "TF_Histone/V6.5_CMD5_Cutag_H3K4me1.tsv"),
    InputSpec("H3K4me3", "TF_Histone/V6.5_CMD5_Cutag_H3K4me3.tsv"),
    InputSpec("Hand2", "TF_Histone/V6.5_CMD5_Cutag_Hand2.tsv"),
    InputSpec("Isl1", "TF_Histone/V6.5_CMD5_Cutag_Isl1.tsv"),
    InputSpec("Nkx2_5", "TF_Histone/V6.5_CMD5_Cutag_Nkx2.5.tsv"),
    InputSpec("Tbx5", "TF_Histone/V6.5_CMD5_Cutag_Tbx5.tsv"),
    InputSpec("Rloop", "CPC_RloopSeq.tsv"),
)

# Stable tuple API reused by the two downstream preprocessing scripts. Pol II
# remains outside INPUT_SPECS so the pretraining matrix stays at 288 features.
INPUT_FILES = [
    (spec.name, spec.relative_path, spec.factor_prefix) for spec in INPUT_SPECS
] + [
    (
        "PolIIS5P",
        "TF_Histone/V6.5_CMD5_Cutag_PolII_S5P.tsv",
        None,
    )
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


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    pretrain: Path
    main_table: Path
    nonzero_table: Path
    regions: Path
    feature_matrix: Path
    feature_columns: Path


@dataclass
class RunStatistics:
    modality_count: int
    validation_rows: list[dict[str, int | float]] = field(default_factory=list)
    chromosome_nonzero: dict[str, np.ndarray] = field(default_factory=dict)
    total_bins: int = 0
    training_rows: int = 0

    def __post_init__(self) -> None:
        self.modality_counts = np.zeros(self.modality_count, dtype=np.int64)
        self.joint_counts = np.zeros(
            (self.modality_count, self.modality_count), dtype=np.int64
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Divide the mm10 genome into bins and build a table of "
            "within-bin multimodal relationships."
        )
    )
    parser.add_argument(
        "--data_dir", type=Path, default=Path("data/raw_multimodals_data")
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/preprocessed/pretrain_data")
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=DEFAULT_BIN_SIZE,
        help=f"Bin size in base pairs (default: {DEFAULT_BIN_SIZE}).",
    )
    parser.add_argument(
        "--chrom_selection",
        type=str,
        default=None,
        help="Optional comma-separated subset of chr1-chr19, chrX and chrY.",
    )
    parser.add_argument(
        "--scaling_factors",
        type=Path,
        default=None,
        help="Scaling-factor file (default: SF_Summary.txt under data_dir).",
    )
    parser.add_argument(
        "--h5_chunk_rows",
        type=int,
        default=DEFAULT_H5_CHUNK_ROWS,
        help=f"Rows per HDF5 chunk (default: {DEFAULT_H5_CHUNK_ROWS}).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.bin_size <= 0:
        raise ValueError("--bin_size must be a positive integer")
    if args.h5_chunk_rows <= 0:
        raise ValueError("--h5_chunk_rows must be a positive integer")


def select_chromosomes(selection: str | None) -> dict[str, int]:
    if selection is None:
        return dict(MM10_CHROM_SIZES)

    chromosomes = [chrom.strip() for chrom in selection.split(",") if chrom.strip()]
    invalid = [chrom for chrom in chromosomes if chrom not in MM10_CHROM_SIZES]
    if invalid:
        raise ValueError(
            f"Invalid chromosome(s): {', '.join(invalid)}. "
            f"Available chromosomes: {', '.join(MM10_CHROM_SIZES)}"
        )
    return {chrom: MM10_CHROM_SIZES[chrom] for chrom in chromosomes}


def build_output_paths(output_dir: Path, bin_size: int) -> OutputPaths:
    root = output_dir.resolve() / f"bin_{bin_size}bp"
    pretrain = root / "pretrain_input"
    pretrain.mkdir(parents=True, exist_ok=True)
    prefix = f"mm10_{bin_size}"
    return OutputPaths(
        root=root,
        pretrain=pretrain,
        main_table=root / f"{prefix}_multimodal_chrall_all_bins.tsv.gz",
        nonzero_table=root / f"{prefix}_multimodal_chrall_nonallzero_bins.tsv.gz",
        regions=pretrain / f"{prefix}_regions.tsv",
        feature_matrix=pretrain / f"{prefix}_feature_matrix.h5",
        feature_columns=pretrain / f"{prefix}_feature_columns.tsv",
    )


def read_scaling_factors(
    scaling_factors: Path | None, data_dir: Path | None = None
) -> dict[str, float]:
    if scaling_factors is None and data_dir is None:
        raise ValueError("data_dir is required when scaling_factors is not provided")
    path = (
        scaling_factors.resolve()
        if scaling_factors
        else Path(data_dir) / "SF_Summary.txt"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    factors: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ">" not in line:
                raise ValueError(f"Invalid scaling-factor line: {line}")
            sample, value = line.split(">", 1)
            factors[sample.strip()] = float(value.strip())
    return factors


def find_scaling_factor_file(data_dir: Path, explicit: Path | None = None) -> Path:
    """Resolve the scaling-factor file for downstream preprocessing."""
    path = explicit.resolve() if explicit else data_dir.resolve() / "SF_Summary.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def scaling_factor_name(column: str, factor_prefix: str | None) -> str:
    sample = column.removesuffix("_reads")
    if factor_prefix is None:
        return sample
    if "_rep" not in sample:
        raise ValueError(f"Cannot extract replicate number from column: {column}")
    replicate = sample.rsplit("_rep", 1)[1]
    return f"{factor_prefix}_rep{replicate}"


def load_modality(
    modality_name: str,
    path: Path,
    factor_prefix: str | None,
    scaling_lookup: dict[str, float],
    allowed_chroms: set[str],
) -> Modality:
    frame = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    if frame.shape[1] < 4:
        raise ValueError(f"{path} must contain coordinates and experiment columns")

    chrom_column, start_column, end_column = frame.columns[:3]
    experiment_columns = [str(column) for column in frame.columns[3:]]
    factor_names = [
        scaling_factor_name(column, factor_prefix)
        for column in experiment_columns
    ]
    missing = [
        column
        for column, factor_name in zip(experiment_columns, factor_names)
        if factor_name not in scaling_lookup
    ]
    if missing:
        raise ValueError(f"No scaling factor for {path.name}: {', '.join(missing)}")

    factors = [scaling_lookup[name] for name in factor_names]
    experiments = (
        frame[experiment_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    # Every replicate is corrected first; the corrected replicates then contribute equally.
    peak_values = (experiments * np.asarray(factors)).mean(axis=1)

    peaks = pd.DataFrame(
        {
            "chrom": frame[chrom_column].astype(str).str.strip(),
            "start": pd.to_numeric(frame[start_column], errors="coerce"),
            "end": pd.to_numeric(frame[end_column], errors="coerce"),
            "value": peak_values,
        }
    ).dropna(subset=["start", "end"])
    peaks[["start", "end"]] = peaks[["start", "end"]].astype(np.int64)
    peaks = peaks.loc[
        peaks["chrom"].isin(allowed_chroms)
        & peaks["start"].ge(0)
        & peaks["end"].gt(peaks["start"])
    ]
    # Duplicate genomic intervals are collapsed without changing their strongest signal.
    peaks = (
        peaks.groupby(["chrom", "start", "end"], as_index=False, sort=False)["value"]
        .max()
        .sort_values(["chrom", "start", "end"])
        .reset_index(drop=True)
    )
    return Modality(modality_name, path, experiment_columns, factors, peaks)


def load_modalities(
    data_dir: Path,
    scaling_lookup: dict[str, float],
    chromosomes: set[str],
) -> list[Modality]:
    return [
        load_modality(
            spec.name,
            data_dir / spec.relative_path,
            spec.factor_prefix,
            scaling_lookup,
            chromosomes,
        )
        for spec in INPUT_SPECS
    ]


def overlap_length(
    left_start: int, left_end: int, right_start: int, right_end: int
) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def merge_intervals(
    intervals: list[tuple[int, int]], region_end: int
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


def intersect_intervals(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    shared: list[tuple[int, int]] = []
    left_index = right_index = 0
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


def interval_bases(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


@dataclass
class RegionModality:
    """One modality summarized inside an arbitrary downstream region."""

    peak: float
    self_ratio: float
    bin_ratio: float
    covered_bases: int
    intervals: list[tuple[int, int]]
    interval_text: str


def map_records_to_regions(
    regions: pd.DataFrame,
    records: pd.DataFrame,
    value_column: str,
) -> list[list[tuple[int, int, object]]]:
    """Map sorted peak records to possibly variable-length genomic regions."""
    mapped: list[list[tuple[int, int, object]]] = [[] for _ in range(len(regions))]
    for chrom, region_group in regions.groupby("chrom", sort=False):
        record_rows = list(
            records.loc[records["chrom"] == chrom, ["start", "end", value_column]]
            .itertuples(index=False, name=None)
        )
        active: list[tuple[int, int, object]] = []
        pointer = 0
        for row in region_group.itertuples():
            while pointer < len(record_rows) and record_rows[pointer][0] < row.end:
                active.append(record_rows[pointer])
                pointer += 1
            active = [record for record in active if record[1] > row.start]
            mapped[row.Index] = [record for record in active if record[0] < row.end]
    return mapped


def merge_clipped(
    records: list[tuple[int, int, object]], start: int, end: int
) -> list[tuple[int, int]]:
    """Clip peaks to one region and merge their covered bases."""
    intervals = sorted(
        (max(left, start), min(right, end))
        for left, right, _ in records
        if min(right, end) > max(left, start)
    )
    merged: list[tuple[int, int]] = []
    for left, right in intervals:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def calculate_modality(
    records: list[tuple[int, int, object]], start: int, end: int
) -> RegionModality:
    """Apply the pretraining peak and coverage rules to any genomic region."""
    if not records:
        return RegionModality(0.0, 0.0, 0.0, 0, [], "-")

    def rank(record: tuple[int, int, object]) -> tuple[float, float]:
        left, right, value = record
        overlap = overlap_length(left, right, start, end)
        return float(value), overlap / (right - left)

    selected = max(records, key=rank)
    selected_overlap = overlap_length(selected[0], selected[1], start, end)
    intervals = merge_clipped(records, start, end)
    covered = interval_bases(intervals)
    return RegionModality(
        peak=float(selected[2]),
        self_ratio=selected_overlap / (selected[1] - selected[0]),
        bin_ratio=covered / (end - start),
        covered_bases=covered,
        intervals=intervals,
        interval_text=";".join(
            f"{left}-{right}" for left, right, _ in sorted(set(records))
        ),
    )


def covered_bases_per_bin(
    intervals: list[tuple[int, int]], bin_size: int, bin_count: int
) -> np.ndarray:
    covered = np.zeros(bin_count, dtype=np.int32)
    for start, end in intervals:
        first_bin = start // bin_size
        last_bin = min((end - 1) // bin_size, bin_count - 1)
        for bin_index in range(first_bin, last_bin + 1):
            bin_start = bin_index * bin_size
            covered[bin_index] += overlap_length(
                start, end, bin_start, bin_start + bin_size
            )
    return covered


def calculate_modality_for_bins(
    modality: Modality, chrom: str, bin_size: int, bin_count: int
) -> BinModality:
    peak_values = np.zeros(bin_count, dtype=np.float32)
    self_ratios = np.zeros(bin_count, dtype=np.float32)
    interval_text = np.full(bin_count, "-", dtype=object)
    interval_lists: dict[int, list[str]] = {}
    region_end = bin_count * bin_size
    chrom_peaks = modality.peaks.loc[
        modality.peaks["chrom"].eq(chrom), ["start", "end", "value"]
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
            overlap = overlap_length(start, end, bin_start, bin_start + bin_size)
            self_ratio = overlap / (end - start)
            # The representative peak is the strongest one; self ratio breaks ties.
            if value > peak_values[bin_index] or (
                value == peak_values[bin_index]
                and self_ratio > self_ratios[bin_index]
            ):
                peak_values[bin_index] = value
                self_ratios[bin_index] = self_ratio
            interval_lists.setdefault(bin_index, []).append(original_interval)

    for bin_index, values in interval_lists.items():
        interval_text[bin_index] = ";".join(values)

    # Coverage uses the union of all peaks, so overlapping peaks are counted once.
    merged_intervals = merge_intervals(raw_intervals, region_end)
    covered_bases = covered_bases_per_bin(merged_intervals, bin_size, bin_count)
    return BinModality(
        peak=peak_values,
        self_ratio=self_ratios,
        bin_ratio=covered_bases.astype(np.float32) / float(bin_size),
        intervals=interval_text,
        covered_bases=covered_bases,
        merged_intervals=merged_intervals,
    )


def calculate_overlapping_ratios(
    left: BinModality,
    right: BinModality,
    bin_size: int,
    bin_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    shared_intervals = intersect_intervals(
        left.merged_intervals, right.merged_intervals
    )
    shared_bases = covered_bases_per_bin(shared_intervals, bin_size, bin_count)

    # Both directions share a numerator but use different denominator modalities.
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


def modality_feature_names(modality: str, all_modalities: list[str]) -> list[str]:
    return [
        f"{modality}_peak",
        f"{modality}_self_ratio",
        f"{modality}_bin_ratio",
        *[
            f"{other}_overlapping_{modality}"
            for other in all_modalities
            if other != modality
        ],
    ]


def training_feature_names(modalities: list[Modality]) -> list[str]:
    names = [modality.name for modality in modalities]
    return [
        feature
        for modality in names
        for feature in modality_feature_names(modality, names)
    ]


def calculate_pair_columns(
    modalities: list[Modality],
    results: dict[str, BinModality],
    bin_size: int,
    bin_count: int,
) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    for left_index, left in enumerate(modalities):
        for right in modalities[left_index + 1 :]:
            right_over_left, left_over_right = calculate_overlapping_ratios(
                results[left.name], results[right.name], bin_size, bin_count
            )
            columns[f"{right.name}_overlapping_{left.name}"] = right_over_left
            columns[f"{left.name}_overlapping_{right.name}"] = left_over_right
    return columns


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
            modality, chrom, bin_size, bin_count
        )
        for modality in modalities
    }
    pair_columns = calculate_pair_columns(
        modalities, results, bin_size, bin_count
    )

    data: dict[str, object] = {
        "chrom": np.full(bin_count, chrom, dtype=object),
        "bin_start": starts,
        "bin_end": starts + bin_size,
        "bin_id": np.asarray(
            [f"{chrom}:{start}-{start + bin_size}" for start in starts],
            dtype=object,
        ),
    }
    ordered_features: list[str] = []
    numeric_features: list[str] = []
    modality_names = [modality.name for modality in modalities]

    for name in modality_names:
        feature_names = modality_feature_names(name, modality_names)
        group: dict[str, object] = {
            f"{name}_peak": results[name].peak,
            f"{name}_self_ratio": results[name].self_ratio,
            f"{name}_bin_ratio": results[name].bin_ratio,
            **{column: pair_columns[column] for column in feature_names[3:]},
            f"{name}_iv": results[name].intervals,
        }
        data.update(group)
        ordered_features.extend(group)
        numeric_features.extend(feature_names)

    # Rounding occurs before zero counting and HDF5 writing, as in the current pipeline.
    for column in numeric_features:
        data[column] = np.round(np.asarray(data[column], dtype=np.float32), 2)

    numeric = np.column_stack([data[column] for column in numeric_features])
    zero_count = (numeric == 0).sum(axis=1).astype(np.int16)
    nonzero_count = (numeric.shape[1] - zero_count).astype(np.int16)
    ratio_columns = [
        column
        for column in numeric_features
        if column.endswith("_ratio") or "_overlapping_" in column
    ]
    invalid_ratio_count = sum(
        int(
            (
                (np.asarray(data[column]) < 0)
                | (np.asarray(data[column]) > 1.000001)
            ).sum()
        )
        for column in ratio_columns
    )
    if invalid_ratio_count:
        raise RuntimeError(
            f"{chrom}: found {invalid_ratio_count} ratio values outside [0, 1]"
        )

    data["zero_count"] = zero_count
    data["nonzero_count"] = nonzero_count
    columns = [
        "chrom",
        "bin_start",
        "bin_end",
        "bin_id",
        "zero_count",
        "nonzero_count",
        *ordered_features,
    ]
    frame = pd.DataFrame(data)[columns]
    summary = {
        "chrom": chrom,
        "total_bins": bin_count,
        "all_zero_bins": int((nonzero_count == 0).sum()),
        "non_all_zero_bins": int((nonzero_count > 0).sum()),
        "max_ratio": max(
            (
                float(np.asarray(data[column]).max(initial=0.0))
                for column in ratio_columns
            ),
            default=0.0,
        ),
        "invalid_ratio_count": invalid_ratio_count,
    }
    return frame, summary


def update_statistics(
    statistics: RunStatistics, frame: pd.DataFrame, summary: dict[str, int | float]
) -> None:
    statistics.validation_rows.append(summary)
    chrom = str(summary["chrom"])
    statistics.chromosome_nonzero[chrom] = frame["nonzero_count"].to_numpy() > 0

    modality_names = [spec.name for spec in INPUT_SPECS]
    presence = np.column_stack(
        [frame[f"{name}_bin_ratio"].to_numpy() > 0 for name in modality_names]
    ).astype(np.int64)
    statistics.modality_counts += presence.sum(axis=0)
    statistics.joint_counts += presence.T @ presence
    statistics.total_bins += len(frame)


def write_manifest(path: Path, modalities: list[Modality]) -> None:
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


def create_feature_dataset(
    handle: h5py.File,
    feature_count: int,
    chunk_rows: int,
    bin_size: int,
    columns_file: str,
) -> h5py.Dataset:
    dataset = handle.create_dataset(
        "matrix",
        shape=(0, feature_count),
        maxshape=(None, feature_count),
        chunks=(chunk_rows, feature_count),
        dtype=np.float32,
        compression="gzip",
        compression_opts=1,
        shuffle=True,
    )
    dataset.attrs.update(
        bin_size=bin_size,
        feature_count=feature_count,
        columns_file=columns_file,
    )
    return dataset


def append_training_rows(
    nonzero: pd.DataFrame,
    feature_columns: list[str],
    feature_dataset: h5py.Dataset,
    bin_handle,
    first_index: int,
) -> int:
    block = nonzero[feature_columns].to_numpy(dtype=np.float32, copy=False)
    next_index = first_index + len(block)
    feature_dataset.resize(next_index, axis=0)
    feature_dataset[first_index:next_index] = block

    regions = nonzero[["chrom", "bin_start", "bin_end", "bin_id"]].rename(
        columns={"bin_start": "start", "bin_end": "end", "bin_id": "name"}
    )
    regions["index"] = np.arange(first_index, next_index, dtype=np.int64)
    regions.to_csv(
        bin_handle,
        sep="\t",
        index=False,
        header=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    return next_index


def write_dataset_files(
    args: argparse.Namespace,
    chromosomes: dict[str, int],
    modalities: list[Modality],
    paths: OutputPaths,
    feature_columns: list[str],
) -> RunStatistics:
    statistics = RunStatistics(len(modalities))
    first_main = True
    first_nonzero = True

    with (
        gzip.open(paths.main_table, "wt", newline="", compresslevel=1) as main_handle,
        gzip.open(
            paths.nonzero_table, "wt", newline="", compresslevel=1
        ) as nonzero_handle,
        paths.regions.open("w", encoding="utf-8", newline="") as regions_handle,
        h5py.File(paths.feature_matrix, "w") as h5_handle,
    ):
        regions_handle.write("chrom\tstart\tend\tname\tindex\n")
        feature_dataset = create_feature_dataset(
            h5_handle,
            len(feature_columns),
            args.h5_chunk_rows,
            args.bin_size,
            paths.feature_columns.name,
        )

        # One chromosome is materialized at a time to bound peak memory use.
        for chrom, chrom_size in chromosomes.items():
            frame, summary = build_chromosome_bins(
                chrom, chrom_size, args.bin_size, modalities
            )
            update_statistics(statistics, frame, summary)
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
                statistics.training_rows = append_training_rows(
                    nonzero,
                    feature_columns,
                    feature_dataset,
                    regions_handle,
                    statistics.training_rows,
                )

            print(
                f"{chrom}: {len(frame):,} full bins, "
                f"{len(nonzero):,} bins with at least one non-zero value"
            )

        feature_dataset.attrs["row_count"] = statistics.training_rows
    return statistics


def write_validation_summary(path: Path, rows: list[dict[str, int | float]]) -> None:
    validation = pd.DataFrame(rows)
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
    validation.to_csv(path, sep="\t", index=False, float_format="%.2f")


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
        enrichment, index=modality_names, columns=modality_names
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


def report_outputs(paths: OutputPaths) -> None:
    print(f"Main table: {paths.main_table}")
    print(f"Non-all-zero table: {paths.nonzero_table}")
    print(f"Validation summary: {paths.root / 'validation_summary.tsv'}")
    print(f"Column manifest: {paths.root / 'column_manifest.tsv'}")
    print(f"Pretraining bins: {paths.regions}")
    print(f"Pretraining features: {paths.feature_matrix}")
    print(f"Feature columns: {paths.feature_columns}")
    print(f"Chromosome heatmap: {paths.root / 'chromosome_nonzero_heatmap.png'}")
    print(
        "Co-occurrence enrichment: "
        f"{paths.root / 'modality_cooccurrence_enrichment_heatmap.png'}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    data_dir = args.data_dir.resolve()
    chromosomes = select_chromosomes(args.chrom_selection)
    paths = build_output_paths(args.output_dir, args.bin_size)
    scaling_lookup = read_scaling_factors(args.scaling_factors, data_dir)
    modalities = load_modalities(data_dir, scaling_lookup, set(chromosomes))

    feature_columns = training_feature_names(modalities)
    if len(feature_columns) != EXPECTED_TRAINING_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAINING_FEATURES} training features, "
            f"found {len(feature_columns)}"
        )
    pd.DataFrame(
        {"index": np.arange(len(feature_columns)), "name": feature_columns}
    ).to_csv(paths.feature_columns, sep="\t", index=False)

    statistics = write_dataset_files(
        args, chromosomes, modalities, paths, feature_columns
    )
    write_validation_summary(
        paths.root / "validation_summary.tsv", statistics.validation_rows
    )
    write_manifest(paths.root / "column_manifest.tsv", modalities)
    create_research_outputs(
        statistics.chromosome_nonzero,
        [modality.name for modality in modalities],
        statistics.modality_counts,
        statistics.joint_counts,
        statistics.total_bins,
        paths.root,
    )
    report_outputs(paths)


if __name__ == "__main__":
    main()
