#!/usr/bin/env python3
"""Build factorized G4-to-CRE perturbation data in one HDF5 file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    from .preprocess_pretrain_bin import (
        INPUT_FILES, MM10_CHROM_SIZES, calculate_modality,
        find_scaling_factor_file, intersect_intervals, interval_bases,
        load_modality, map_records_to_regions, read_scaling_factors,
    )
except ImportError:
    from preprocess_pretrain_bin import (
        INPUT_FILES, MM10_CHROM_SIZES, calculate_modality,
        find_scaling_factor_file, intersect_intervals, interval_bases,
        load_modality, map_records_to_regions, read_scaling_factors,
    )

PRETRAIN_MODALITIES = [
    name for name, _, _ in INPUT_FILES if name != "PolIIS5P"
]
CRE_INPUT_MODALITIES = [name for name in PRETRAIN_MODALITIES if name != "G4"]
TARGET_MODALITIES = [name for name, _, _ in INPUT_FILES if name != "G4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--perturbed-dir", type=Path, required=True)
    parser.add_argument("--hic-interactions", type=Path, required=True)
    parser.add_argument(
        "--pretrain-columns",
        type=Path,
        default=Path(
            "data/preprocessed/pretrain_data/bin_1000bp/pretrain_input/"
            "mm10_1000_feature_columns.tsv"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/preprocessed/downstream_data/G4RegCRE/G4_1000bp_Reg_CRE_ATACPeak/training_data/g4_reg_cre_dataset.h5"),
    )
    parser.add_argument("--g4-window-size", type=int, default=1000)
    parser.add_argument(
        "--cre-method", choices=["ATACPeak", "ATAC_NFR"], default="ATACPeak",
        help="Use complete ATAC peaks or exact ATAC-NFR intersections as CRE candidates.",
    )
    parser.add_argument(
        "--perturbation-mode", choices=["global", "local"], default="global"
    )
    parser.add_argument(
        "--targeted-g4-bed", type=Path, default=None,
        help="Required for local perturbation; BED intervals identify targeted G4 peaks.",
    )
    parser.add_argument("--control-scaling-factors", type=Path, default=None)
    parser.add_argument("--perturbed-scaling-factors", type=Path, default=None)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    return parser.parse_args()


def load_modalities(data_dir: Path, scaling_path: Path | None) -> dict[str, object]:
    factors = read_scaling_factors(find_scaling_factor_file(data_dir, scaling_path))
    allowed = set(MM10_CHROM_SIZES)
    return {
        name: load_modality(name, data_dir / path, prefix, factors, allowed)
        for name, path, prefix in INPUT_FILES
    }


def intersect_peaks(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return exact ATAC-NFR intersection intervals without merge or extension."""
    rows: list[tuple[str, int, int]] = []
    for chrom in MM10_CHROM_SIZES:
        a = left.loc[left.chrom == chrom, ["start", "end"]].to_numpy(np.int64)
        b = right.loc[right.chrom == chrom, ["start", "end"]].to_numpy(np.int64)
        j = 0
        for a_start, a_end in a:
            while j < len(b) and b[j, 1] <= a_start:
                j += 1
            k = j
            while k < len(b) and b[k, 0] < a_end:
                start, end = max(a_start, b[k, 0]), min(a_end, b[k, 1])
                if end > start:
                    rows.append((chrom, int(start), int(end)))
                k += 1
    frame = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    frame = frame.drop_duplicates().sort_values(["chrom", "start", "end"])
    frame = frame.reset_index(drop=True)
    frame["cre_id"] = [f"CRE_{i:07d}" for i in range(len(frame))]
    return frame



def select_cre_candidates(modalities: dict[str, object], method: str) -> pd.DataFrame:
    if method == "ATAC_NFR":
        return intersect_peaks(modalities["ATAC"].peaks, modalities["NFR"].peaks)
    frame = modalities["ATAC"].peaks[["chrom", "start", "end"]].drop_duplicates()
    frame = frame.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    frame["cre_id"] = [f"CRE_{index:07d}" for index in range(len(frame))]
    return frame


def centered_g4_windows(peaks: pd.DataFrame, size: int) -> pd.DataFrame:
    if size <= 0:
        raise ValueError("--g4-window-size must be positive")
    rows = []
    peaks = peaks.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    for peak in peaks.itertuples(index=False):
        center = (int(peak.start) + int(peak.end)) // 2
        start = center - size // 2
        end = start + size
        chrom_size = MM10_CHROM_SIZES[peak.chrom]
        if start < 0:
            start, end = 0, size
        if end > chrom_size:
            start, end = chrom_size - size, chrom_size
        rows.append({
            "chrom": peak.chrom,
            "start": start,
            "end": end,
            "g4_start": int(peak.start),
            "g4_end": int(peak.end),
            "g4_original_peak": float(peak.value),
            "g4_peak_longer_than_window": int(peak.end - peak.start > size),
        })
    frame = pd.DataFrame(rows)
    frame["g4_id"] = [f"G4_{i:07d}" for i in range(len(frame))]
    return frame


def read_pretrain_features(path: Path) -> list[str]:
    table = pd.read_csv(path, sep="\t")
    names = table.loc[~table["name"].str.contains("PolIIS5P"), "name"].tolist()
    if len(names) != 288:
        raise ValueError(f"Expected 288 non-PolII features, found {len(names)}")
    return names


def overlap_name(name: str) -> tuple[str, str] | None:
    if "_overlapping_" not in name:
        return None
    numerator, denominator = name.split("_overlapping_", 1)
    return numerator, denominator


def build_feature_matrix(
    regions: pd.DataFrame,
    modalities: dict[str, object],
    feature_names: list[str],
    allowed_modalities: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate one shared feature schema and a true observed-column mask."""
    allowed = set(allowed_modalities)
    states: dict[str, list[object]] = {}
    for name in allowed_modalities:
        mapped = map_records_to_regions(regions, modalities[name].peaks, "value")
        states[name] = [
            calculate_modality(records, int(region.start), int(region.end))
            for records, region in zip(mapped, regions.itertuples(index=False))
        ]

    matrix = np.zeros((len(regions), len(feature_names)), dtype=np.float32)
    observed = np.zeros(len(feature_names), dtype=bool)
    for column, feature in enumerate(feature_names):
        pair = overlap_name(feature)
        if pair:
            numerator, denominator = pair
            if numerator not in allowed or denominator not in allowed:
                continue
            observed[column] = True
            for row in range(len(regions)):
                denominator_bases = states[denominator][row].covered_bases
                if denominator_bases:
                    shared = intersect_intervals(
                        states[numerator][row].intervals,
                        states[denominator][row].intervals,
                    )
                    matrix[row, column] = interval_bases(shared) / denominator_bases
            continue

        modality, form = feature.rsplit("_", 1)
        if form == "ratio":
            modality, form = feature.rsplit("_", 2)[0], "_".join(feature.rsplit("_", 2)[1:])
        if modality not in allowed or form not in {"peak", "self_ratio", "bin_ratio"}:
            continue
        observed[column] = True
        matrix[:, column] = [getattr(state, form) for state in states[modality]]
    return np.round(matrix, 2), observed


def target_feature_names() -> list[str]:
    names = []
    for modality in TARGET_MODALITIES:
        names.extend([f"{modality}_peak", f"{modality}_self_ratio", f"{modality}_bin_ratio"])
        for other in TARGET_MODALITIES:
            if other != modality:
                names.append(f"{other}_overlapping_{modality}")
    return names


def read_bed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", comment="#", header=None, usecols=[0, 1, 2])
    frame.columns = ["chrom", "start", "end"]
    return frame.loc[frame.chrom.isin(MM10_CHROM_SIZES)].sort_values(
        ["chrom", "start", "end"]
    )


def interval_lookup(regions: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result = {}
    for chrom, group in regions.groupby("chrom", sort=False):
        result[chrom] = (
            group.start.to_numpy(np.int64), group.end.to_numpy(np.int64),
            group.index.to_numpy(np.int64),
        )
    return result


def hits(lookup, chrom: str, start: int, end: int) -> np.ndarray:
    if chrom not in lookup:
        return np.empty(0, dtype=np.int64)
    starts, ends, indices = lookup[chrom]
    stop = np.searchsorted(starts, end, side="left")
    return indices[:stop][ends[:stop] > start]


def sparse_relations(
    g4: pd.DataFrame, cre: pd.DataFrame, hic_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Store overlap and Hi-C pairs; all remaining Cartesian pairs are implicit proximal."""
    g_count = len(g4)
    g_lookup = interval_lookup(
        g4[["chrom", "g4_start", "g4_end"]].rename(
            columns={"g4_start": "start", "g4_end": "end"}
        )
    )
    c_lookup = interval_lookup(cre)
    overlap_keys = set()
    for c in cre.itertuples():
        for gi in hits(g_lookup, c.chrom, int(c.start), int(c.end)):
            overlap_keys.add(int(c.Index) * g_count + int(gi))

    loops = pd.read_csv(hic_path, sep="\t", comment="#", header=None)
    if loops.shape[1] < 6:
        raise ValueError("Hi-C interaction file must be BEDPE with at least 6 columns")
    far_scores: dict[int, float] = {}
    for row in loops.itertuples(index=False, name=None):
        c1, s1, e1, c2, s2, e2 = str(row[0]), int(row[1]), int(row[2]), str(row[3]), int(row[4]), int(row[5])
        try:
            score = float(row[6]) if len(row) > 6 and pd.notna(row[6]) else 1.0
        except (TypeError, ValueError):
            score = 1.0
        for g_chrom, gs, ge, c_chrom, cs, ce in (
            (c1, s1, e1, c2, s2, e2), (c2, s2, e2, c1, s1, e1)
        ):
            for gi in hits(g_lookup, g_chrom, gs, ge):
                for ci in hits(c_lookup, c_chrom, cs, ce):
                    key = int(ci) * g_count + int(gi)
                    if key not in overlap_keys:
                        far_scores[key] = max(score, far_scores.get(key, -np.inf))
    overlap = np.asarray(sorted(overlap_keys), dtype=np.int64)
    far = np.asarray(sorted(far_scores), dtype=np.int64)
    scores = np.asarray([far_scores[key] for key in far], dtype=np.float32)
    return overlap, far, scores


def perturbation_matrix(
    windows: pd.DataFrame,
    control_values: np.ndarray,
    perturbed_values: np.ndarray,
    feature_names: list[str],
    mode: str,
    targeted_bed: Path | None,
) -> tuple[np.ndarray, list[str]]:
    columns = [feature_names.index(f"G4_{form}") for form in ("peak", "self_ratio", "bin_ratio")]
    targeted = np.ones(len(windows), dtype=np.float32)
    if mode == "local":
        if targeted_bed is None:
            raise ValueError("--targeted-g4-bed is required for local perturbation")
        lookup = interval_lookup(read_bed(targeted_bed).reset_index(drop=True))
        targeted = np.asarray([
            float(len(hits(lookup, row.chrom, row.g4_start, row.g4_end)) > 0)
            for row in windows.itertuples(index=False)
        ], dtype=np.float32)
    delta = perturbed_values[:, columns] - control_values[:, columns]
    return np.column_stack([targeted, delta]).astype(np.float32), [
        "is_targeted", "delta_g4_peak", "delta_g4_self_ratio", "delta_g4_bin_ratio"
    ]


def string_dataset(handle: h5py.Group, name: str, values) -> None:
    handle.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype("utf-8"))


def write_h5(
    path: Path, g4: pd.DataFrame, cre: pd.DataFrame,
    g4_control: np.ndarray, cre_control: np.ndarray, cre_target: np.ndarray,
    g4_observed: np.ndarray, cre_observed: np.ndarray,
    feature_names: list[str], target_names: list[str], perturbation: np.ndarray,
    perturbation_names: list[str], overlap: np.ndarray, far: np.ndarray,
    far_scores: np.ndarray, args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks_g = (min(args.chunk_rows, len(g4)), g4_control.shape[1])
    chunks_c = (min(args.chunk_rows, len(cre)), cre_control.shape[1])
    with h5py.File(path, "w") as h5:
        gg, cg, rg = h5.create_group("g4"), h5.create_group("cre"), h5.create_group("relations")
        gg.create_dataset("control_features", data=g4_control, chunks=chunks_g, compression="gzip", compression_opts=1)
        gg.create_dataset("perturbation", data=perturbation, chunks=(chunks_g[0], perturbation.shape[1]), compression="gzip", compression_opts=1)
        gg.create_dataset("observed_features", data=g4_observed)
        for name in ("start", "end", "g4_start", "g4_end"):
            gg.create_dataset(name, data=g4[name].to_numpy(np.int64))
        string_dataset(gg, "id", g4.g4_id)
        string_dataset(gg, "chrom", g4.chrom)

        cg.create_dataset("control_features", data=cre_control, chunks=chunks_c, compression="gzip", compression_opts=1)
        cg.create_dataset("perturbed_targets", data=cre_target, chunks=(chunks_c[0], cre_target.shape[1]), compression="gzip", compression_opts=1)
        cg.create_dataset("observed_features", data=cre_observed)
        cg.create_dataset("start", data=cre.start.to_numpy(np.int64))
        cg.create_dataset("end", data=cre.end.to_numpy(np.int64))
        string_dataset(cg, "id", cre.cre_id)
        string_dataset(cg, "chrom", cre.chrom)

        rg.create_dataset("overlap_pair_key", data=overlap, compression="gzip", compression_opts=1)
        rg.create_dataset("far_pair_key", data=far, compression="gzip", compression_opts=1)
        rg.create_dataset("far_hic_score", data=far_scores, compression="gzip", compression_opts=1)
        string_dataset(h5, "pretrain_feature_names", feature_names)
        string_dataset(h5, "target_feature_names", target_names)
        string_dataset(h5, "perturbation_feature_names", perturbation_names)
        h5.attrs.update(
            assembly="mm10", g4_window_size=args.g4_window_size,
            perturbation_mode=args.perturbation_mode,
            relation_encoding="[overlap,proximal_no_HiC,far_HiC]",
            pair_layout="implicit Cartesian CRE x G4; overlap and far pairs stored sparsely",
            pair_count=int(len(g4) * len(cre)),
            parameters=json.dumps(vars(args), default=str),
        )


def main() -> None:
    args = parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    control = load_modalities(args.control_dir.resolve(), args.control_scaling_factors)
    perturbed = load_modalities(args.perturbed_dir.resolve(), args.perturbed_scaling_factors)
    features = read_pretrain_features(args.pretrain_columns.resolve())
    targets = target_feature_names()

    cre = select_cre_candidates(control, args.cre_method)
    g4 = centered_g4_windows(control["G4"].peaks, args.g4_window_size)
    if cre.empty or g4.empty:
        raise RuntimeError(f"No candidates: G4={len(g4)}, CRE={len(cre)}")
    g4_control, g4_observed = build_feature_matrix(g4, control, features, PRETRAIN_MODALITIES)
    g4_perturbed, _ = build_feature_matrix(g4, perturbed, features, PRETRAIN_MODALITIES)
    cre_control, cre_observed = build_feature_matrix(cre, control, features, CRE_INPUT_MODALITIES)
    cre_target, _ = build_feature_matrix(cre, perturbed, targets, TARGET_MODALITIES)
    perturbation, perturbation_names = perturbation_matrix(
        g4, g4_control, g4_perturbed, features,
        args.perturbation_mode, args.targeted_g4_bed,
    )
    overlap, far, far_scores = sparse_relations(g4, cre, args.hic_interactions.resolve())
    write_h5(
        args.output.resolve(), g4, cre, g4_control, cre_control, cre_target,
        g4_observed, cre_observed, features, targets, perturbation,
        perturbation_names, overlap, far, far_scores, args,
    )
    summary = args.output.parent.parent / "g4_reg_cre_summary.tsv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["item", "value"])
        writer.writerows([
            ("g4_count", len(g4)), ("cre_count", len(cre)),
            ("implicit_pair_count", len(g4) * len(cre)),
            ("overlap_pair_count", len(overlap)), ("far_hic_pair_count", len(far)),
            ("proximal_implicit_pair_count", len(g4) * len(cre) - len(overlap) - len(far)),
            ("g4_feature_count", int(g4_observed.sum())),
            ("cre_feature_count", int(cre_observed.sum())),
            ("target_feature_count", len(targets)),
        ])
    print(f"Wrote {args.output.resolve()}")
    print(f"G4={len(g4):,}, CRE={len(cre):,}, implicit pairs={len(g4) * len(cre):,}")


if __name__ == "__main__":
    main()
