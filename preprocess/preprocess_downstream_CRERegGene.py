#!/usr/bin/env python3
"""Build G4-centered candidates and gene-level CRE-to-gene training data."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import urllib.request
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .preprocess_pretrain_bin import (
        INPUT_FILES, MM10_CHROM_SIZES, RegionModality, calculate_modality,
        find_scaling_factor_file, intersect_intervals, interval_bases,
        load_modality, map_records_to_regions, merge_clipped,
        read_scaling_factors, training_feature_names,
    )
except ImportError:
    from preprocess_pretrain_bin import (
        INPUT_FILES, MM10_CHROM_SIZES, RegionModality, calculate_modality,
        find_scaling_factor_file, intersect_intervals, interval_bases,
        load_modality, map_records_to_regions, merge_clipped,
        read_scaling_factors, training_feature_names,
    )

ENSEMBL_GTF_URL = (
    "https://ftp.ensembl.org/pub/release-102/gtf/mus_musculus/"
    "Mus_musculus.GRCm38.102.gtf.gz"
)
LABEL_CODES = {"Promoter": 0, "Enhancer": 1, "TAD": 2, "Others": 3}
LABEL_COLORS = {
    "Promoter": "#D95F5F", "Enhancer": "#E6A84A",
    "TAD": "#4C78A8", "Others": "#8C8C8C",
}
LEGACY_OUTPUTS = [
    "mm10_g4_1kb_configuration.tsv",
    "mm10_g4_1kb_feature_columns.tsv",
    "mm10_g4_1kb_features.h5",
    "mm10_g4_1kb_label_encoding.tsv",
    "mm10_g4_1kb_label_summary.tsv",
    "mm10_g4_1kb_modality_manifest.tsv",
    "mm10_g4_1kb_multimodal.tsv.gz",
    "mm10_g4_1kb_regions_and_labels.tsv",
    "mm10_g4_1kb_rule_label_source_features.tsv",
]


MODEL_NAMES = ["ABC", "CIA", "ENCODE-rE2G", "EpiMap", "GraphReg", "Enformer"]
MODEL_STATUS = {
    "ABC": "ABC-compatible: measured ATAC/H3K27ac and distance contact",
    "CIA": "CIA-inspired proxy: measured CTCF and public cardiac Hi-C boundaries",
    "ENCODE-rE2G": "rE2G-inspired proxy: ABC, chromatin, expression, distance and neighborhood",
    "EpiMap": "EpiMap-inspired proxy: single-condition enhancer state and expression compatibility",
    "GraphReg": "GraphReg-inspired proxy: boundary-constrained one-step regulatory graph",
    "Enformer": "Enformer-inspired proxy: local G4 sequence-propensity features; no Enformer checkpoint",
}
BOUNDARY_URL = (
    "https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/"
    "75a0db05-a917-45ca-899d-ccfe6a4685ba/4DNFILRQWE9Q.bed.gz"
)
ROLE_CODES = {"Promoter": 0, "Enhancer": 1, "TAD_boundary": 2}
ROLE_PREFIXES = {
    "Promoter": "promoter", "Enhancer": "enhancer", "TAD_boundary": "boundary"
}




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_multimodals_data"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/preprocessed/downstream_data/CRERegGene/CRE_500kb_Reg_Gene_mm10"),
    )
    parser.add_argument("--scaling-factors", type=Path, default=None)
    parser.add_argument(
        "--ensembl-gtf", type=Path,
        default=Path("data/annotation/Mus_musculus.GRCm38.102.gtf.gz"),
    )
    parser.add_argument("--ensembl-url", default=ENSEMBL_GTF_URL)
    parser.add_argument("--g4-window-size", type=int, default=1000)
    parser.add_argument("--promoter-upstream", type=int, default=2000)
    parser.add_argument("--promoter-downstream", type=int, default=1000)
    parser.add_argument(
        "--ct-tadb-boundaries", type=Path,
        default=Path("data/reference/ct_tadb_mm10_boundaries.bed.gz"),
    )
    parser.add_argument("--ct-tadb-threshold", type=float, default=0.5)
    parser.add_argument("--rna", type=Path, default=Path("CPC_RNASeq.tsv"))
    parser.add_argument("--min-tpm", type=float, default=1.0)
    parser.add_argument("--abc-threshold", type=float, default=0.02)
    parser.add_argument("--abc-max-distance", type=int, default=500_000)
    parser.add_argument("--contact-min-distance", type=int, default=5_000)
    parser.add_argument("--contact-gamma", type=float, default=0.87)
    parser.add_argument("--boundary-max-distance", type=int, default=1_000_000)
    parser.add_argument(
        "--hic-boundaries", type=Path,
        default=Path("data/reference/4dn_e12_5_cardiomyocyte_boundaries_mm10.bed.gz"),
    )
    parser.add_argument("--top-genes-per-model", type=int, default=3)
    parser.add_argument("--h5-chunk-rows", type=int, default=2048)
    parser.add_argument(
        "--rebuild-g4-regions", action="store_true",
        help="Rebuild G4-centered position files; requires official CT-TADB output.",
    )
    return parser.parse_args()


def window_prefix(window_size: int) -> str:
    size = f"{window_size // 1000}kb" if window_size % 1000 == 0 else f"{window_size}bp"
    return f"g4_position_{size}"


def ensure_ensembl_gtf(path: Path, url: str) -> Path:
    path = path.resolve()
    if path.exists():
        return path
    candidates = sorted(
        path.parent.glob("Mus_musculus.GRCm38.102*.gtf.gz"),
        key=lambda candidate: candidate.stat().st_size,
        reverse=True,
    )
    if candidates:
        print(f"Using existing Ensembl GRCm38 annotation: {candidates[0]}")
        return candidates[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading Ensembl GRCm38 gene annotation: {url}")
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)
    return path


def parse_gtf_attributes(text: str) -> dict[str, str]:
    return dict(re.findall(r'(\S+) "([^"]+)"', text))


def read_ensembl_promoters(
    path: Path,
    upstream: int,
    downstream: int,
) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[tuple[str, int, int, str]] = []
    allowed = set(MM10_CHROM_SIZES)
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            chrom = fields[0] if fields[0].startswith("chr") else f"chr{fields[0]}"
            if chrom not in allowed:
                continue
            start_1based, end_1based = int(fields[3]), int(fields[4])
            strand = fields[6]
            attributes = parse_gtf_attributes(fields[8])
            gene_id = attributes.get("gene_id", "-")
            gene_name = attributes.get("gene_name", gene_id)
            if strand == "+":
                tss = start_1based - 1
                start, end = tss - upstream, tss + downstream
            else:
                tss = end_1based - 1
                start, end = tss - downstream, tss + upstream
            start = max(0, start)
            end = min(MM10_CHROM_SIZES[chrom], end)
            if end > start:
                name = f"{gene_id}|{gene_name}|{strand}|TSS:{tss}"
                rows.append((chrom, start, end, name))
    if not rows:
        raise ValueError(f"No mm10/GRCm38 gene promoters found in {path}")
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "name"]).sort_values(
        ["chrom", "start", "end"], ignore_index=True
    )


def read_ct_tadb_boundaries(path: Path, threshold: float) -> pd.DataFrame:
    """Read boundary probabilities produced by the official CT-TADB model."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"CT-TADB prediction file not found: {path}\n"
            "Generate it with the official CT-TADB model using 10-kb mm10 DNA "
            "sequences and all 12 required epigenomic tracks. The project currently "
            "contains only CTCF, H3K4me1, H3K4me3, H3K27ac and H3K27me3; do not "
            "replace missing CT-TADB tracks with zeros."
        )
    boundaries = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "value"],
        compression="infer",
    )
    boundaries["start"] = pd.to_numeric(boundaries["start"], errors="raise")
    boundaries["end"] = pd.to_numeric(boundaries["end"], errors="raise")
    boundaries["value"] = pd.to_numeric(boundaries["value"], errors="raise")
    boundaries = boundaries[
        boundaries["chrom"].isin(MM10_CHROM_SIZES)
        & (boundaries["start"] >= 0)
        & (boundaries["end"] > boundaries["start"])
        & boundaries["value"].between(0, 1)
        & (boundaries["value"] >= threshold)
    ].copy()
    if boundaries.empty:
        raise ValueError(
            f"No valid CT-TADB boundaries remain at probability >= {threshold}"
        )
    boundaries[["start", "end"]] = boundaries[["start", "end"]].astype(int)
    return boundaries.sort_values(["chrom", "start", "end"], ignore_index=True)


def build_g4_windows(g4_peaks: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Extend each G4 toward ``window_size`` without entering adjacent G4 peaks."""
    rows: list[dict[str, object]] = []
    peaks = g4_peaks.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    for chrom, group in peaks.groupby("chrom", sort=False):
        group = group.reset_index(drop=True)
        previous_max_end = np.maximum.accumulate(group["end"].to_numpy())
        for local_index, peak in enumerate(group.itertuples(index=False)):
            peak_start, peak_end = int(peak.start), int(peak.end)
            previous_end = int(previous_max_end[local_index - 1]) if local_index else 0
            next_start = (
                int(group.iloc[local_index + 1]["start"])
                if local_index + 1 < len(group)
                else MM10_CHROM_SIZES[chrom]
            )
            left_room = max(0, peak_start - previous_end)
            right_room = max(0, next_start - peak_end)
            extra = max(0, window_size - (peak_end - peak_start))
            left_goal, right_goal = extra // 2, extra - extra // 2
            left_extension = min(left_goal, left_room)
            right_extension = min(right_goal, right_room)

            left_deficit = left_goal - left_extension
            right_extension += min(left_deficit, right_room - right_extension)
            right_deficit = max(0, right_goal - right_extension)
            left_extension += min(right_deficit, left_room - left_extension)

            start = peak_start - left_extension
            end = peak_end + right_extension
            rows.append({
                "chrom": chrom,
                "start": start,
                "end": end,
                "g4_id": f"G4_{len(rows):07d}",
                "g4_start": peak_start,
                "g4_end": peak_end,
                "g4_length": peak_end - peak_start,
                "window_length": end - start,
                "left_extension": left_extension,
                "right_extension": right_extension,
                "target_size_reached": int(end - start >= window_size),
                "neighbor_overlap_conflict": int(
                    previous_end > peak_start or next_start < peak_end
                ),
                "anchor_g4_peak": float(peak.value),
                "g4_peak_exceeds_window": int(peak_end - peak_start > window_size),
            })
    regions = pd.DataFrame(rows)
    regions["window_id"] = (
        regions["g4_id"]
        + "|"
        + regions["chrom"]
        + ":"
        + regions["start"].astype(str)
        + "-"
        + regions["end"].astype(str)
    )
    return regions


def plot_dataset_overview(
    labels: pd.DataFrame,
    matrix: np.ndarray,
    feature_names: list[str],
    modalities: list[object],
    target_size: int,
    output: Path,
) -> None:
    """Plot class balance, window QC, chromosome composition and modality coverage."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    class_order = list(LABEL_CODES)
    counts = labels["position_label"].value_counts().reindex(class_order, fill_value=0)
    bars = axes[0, 0].bar(
        class_order, counts, color=[LABEL_COLORS[name] for name in class_order]
    )
    axes[0, 0].bar_label(
        bars,
        labels=[f"{count:,}\n({count / len(labels):.1%})" for count in counts],
        padding=3,
    )
    axes[0, 0].set(title="A  Position labels", ylabel="G4 regions")
    axes[0, 0].margins(y=0.15)
    axes[0, 0].spines[["top", "right"]].set_visible(False)

    lengths = labels["window_length"]
    size_counts = pd.Series({
        f"<{target_size}": int((lengths < target_size).sum()),
        f"={target_size}": int((lengths == target_size).sum()),
        f">{target_size}": int((lengths > target_size).sum()),
    })
    bars = axes[0, 1].bar(
        size_counts.index, size_counts, color=["#72A0C1", "#59A14F", "#B07AA1"]
    )
    axes[0, 1].bar_label(bars, labels=[f"{value:,}" for value in size_counts], padding=3)
    axes[0, 1].set(title="B  Adaptive window length", ylabel="G4 regions")
    axes[0, 1].margins(y=0.15)
    axes[0, 1].spines[["top", "right"]].set_visible(False)

    chromosome_order = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]
    chromosome_order = [name for name in chromosome_order if name in set(labels["chrom"])]
    composition = pd.crosstab(
        labels["chrom"], labels["position_label"], normalize="index"
    ).reindex(index=chromosome_order, columns=class_order, fill_value=0)
    image = axes[1, 0].imshow(composition, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    axes[1, 0].set(
        title="C  Label fraction by chromosome",
        xticks=np.arange(len(class_order)),
        xticklabels=class_order,
        yticks=np.arange(len(chromosome_order)),
        yticklabels=chromosome_order,
    )
    axes[1, 0].tick_params(axis="x", rotation=25)
    fig.colorbar(image, ax=axes[1, 0], label="Fraction", shrink=0.8)

    feature_index = {name: index for index, name in enumerate(feature_names)}
    names = [modality.name for modality in modalities]
    coverage = [
        np.mean(matrix[:, feature_index[f"{name}_peak"]] > 0) for name in names
    ]
    order = np.argsort(coverage)
    axes[1, 1].barh(
        np.asarray(names)[order], np.asarray(coverage)[order], color="#4E79A7"
    )
    axes[1, 1].set(
        title="D  Non-zero peak coverage", xlabel="Fraction of G4 regions", xlim=(0, 1)
    )
    axes[1, 1].spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def remove_legacy_outputs(output_dir: Path) -> None:
    for name in LEGACY_OUTPUTS:
        (output_dir / name).unlink(missing_ok=True)


def describe_column(name: str) -> tuple[str, str]:
    if name == "label":
        return "position", "class_label"
    if "_overlapping_" in name:
        return "pairwise", "overlap_ratio"
    modality, data_form = name.rsplit("_", 1)
    if name.endswith("_self_ratio"):
        modality, data_form = name.removesuffix("_self_ratio"), "self_ratio"
    elif name.endswith("_bin_ratio"):
        modality, data_form = name.removesuffix("_bin_ratio"), "bin_ratio"
    return modality, data_form




def read_genes(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, object]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            chrom = fields[0] if fields[0].startswith("chr") else f"chr{fields[0]}"
            if chrom not in MM10_CHROM_SIZES:
                continue
            attributes = parse_gtf_attributes(fields[8])
            gene_id = attributes.get("gene_id", "").split(".")[0]
            if not gene_id:
                continue
            start, end, strand = int(fields[3]) - 1, int(fields[4]), fields[6]
            tss = start if strand == "+" else end - 1
            promoter_start = max(0, tss - (2000 if strand == "+" else 1000))
            promoter_end = min(
                MM10_CHROM_SIZES[chrom], tss + (1000 if strand == "+" else 2000)
            )
            rows.append({
                "gene_id": gene_id,
                "gene_name": attributes.get("gene_name", gene_id),
                "gene_type": attributes.get(
                    "gene_biotype", attributes.get("gene_type", "unknown")
                ),
                "chrom": chrom,
                "gene_start": start,
                "gene_end": end,
                "strand": strand,
                "tss": tss,
                "promoter_start": promoter_start,
                "promoter_end": promoter_end,
            })
    genes = pd.DataFrame(rows).drop_duplicates("gene_id")
    return genes.sort_values(["chrom", "tss", "gene_id"]).reset_index(drop=True)


def add_expression(genes: pd.DataFrame, path: Path) -> pd.DataFrame:
    rna = pd.read_csv(path, sep="\t")
    rna["gene_id"] = rna["gene_id"].astype(str).str.split(".").str[0]
    values = rna.drop(columns="gene_id").apply(pd.to_numeric, errors="coerce")
    rna["rna_tpm"] = values.mean(axis=1).round(2)
    return genes.merge(rna[["gene_id", "rna_tpm"]], on="gene_id", how="left").fillna(
        {"rna_tpm": 0.0}
    )


def load_required_modalities(data_dir: Path, scaling_path: Path | None) -> dict[str, object]:
    factors = read_scaling_factors(find_scaling_factor_file(data_dir, scaling_path))
    required = {"ATAC", "H3K27ac", "PolIIS5P"}
    return {
        name: load_modality(
            name, data_dir / path, prefix, factors, set(MM10_CHROM_SIZES)
        )
        for name, path, prefix in INPUT_FILES
        if name in required
    }


def build_abc_elements(atac: object, h3k27ac: object) -> pd.DataFrame:
    """Use ATAC peaks with overlapping H3K27ac as the ABC candidate universe."""
    h3_maps = map_records_to_regions(atac.peaks, h3k27ac.peaks, "value")
    rows = []
    for index, peak in enumerate(atac.peaks.itertuples(index=False)):
        records = h3_maps[index]
        if not records or peak.value <= 0:
            continue
        h3_value = max(float(record[2]) for record in records)
        if h3_value <= 0:
            continue
        rows.append((peak.chrom, peak.start, peak.end, np.sqrt(peak.value * h3_value)))
    elements = pd.DataFrame(rows, columns=["chrom", "start", "end", "activity"])
    elements = elements.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    elements["element_id"] = np.arange(len(elements), dtype=np.int64)
    elements["center"] = (elements["start"] + elements["end"]) // 2
    return elements


def score_abc(
    enhancers: pd.DataFrame,
    genes: pd.DataFrame,
    elements: pd.DataFrame,
    max_distance: int,
    min_distance: int,
    gamma: float,
) -> pd.DataFrame:
    mapped = map_records_to_regions(enhancers, elements, "element_id")
    element_to_g4: dict[int, set[str]] = {}
    for row, records in zip(enhancers.itertuples(index=False), mapped):
        for _, _, element_id in records:
            element_to_g4.setdefault(int(element_id), set()).add(row.g4_id)

    best: dict[tuple[str, str], dict[str, object]] = {}
    for chrom, chrom_elements in elements.groupby("chrom", sort=False):
        chrom_elements = chrom_elements.reset_index(drop=True)
        centers = chrom_elements["center"].to_numpy(dtype=np.int64)
        activity = chrom_elements["activity"].to_numpy(dtype=np.float64)
        ids = chrom_elements["element_id"].to_numpy(dtype=np.int64)
        target_positions = np.asarray(
            [index for index, element_id in enumerate(ids) if element_id in element_to_g4],
            dtype=np.int64,
        )
        if not len(target_positions):
            continue
        for gene in genes.loc[genes["chrom"] == chrom].itertuples(index=False):
            left = np.searchsorted(centers, gene.tss - max_distance, side="left")
            right = np.searchsorted(centers, gene.tss + max_distance, side="right")
            if right <= left:
                continue
            distance = np.abs(centers[left:right] - gene.tss)
            contact = (np.maximum(distance, min_distance) / min_distance) ** (-gamma)
            contribution = activity[left:right] * contact
            denominator = contribution.sum()
            if denominator <= 0:
                continue
            first = np.searchsorted(target_positions, left, side="left")
            last = np.searchsorted(target_positions, right, side="left")
            for position in target_positions[first:last]:
                element_id = int(ids[position])
                score = float(contribution[position - left] / denominator)
                for g4_id in element_to_g4[element_id]:
                    key = (g4_id, gene.gene_id)
                    if key not in best or score > best[key]["abc_score"]:
                        best[key] = {
                            "g4_id": g4_id,
                            "gene_id": gene.gene_id,
                            "abc_score": score,
                            "enhancer_tss_distance": int(abs(centers[position] - gene.tss)),
                            "abc_element": (
                                f"{chrom}:{chrom_elements.iloc[position]['start']}-"
                                f"{chrom_elements.iloc[position]['end']}"
                            ),
                        }
    return pd.DataFrame(best.values())


def ensure_boundaries(path: Path) -> pd.DataFrame:
    """Load public E12.5 mouse cardiomyocyte Hi-C boundary calls."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BOUNDARY_URL, path)
    frame = pd.read_csv(
        path, sep="\t", header=None,
        names=["chrom", "start", "end", "strength_class", "strength"],
    )
    frame = frame.loc[frame["chrom"].isin(MM10_CHROM_SIZES)].copy()
    frame["center"] = (frame["start"] + frame["end"]) // 2
    return frame.sort_values(["chrom", "center"]).reset_index(drop=True)


def percentile(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average").fillna(0.0)


def add_boundary_features(
    evidence: pd.DataFrame, boundaries: pd.DataFrame
) -> pd.DataFrame:
    evidence = evidence.reset_index(drop=True)
    counts = np.zeros(len(evidence), dtype=np.int32)
    strengths = np.zeros(len(evidence), dtype=np.float64)
    nearest = np.full(len(evidence), np.inf)
    for chrom, indexes in evidence.groupby("chrom", sort=False).groups.items():
        chrom_boundaries = boundaries.loc[boundaries["chrom"] == chrom]
        positions = chrom_boundaries["center"].to_numpy(np.int64)
        scores = chrom_boundaries["strength"].to_numpy(np.float64)
        if not len(positions):
            continue
        subset = evidence.loc[indexes]
        g4 = subset["enhancer_center"].to_numpy(np.int64)
        tss = subset["tss"].to_numpy(np.int64)
        left = np.searchsorted(positions, np.minimum(g4, tss), side="right")
        right = np.searchsorted(positions, np.maximum(g4, tss), side="left")
        cumulative = np.r_[0.0, np.cumsum(np.maximum(scores, 0))]
        counts[indexes] = right - left
        strengths[indexes] = cumulative[right] - cumulative[left]
        insertion = np.searchsorted(positions, g4)
        before = np.maximum(insertion - 1, 0)
        after = np.minimum(insertion, len(positions) - 1)
        nearest[indexes] = np.minimum(abs(g4 - positions[before]), abs(g4 - positions[after]))
    evidence["boundaries_between"] = counts
    evidence["boundary_strength_between"] = strengths
    evidence["nearest_boundary_distance"] = nearest
    evidence["same_domain_proxy"] = np.exp(-strengths) / (1 + counts)
    return evidence


def add_six_model_scores(
    abc: pd.DataFrame,
    regions: pd.DataFrame,
    features: np.ndarray,
    feature_names: list[str],
    genes: pd.DataFrame,
    boundaries: pd.DataFrame,
) -> pd.DataFrame:
    """Score one common candidate set using six transparent evidence channels."""
    evidence = abc.rename(columns={
        "abc_score": "ABC_score",
        "enhancer_tss_distance": "distance",
    }).copy()
    evidence["chrom"] = evidence["abc_element"].str.split(":").str[0]
    interval = evidence["abc_element"].str.extract(r":(\d+)-(\d+)").astype(int)
    evidence["enhancer_center"] = (interval[0] + interval[1]) // 2
    evidence = evidence.merge(genes[["gene_id", "tss", "rna_tpm"]], on="gene_id", how="left")

    feature_index = {name: index for index, name in enumerate(feature_names)}
    wanted = [
        "G4_peak", "G4_bin_ratio", "ATAC_peak", "H3K27ac_peak",
        "H3K4me1_peak", "H3K4me3_peak", "CTCF_peak", "Rloop_peak",
    ]
    g4 = regions[["g4_id", "g4_row_index", "g4_start", "g4_end"]].copy()
    rows = g4["g4_row_index"].to_numpy(np.int64)
    for name in wanted:
        g4[name] = features[rows, feature_index[name]] if name in feature_index else 0.0
    evidence = evidence.merge(g4.drop(columns="g4_row_index"), on="g4_id", how="left")
    evidence["distance_contact"] = (
        np.maximum(evidence["distance"], 5_000) / 5_000
    ) ** -0.87
    evidence = add_boundary_features(evidence, boundaries)

    rank_columns = ["distance_contact", "rna_tpm", *wanted]
    for column in rank_columns:
        evidence[f"{column}_pct"] = percentile(evidence[column].clip(lower=0))
    evidence["ABC_norm"] = percentile(evidence["ABC_score"])
    activity = np.cbrt(
        np.maximum(evidence["ATAC_peak_pct"], 1e-6)
        * np.maximum(evidence["H3K27ac_peak_pct"], 1e-6)
        * np.maximum(evidence["H3K4me1_peak_pct"], 1e-6)
    )
    evidence["CIA_score"] = (
        activity * evidence["distance_contact_pct"]
        * (0.5 + 0.5 * evidence["CTCF_peak_pct"])
        * evidence["same_domain_proxy"]
    )
    neighborhood = evidence.groupby("gene_id")["ATAC_peak_pct"].transform("mean")
    logit = (
        1.8 * evidence["ABC_norm"]
        + evidence["distance_contact_pct"]
        + 0.8 * activity
        + 0.5 * evidence["rna_tpm_pct"]
        + 0.4 * evidence["H3K4me3_peak_pct"]
        + 0.3 * neighborhood - 2.4
    )
    evidence["ENCODE-rE2G_score"] = 1 / (1 + np.exp(-logit))
    evidence["EpiMap_score"] = (
        activity * np.sqrt(np.maximum(evidence["rna_tpm_pct"], 1e-6))
        * evidence["distance_contact_pct"]
    )
    message = activity * evidence["distance_contact_pct"] * evidence["same_domain_proxy"]
    gene_message = evidence.assign(_message=message).groupby("gene_id")["_message"].transform("sum")
    evidence["GraphReg_score"] = message / np.maximum(gene_message, 1e-12)
    sequence_proxy = np.sqrt(
        np.maximum(evidence["G4_peak_pct"], 0)
        * np.maximum(evidence["G4_bin_ratio_pct"], 0)
    )
    evidence["Enformer_score"] = (
        sequence_proxy * activity * evidence["distance_contact_pct"]
    )
    return evidence


def vote_six_models(
    evidence: pd.DataFrame, abc_threshold: float, top_n: int
) -> pd.DataFrame:
    """Keep strict six-way consensus and a separate exploratory majority flag."""
    for model in MODEL_NAMES:
        score = f"{model}_score"
        evidence[f"{model}_norm"] = evidence.groupby("g4_id")[score].rank(
            pct=True, method="average"
        )
        rank = evidence.groupby("g4_id")[score].rank(ascending=False, method="min")
        evidence[f"{model}_support"] = rank <= top_n
    evidence["ABC_support"] &= evidence["ABC_score"] >= abc_threshold
    support_columns = [f"{model}_support" for model in MODEL_NAMES]
    evidence["support_count"] = evidence[support_columns].sum(axis=1).astype(np.int8)
    evidence["models_supporting"] = evidence.apply(
        lambda row: ";".join(
            model for model in MODEL_NAMES if row[f"{model}_support"]
        ),
        axis=1,
    )
    evidence["strict_consensus"] = (
        evidence["support_count"] == len(MODEL_NAMES)
    ).astype(np.int8)
    evidence["majority_consensus"] = (evidence["support_count"] >= 4).astype(np.int8)
    evidence["consensus_score"] = evidence[
        [f"{model}_norm" for model in MODEL_NAMES]
    ].mean(axis=1)
    evidence["consensus_status"] = np.select(
        [evidence["strict_consensus"].eq(1), evidence["majority_consensus"].eq(1)],
        ["six_model_unanimous", "majority_4_of_6"],
        default="insufficient_support",
    )
    evidence["accepted"] = evidence["strict_consensus"]
    return evidence.sort_values(
        ["strict_consensus", "support_count", "consensus_score"], ascending=False
    )


def promoter_assignments(regions: pd.DataFrame, genes: pd.DataFrame) -> list[dict]:
    gene_tss = dict(zip(genes["gene_id"], genes["tss"]))
    rows = []
    for region in regions.loc[regions["position_label"] == "Promoter"].itertuples():
        for item in str(region.promoter_genes).split(";"):
            gene_id = item.split("|", 1)[0].split(".")[0]
            if gene_id in gene_tss:
                center = (region.start + region.end) // 2
                rows.append({
                    "gene_id": gene_id,
                    "g4_id": region.g4_id,
                    "g4_role": "Promoter",
                    "assignment_method": "expanded_G4_overlap_Ensembl_promoter",
                    "assignment_status": "direct",
                    "boundary_side": "-",
                    "assignment_distance": abs(center - gene_tss[gene_id]),
                    "assignment_score": region.promoter_overlap_ratio,
                    "models_supporting": "direct_overlap",
                })
    return rows


def boundary_assignments_public(
    regions: pd.DataFrame,
    genes: pd.DataFrame,
    boundaries: pd.DataFrame,
    max_distance: int,
) -> list[dict]:
    """Assign boundary G4s to genes flanking the nearest public Hi-C boundary."""
    rows = []
    tad = regions.loc[regions["position_label"] == "TAD"]
    for chrom, group in tad.groupby("chrom", sort=False):
        chrom_genes = genes.loc[genes["chrom"] == chrom].sort_values("tss").reset_index(drop=True)
        positions = boundaries.loc[boundaries["chrom"] == chrom, "center"].to_numpy(np.int64)
        if chrom_genes.empty or not len(positions):
            continue
        tss = chrom_genes["tss"].to_numpy(np.int64)
        for region in group.itertuples(index=False):
            center = (region.start + region.end) // 2
            insertion = np.searchsorted(positions, center)
            choices = {max(0, insertion - 1), min(len(positions) - 1, insertion)}
            boundary = int(positions[min(choices, key=lambda i: abs(positions[i] - center))])
            split = np.searchsorted(tss, boundary, side="right")
            for position, side in ((split - 1, "left"), (split, "right")):
                if not 0 <= position < len(chrom_genes):
                    continue
                gene = chrom_genes.iloc[position]
                distance = abs(int(gene.tss) - boundary)
                if distance <= max_distance:
                    rows.append({
                        "gene_id": gene.gene_id,
                        "g4_id": region.g4_id,
                        "g4_role": "TAD_boundary",
                        "assignment_method": "public_HiC_boundary_flanking_TSS",
                        "assignment_status": "public_reference_proxy",
                        "boundary_side": side,
                        "assignment_distance": distance,
                        "assignment_score": round(1 / (1 + distance / 1000), 4),
                        "models_supporting": "4DN_cardiac_HiC_boundary",
                    })
    return rows


def add_polii_targets(genes: pd.DataFrame, polii: object) -> pd.DataFrame:
    def summarize(start_column: str, end_column: str) -> np.ndarray:
        regions = genes[["chrom", start_column, end_column]].copy()
        regions["source_index"] = np.arange(len(genes))
        regions = regions.rename(columns={start_column: "start", end_column: "end"})
        regions = regions.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        maps = map_records_to_regions(regions, polii.peaks, "value")
        values = np.zeros((len(genes), 2), dtype=np.float64)
        for index, region in enumerate(regions.itertuples(index=False)):
            state = calculate_modality(maps[index], region.start, region.end)
            values[region.source_index] = state.peak, state.bin_ratio
        return values

    body = summarize("gene_start", "gene_end")
    promoter = summarize("promoter_start", "promoter_end")
    targets = np.column_stack([body[:, 0], body[:, 1], promoter[:, 0], promoter[:, 1]])
    target_frame = pd.DataFrame(
        np.round(targets, 2),
        columns=[
            "polii_gene_peak", "polii_gene_ratio",
            "polii_promoter_peak", "polii_promoter_ratio",
        ],
    )
    return pd.concat([genes.reset_index(drop=True), target_frame], axis=1)


def build_gene_summary(genes: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    assigned = genes.loc[genes["gene_id"].isin(assignments["gene_id"])].copy()
    assigned = assigned.sort_values(["chrom", "tss", "gene_id"]).reset_index(drop=True)
    assigned["gene_index"] = np.arange(len(assigned), dtype=np.int64)
    for role, prefix in ROLE_PREFIXES.items():
        grouped = assignments.loc[assignments["g4_role"] == role].groupby("gene_id")["g4_id"]
        assigned[f"{prefix}_g4_count"] = assigned["gene_id"].map(grouped.nunique()).fillna(0).astype(int)
        assigned[f"{prefix}_g4_ids"] = assigned["gene_id"].map(
            grouped.apply(lambda values: ";".join(sorted(set(values))))
        ).fillna("-")
    assigned["total_g4_count"] = assigned[
        [f"{prefix}_g4_count" for prefix in ROLE_PREFIXES.values()]
    ].sum(axis=1)
    return assigned


def write_h5(
    path: Path,
    summary: pd.DataFrame,
    assignments: pd.DataFrame,
    features: np.ndarray,
    feature_names: list[str],
    chunk_rows: int,
    model_status: dict[str, str],
    parameters: dict[str, object],
) -> None:
    gene_index = dict(zip(summary["gene_id"], summary["gene_index"]))
    ordered = assignments.assign(
        gene_index=assignments["gene_id"].map(gene_index),
        role_code=assignments["g4_role"].map(ROLE_CODES),
    ).sort_values(["gene_index", "role_code", "g4_row_index"])
    offsets = np.zeros(len(summary) + 1, dtype=np.int64)
    counts = ordered.groupby("gene_index").size()
    offsets[1:] = np.cumsum([counts.get(index, 0) for index in range(len(summary))])
    selected = features[ordered["g4_row_index"].to_numpy(dtype=np.int64)]
    target_columns = [
        "polii_gene_peak", "polii_gene_ratio",
        "polii_promoter_peak", "polii_promoter_ratio",
    ]
    strings = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "features", data=selected,
            chunks=(min(chunk_rows, len(selected)), selected.shape[1]),
            compression="gzip", compression_opts=1, shuffle=True,
        )
        handle.create_dataset("gene_offsets", data=offsets)
        handle.create_dataset("gene_index", data=ordered["gene_index"].to_numpy(np.int64))
        handle.create_dataset("g4_row_index", data=ordered["g4_row_index"].to_numpy(np.int64))
        handle.create_dataset("role", data=ordered["role_code"].to_numpy(np.int8))
        handle.create_dataset("polii_targets", data=summary[target_columns].to_numpy(np.float32))
        handle.create_dataset("rna_tpm", data=summary["rna_tpm"].to_numpy(np.float32))
        handle.create_dataset("gene_id", data=summary["gene_id"].to_numpy(), dtype=strings)
        handle.create_dataset(
            "feature_names", data=np.asarray(feature_names, dtype=object), dtype=strings
        )
        handle.attrs.update(
            assembly="GRCm38", feature_count=len(feature_names),
            gene_count=len(summary), assignment_count=len(ordered),
            role_encoding="Promoter=0;Enhancer=1;TAD_boundary=2",
            target_columns=";".join(target_columns),
            model_status=json.dumps(model_status, ensure_ascii=True),
            parameters=json.dumps(parameters, ensure_ascii=True),
            abc_implementation=(
                "activity=sqrt(scaled_ATAC_peak*scaled_H3K27ac_peak);"
                "contact=distance_powerlaw;normalized_across_candidate_elements"
            ),
        )


def write_statistics(
    output_dir: Path, evidence: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    support_columns = [f"{model}_support" for model in MODEL_NAMES]
    summary = pd.DataFrame({
        "model": MODEL_NAMES,
        "implementation": [MODEL_STATUS[model] for model in MODEL_NAMES],
        "supported_pairs": [int(evidence[column].sum()) for column in support_columns],
        "supported_g4": [evidence.loc[evidence[column], "g4_id"].nunique() for column in support_columns],
        "supported_genes": [evidence.loc[evidence[column], "gene_id"].nunique() for column in support_columns],
    })
    summary.to_csv(output_dir / "eg_model_summary.tsv", sep="\t", index=False)
    pair_sets = [set(map(tuple, evidence.loc[evidence[column], ["g4_id", "gene_id"]].to_numpy())) for column in support_columns]
    jaccard = np.zeros((6, 6))
    for i, left in enumerate(pair_sets):
        for j, right in enumerate(pair_sets):
            jaccard[i, j] = len(left & right) / max(len(left | right), 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    evidence["support_count"].value_counts().sort_index().reindex(range(7), fill_value=0).plot.bar(ax=axes[0, 0], color="#4C78A8")
    axes[0, 0].set(xlabel="Number of supporting methods", ylabel="E-G pairs", title="Consensus support")
    axes[0, 1].barh(MODEL_NAMES, summary["supported_pairs"], color=["#D1495B", "#2A9D8F", "#E9C46A", "#7A5195", "#4C78A8", "#F28E2B"])
    axes[0, 1].set(xlabel="Supported E-G pairs", title="Method-level support")
    image = axes[1, 0].imshow(jaccard, vmin=0, vmax=1, cmap="Reds")
    axes[1, 0].set_xticks(range(6), MODEL_NAMES, rotation=35, ha="right")
    axes[1, 0].set_yticks(range(6), MODEL_NAMES)
    axes[1, 0].set_title("Pairwise Jaccard agreement")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)
    colors = {"insufficient_support": "#B8B8B8", "majority_4_of_6": "#E9C46A", "six_model_unanimous": "#D1495B"}
    for label, color in colors.items():
        values = evidence.loc[evidence["consensus_status"] == label, "distance"]
        if len(values):
            axes[1, 1].hist(np.log10(values + 1), bins=35, alpha=0.65, label=label, color=color, density=True)
    axes[1, 1].set(xlabel="log10 enhancer-TSS distance + 1", ylabel="Density", title="Distance by consensus tier")
    axes[1, 1].legend(frameon=False, fontsize=8)
    strict = int(evidence["strict_consensus"].sum())
    fig.suptitle(f"Six-method enhancer-gene consensus | strict pairs: {strict:,} | assigned rows: {len(assignments):,}", fontsize=13)
    fig.savefig(output_dir / "eg_consensus_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)




def build_g4_regions(args: argparse.Namespace) -> None:
    if args.g4_window_size <= 0 or args.h5_chunk_rows <= 0:
        raise ValueError("window size and HDF5 chunk rows must be positive")
    if not 0.0 <= args.ct_tadb_threshold <= 1.0:
        raise ValueError("--ct-tadb-threshold must be between 0 and 1")

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    factors = read_scaling_factors(
        find_scaling_factor_file(data_dir, args.scaling_factors)
    )
    modalities = [
        load_modality(
            name,
            data_dir / path,
            prefix,
            factors,
            set(MM10_CHROM_SIZES),
        )
        for name, path, prefix in INPUT_FILES
        if name != "PolIIS5P"
    ]
    g4 = next(modality for modality in modalities if modality.name == "G4")
    regions = build_g4_windows(g4.peaks, args.g4_window_size)
    feature_names = training_feature_names(modalities)
    if len(feature_names) != 288:
        raise RuntimeError(f"Expected 288 features, found {len(feature_names)}")

    modality_maps = {
        modality.name: map_records_to_regions(regions, modality.peaks, "value")
        for modality in modalities
        if modality.name != "G4"
    }
    gtf = ensure_ensembl_gtf(args.ensembl_gtf, args.ensembl_url)
    promoters = read_ensembl_promoters(
        gtf, args.promoter_upstream, args.promoter_downstream
    )
    promoter_maps = map_records_to_regions(regions, promoters, "name")
    ct_tadb_boundaries = read_ct_tadb_boundaries(
        args.ct_tadb_boundaries, args.ct_tadb_threshold
    )
    ct_tadb_maps = map_records_to_regions(regions, ct_tadb_boundaries, "value")

    matrix = np.zeros((len(regions), len(feature_names)), dtype=np.float32)
    table_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []

    for index, region in enumerate(regions.itertuples(index=False)):
        window_length = region.end - region.start
        anchor_start = max(region.g4_start, region.start)
        anchor_end = min(region.g4_end, region.end)
        anchor_bases = anchor_end - anchor_start
        states = {
            "G4": RegionModality(
                peak=float(region.anchor_g4_peak),
                self_ratio=anchor_bases / region.g4_length,
                bin_ratio=anchor_bases / window_length,
                covered_bases=anchor_bases,
                intervals=[(anchor_start, anchor_end)],
                interval_text=f"{region.g4_start}-{region.g4_end}",
            ),
            **{
                modality.name: calculate_modality(
                    modality_maps[modality.name][index], region.start, region.end
                )
                for modality in modalities
                if modality.name != "G4"
            },
        }
        pair_bases: dict[tuple[str, str], int] = {}
        for left_index, left in enumerate(modalities):
            for right in modalities[left_index + 1 :]:
                pair_bases[left.name, right.name] = interval_bases(
                    intersect_intervals(
                        states[left.name].intervals, states[right.name].intervals
                    )
                )

        features: dict[str, float] = {}
        intervals: dict[str, str] = {}
        for modality in modalities:
            name, state = modality.name, states[modality.name]
            features[f"{name}_peak"] = state.peak
            features[f"{name}_self_ratio"] = state.self_ratio
            features[f"{name}_bin_ratio"] = state.bin_ratio
            for other in modalities:
                if other.name == name:
                    continue
                bases = pair_bases.get(
                    (name, other.name), pair_bases.get((other.name, name), 0)
                )
                features[f"{other.name}_overlapping_{name}"] = (
                    bases / state.covered_bases if state.covered_bases else 0.0
                )
            intervals[f"{name}_iv"] = state.interval_text

        values = np.round(
            np.asarray([features[name] for name in feature_names], dtype=np.float32), 2
        )
        matrix[index] = values
        features = dict(zip(feature_names, values.tolist()))
        promoter_records = promoter_maps[index]
        promoter_intervals = merge_clipped(promoter_records, region.start, region.end)
        promoter_bp = interval_bases(promoter_intervals)
        triple = intersect_intervals(
            intersect_intervals(states["ATAC"].intervals, states["H3K27ac"].intervals),
            states["H3K4me1"].intervals,
        )
        triple_bp = interval_bases(triple)
        boundary_records = ct_tadb_maps[index]
        boundary_intervals = merge_clipped(
            boundary_records, region.start, region.end
        )
        boundary_bp = interval_bases(boundary_intervals)
        boundary_score = max(
            (float(record[2]) for record in boundary_records), default=0.0
        )
        metadata = {
            "index": index,
            "chrom": region.chrom,
            "start": region.start,
            "end": region.end,
            "window_id": region.window_id,
            "g4_id": region.g4_id,
            "g4_start": region.g4_start,
            "g4_end": region.g4_end,
            "g4_length": region.g4_length,
            "window_length": region.window_length,
            "left_extension": region.left_extension,
            "right_extension": region.right_extension,
            "target_size_reached": region.target_size_reached,
            "neighbor_overlap_conflict": region.neighbor_overlap_conflict,
            "anchor_g4_peak": round(region.anchor_g4_peak, 2),
            "g4_peak_exceeds_window": region.g4_peak_exceeds_window,
        }
        evidence = {
            "promoter_evidence": int(promoter_bp > 0),
            "promoter_overlap_bp": promoter_bp,
            "promoter_overlap_ratio": round(promoter_bp / window_length, 2),
            "promoter_genes": (
                ";".join(str(record[2]) for record in promoter_records)
                if promoter_records else "-"
            ),
            "enhancer_evidence": int(triple_bp > 0),
            "enhancer_triple_overlap_bp": triple_bp,
            "enhancer_triple_overlap_ratio": round(triple_bp / window_length, 2),
            "tad_evidence": int(boundary_bp > 0),
            "tad_boundary_score": round(boundary_score, 2),
            "tad_boundary_overlap_bp": boundary_bp,
            "tad_boundary_overlap_ratio": round(boundary_bp / window_length, 2),
            "tad_boundary_iv": (
                ";".join(
                    f"{left}-{right}:{float(score):.4f}"
                    for left, right, score in boundary_records
                )
                if boundary_records else "-"
            ),
        }
        table_rows.append({**metadata, **features, **intervals})
        label_rows.append(metadata)
        evidence_rows.append(evidence)

    for table, metadata, evidence in zip(
        table_rows, label_rows, evidence_rows
    ):
        promoter = bool(evidence["promoter_evidence"])
        enhancer = bool(evidence["enhancer_evidence"])
        tad = bool(evidence["tad_evidence"])
        if promoter:
            label, source = "Promoter", "Ensembl_GRCm38_TSS"
            confidence = evidence["promoter_overlap_ratio"]
        elif enhancer:
            label, source = "Enhancer", "ATAC_H3K27ac_H3K4me1_triple_overlap"
            confidence = evidence["enhancer_triple_overlap_ratio"]
        elif tad:
            label, source = "TAD", "CT-TADB"
            confidence = evidence["tad_boundary_score"]
        else:
            label, source, confidence = "Others", "rule_based_other", 1.0
        raw_labels = [
            name for name, present in (
                ("Promoter", promoter), ("Enhancer", enhancer), ("TAD", tad)
            ) if present
        ]
        labels = {
            **evidence,
            "position_label": label,
            "label": LABEL_CODES[label],
            "position_multilabel": ";".join(raw_labels) if raw_labels else "Others",
            "position_ambiguous": int(len(raw_labels) > 1),
            "label_source": source,
            "label_confidence": round(float(confidence), 2),
            "keep_for_training": 1,
        }
        table.update(labels)
        metadata.update(labels)

    size_name = (
        f"{args.g4_window_size // 1000}kb"
        if args.g4_window_size % 1000 == 0
        else f"{args.g4_window_size}bp"
    )
    prefix = f"g4_position_{size_name}"
    metadata_columns = list(label_rows[0])
    grouped_columns: list[str] = []
    for modality in modalities:
        name = modality.name
        grouped_columns.extend([
            f"{name}_peak", f"{name}_self_ratio", f"{name}_bin_ratio",
            *[f"{other.name}_overlapping_{name}" for other in modalities if other.name != name],
            f"{name}_iv",
        ])
    with gzip.open(
        output_dir / f"{prefix}_full.tsv.gz", "wt", compresslevel=1
    ) as handle:
        pd.DataFrame(table_rows)[metadata_columns + grouped_columns].to_csv(
            handle, sep="\t", index=False
        )
    labels = pd.DataFrame(label_rows)
    labels.to_csv(output_dir / f"{prefix}_regions.tsv", sep="\t", index=False)
    matrix_with_label = np.column_stack(
        [matrix, labels["label"].to_numpy(dtype=np.float32)]
    )
    matrix_columns = [*feature_names, "label"]
    column_descriptions = [describe_column(name) for name in matrix_columns]
    pd.DataFrame({
        "index": np.arange(len(matrix_columns)),
        "name": matrix_columns,
        "role": ["input_feature"] * len(feature_names) + ["target_label"],
        "modality": [item[0] for item in column_descriptions],
        "data_form": [item[1] for item in column_descriptions],
    }).to_csv(
        output_dir / f"{prefix}_columns.tsv", sep="\t", index=False
    )

    with h5py.File(output_dir / f"{prefix}_dataset.h5", "w") as handle:
        dataset = handle.create_dataset(
            "matrix", data=matrix_with_label, maxshape=(None, len(matrix_columns)),
            chunks=(min(args.h5_chunk_rows, len(matrix)), len(matrix_columns)),
            compression="gzip", compression_opts=1, shuffle=True,
        )
        dataset.attrs.update(
            row_count=len(matrix), feature_count=288,
            column_count=len(matrix_columns), label_column="label",
            label_column_index=len(feature_names),
            label_encoding="Promoter=0;Enhancer=1;TAD=2;Others=3",
            class_counts=";".join(
                f"{name}={int((labels['position_label'] == name).sum())}"
                for name in LABEL_CODES
            ),
            balanced_class_weights=";".join(
                f"{name}="
                f"{(len(labels) / (len(LABEL_CODES) * count)) if count else 0.0:.4f}"
                for name in LABEL_CODES
                for count in [int((labels["position_label"] == name).sum())]
            ),
            modalities=";".join(modality.name for modality in modalities),
            label_source_features=(
                "ATAC_peak;ATAC_bin_ratio;H3K27ac_peak;H3K27ac_bin_ratio;"
                "H3K4me1_peak;H3K4me1_bin_ratio;CT-TADB_boundary_probability"
            ),
            target_window_size=args.g4_window_size,
            tad_boundary_method="CT-TADB",
            ct_tadb_boundary_file=str(args.ct_tadb_boundaries.resolve()),
            ct_tadb_probability_threshold=args.ct_tadb_threshold,
            ensembl_gtf=str(gtf),
            region_definition="G4_neighbor_aware_adaptive_window",
            promoter_definition=(
                f"strand_specific_TSS_upstream_{args.promoter_upstream}_"
                f"downstream_{args.promoter_downstream}"
            ),
            ensembl_release=102, assembly="GRCm38", polii_s5p_included=False,
        )

    plot_dataset_overview(
        labels, matrix, feature_names, modalities, args.g4_window_size,
        output_dir / f"{prefix}_overview.png",
    )
    remove_legacy_outputs(output_dir)
    print(f"Wrote {len(regions):,} G4-centered {args.g4_window_size}-bp regions")
    print(labels["position_label"].value_counts().to_string())
    print(
        f"TAD labels use CT-TADB boundaries at probability >= "
        f"{args.ct_tadb_threshold:.2f}"
    )



def build_gene_training_data(args: argparse.Namespace) -> None:
    if args.h5_chunk_rows <= 0 or args.abc_max_distance <= 0 or args.top_genes_per_model <= 0:
        raise ValueError("chunk rows and distances must be positive")
    data_dir = args.data_dir.resolve()
    g4_dir = output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'train_data').mkdir(exist_ok=True)
    gtf = ensure_ensembl_gtf(args.ensembl_gtf, args.ensembl_url)
    genes = add_expression(read_genes(gtf), data_dir / args.rna)
    modalities = load_required_modalities(data_dir, args.scaling_factors)
    genes = add_polii_targets(genes, modalities["PolIIS5P"])
    boundaries = ensure_boundaries(args.hic_boundaries.resolve())

    prefix = window_prefix(args.g4_window_size)
    regions = pd.read_csv(g4_dir / f"{prefix}_regions.tsv", sep="\t")
    columns = pd.read_csv(g4_dir / f"{prefix}_columns.tsv", sep="\t")
    feature_names = columns.loc[columns["role"] == "input_feature", "name"].tolist()
    with h5py.File(g4_dir / f"{prefix}_dataset.h5") as handle:
        features = handle["matrix"][:, : len(feature_names)]
    if len(regions) != len(features):
        raise RuntimeError("G4 region rows and feature rows are not aligned")
    regions["g4_row_index"] = np.arange(len(regions), dtype=np.int64)

    elements = build_abc_elements(modalities["ATAC"], modalities["H3K27ac"])
    enhancer_regions = regions.loc[
        regions["position_label"] == "Enhancer", ["chrom", "start", "end", "g4_id"]
    ].reset_index(drop=True)
    abc_genes = genes.loc[genes["rna_tpm"] >= args.min_tpm]
    abc = score_abc(
        enhancer_regions, abc_genes, elements,
        args.abc_max_distance, args.contact_min_distance, args.contact_gamma,
    )
    evidence = add_six_model_scores(
        abc, regions, features, feature_names, genes, boundaries
    )
    evidence = vote_six_models(
        evidence, args.abc_threshold, args.top_genes_per_model
    )

    valid_genes = set(genes["gene_id"])
    assignments = promoter_assignments(regions, genes)
    accepted = evidence.loc[evidence["accepted"] == 1]
    for row in accepted.itertuples(index=False):
        assignments.append({
            "gene_id": row.gene_id,
            "g4_id": row.g4_id,
            "g4_role": "Enhancer",
            "assignment_method": "six_method_unanimous_vote",
            "assignment_status": row.consensus_status,
            "boundary_side": "-",
            "assignment_distance": int(row.distance),
            "assignment_score": round(float(row.consensus_score), 4),
            "models_supporting": row.models_supporting,
        })
    assignments.extend(boundary_assignments_public(
        regions, genes, boundaries, args.boundary_max_distance
    ))
    assignments = pd.DataFrame(assignments).drop_duplicates(
        ["gene_id", "g4_id", "g4_role"]
    )
    assignments = assignments.merge(
        regions[[
            "g4_id", "g4_row_index", "chrom", "start", "end",
            "g4_start", "g4_end", "position_label", "label_confidence",
        ]],
        on="g4_id", how="left",
    )
    assignments = assignments.loc[assignments["gene_id"].isin(valid_genes)]
    summary = build_gene_summary(genes, assignments)
    gene_index = dict(zip(summary["gene_id"], summary["gene_index"]))
    assignments["gene_index"] = assignments["gene_id"].map(gene_index)
    assignments = assignments.sort_values(
        ["gene_index", "g4_role", "g4_row_index"]
    ).reset_index(drop=True)

    feature_frame = pd.DataFrame(features, columns=feature_names)
    feature_frame.insert(0, "g4_id", regions["g4_id"].to_numpy())
    detail = assignments.merge(feature_frame, on="g4_id", how="left")
    with gzip.open(output_dir / "gene_g4_assignments.tsv.gz", "wt", compresslevel=1) as handle:
        detail.to_csv(handle, sep="\t", index=False)
    summary.to_csv(output_dir / "gene_summary.tsv", sep="\t", index=False)
    with gzip.open(
        output_dir / "enhancer_gene_evidence.tsv.gz", "wt", compresslevel=1
    ) as handle:
        evidence.round(6).to_csv(handle, sep="\t", index=False)
    write_h5(
        output_dir / "train_data" / "cre_reg_gene_dataset.h5", summary, assignments, features,
        feature_names, args.h5_chunk_rows, MODEL_STATUS,
        {
            "min_tpm": args.min_tpm,
            "abc_threshold": args.abc_threshold,
            "abc_max_distance": args.abc_max_distance,
            "contact_min_distance": args.contact_min_distance,
            "contact_gamma": args.contact_gamma,
            "boundary_max_distance": args.boundary_max_distance,
            "top_genes_per_model": args.top_genes_per_model,
            "hic_boundaries": str(args.hic_boundaries),
        },
    )
    write_statistics(output_dir, evidence, assignments)

    print(f"ABC candidate elements: {len(elements):,}")
    print(f"Candidate E-G pairs: {len(evidence):,}")
    print(f"Six-model unanimous E-G pairs: {int(evidence['strict_consensus'].sum()):,}")
    print(f"Four-of-six majority E-G pairs: {int(evidence['majority_consensus'].sum()):,}")
    print(f"Genes with assigned G4: {len(summary):,}")
    print(assignments["g4_role"].value_counts().to_string())
    print(evidence["consensus_status"].value_counts().to_string())
    for model, status in MODEL_STATUS.items():
        print(f"{model}: {status}")



def main() -> None:
    args = parse_args()
    prefix = window_prefix(args.g4_window_size)
    required = [
        args.output_dir / f"{prefix}_regions.tsv",
        args.output_dir / f"{prefix}_columns.tsv",
        args.output_dir / f"{prefix}_dataset.h5",
    ]
    if args.rebuild_g4_regions or not all(path.exists() for path in required):
        build_g4_regions(args)
    build_gene_training_data(args)


if __name__ == "__main__":
    main()
