from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from pretrain.dataset import normalize_features


def split_gene_indices(count: int, valid_ratio: float, test_ratio: float, seed: int):
    if min(valid_ratio, test_ratio) < 0 or valid_ratio + test_ratio >= 1:
        raise ValueError("validation and test ratios must be non-negative and sum to < 1")
    order = np.random.default_rng(seed).permutation(count)
    n_test = int(round(count * test_ratio))
    n_valid = int(round(count * valid_ratio))
    return {
        "test": np.sort(order[:n_test]),
        "validation": np.sort(order[n_test : n_test + n_valid]),
        "train": np.sort(order[n_test + n_valid :]),
    }


def target_scales(targets: h5py.Dataset, train_rows: np.ndarray, quantile: float):
    values = np.asarray(targets[train_rows.tolist()], dtype=np.float32)
    scales = np.ones(values.shape[1], dtype=np.float32)
    peak_mask = np.asarray([True, False, True, False])
    for column in np.flatnonzero(peak_mask):
        nonzero = values[:, column][values[:, column] > 0]
        if len(nonzero):
            scales[column] = np.quantile(nonzero, quantile)
    return peak_mask, scales


class CRERegGeneDataset(Dataset):
    def __init__(
        self,
        path: Path,
        gene_indices: np.ndarray,
        peak_mask: np.ndarray,
        peak_scales: np.ndarray,
        peak_normalization: str,
        target_peak_mask: np.ndarray,
        target_peak_scales: np.ndarray,
    ):
        self.path = Path(path)
        self.indices = np.asarray(gene_indices, dtype=np.int64)
        self.peak_mask = np.asarray(peak_mask, dtype=bool)
        self.peak_scales = np.asarray(peak_scales, dtype=np.float32)
        self.peak_normalization = peak_normalization
        self.target_peak_mask = np.asarray(target_peak_mask, dtype=bool)
        self.target_peak_scales = np.asarray(target_peak_scales, dtype=np.float32)
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        h5 = self._open()
        gene = int(self.indices[item])
        start, end = h5["gene_offsets"][gene : gene + 2]
        features = normalize_features(
            h5["features"][start:end], self.peak_mask,
            self.peak_scales, self.peak_normalization,
        )
        target = normalize_features(
            h5["polii_targets"][gene : gene + 1], self.target_peak_mask,
            self.target_peak_scales, self.peak_normalization,
        )[0]
        return {
            "gene_index": torch.tensor(gene),
            "features": torch.from_numpy(features),
            "role": torch.from_numpy(np.asarray(h5["role"][start:end], dtype=np.int64)),
            "target": torch.from_numpy(target),
        }

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


def collate_genes(batch):
    lengths = [len(item["features"]) for item in batch]
    width = max(lengths)
    feature_count = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), width, feature_count)
    roles = torch.zeros(len(batch), width, dtype=torch.long)
    valid = torch.zeros(len(batch), width, dtype=torch.bool)
    for row, item in enumerate(batch):
        length = lengths[row]
        features[row, :length] = item["features"]
        roles[row, :length] = item["role"]
        valid[row, :length] = True
    return {
        "gene_index": torch.stack([item["gene_index"] for item in batch]),
        "features": features,
        "role": roles,
        "valid": valid,
        "target": torch.stack([item["target"] for item in batch]),
    }
