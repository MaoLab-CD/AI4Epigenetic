from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from pretrain.dataset import normalize_features


def split_cre_indices(count: int, valid_ratio: float, test_ratio: float, seed: int):
    if valid_ratio < 0 or test_ratio < 0 or valid_ratio + test_ratio >= 1:
        raise ValueError("validation and test ratios must be non-negative and sum to < 1")
    order = np.random.default_rng(seed).permutation(count)
    n_test = int(round(count * test_ratio))
    n_valid = int(round(count * valid_ratio))
    return {
        "test": np.sort(order[:n_test]),
        "validation": np.sort(order[n_test:n_test + n_valid]),
        "train": np.sort(order[n_test + n_valid:]),
    }


def target_peak_metadata(names: list[str], matrix: h5py.Dataset, train_rows: np.ndarray, quantile=0.995):
    peak_mask = np.asarray([name.endswith("_peak") for name in names], dtype=bool)
    scales = np.ones(len(names), dtype=np.float32)
    values = np.asarray(matrix[train_rows.tolist(), :], dtype=np.float32)
    for column in np.flatnonzero(peak_mask):
        nonzero = values[:, column][values[:, column] > 0]
        if len(nonzero):
            scales[column] = np.quantile(nonzero, quantile)
    return peak_mask, scales


class G4ToCREDataset(Dataset):
    def __init__(
        self,
        path: Path,
        cre_indices: np.ndarray,
        input_peak_mask: np.ndarray,
        input_peak_scales: np.ndarray,
        peak_normalization: str,
        target_peak_mask: np.ndarray,
        target_peak_scales: np.ndarray,
        max_g4_per_cre: int = 256,
        seed: int = 0,
    ):
        self.path = Path(path)
        self.cre_indices = np.asarray(cre_indices, dtype=np.int64)
        self.input_peak_mask = np.asarray(input_peak_mask, dtype=bool)
        self.input_peak_scales = np.asarray(input_peak_scales, dtype=np.float32)
        self.peak_normalization = peak_normalization
        self.target_peak_mask = np.asarray(target_peak_mask, dtype=bool)
        self.target_peak_scales = np.asarray(target_peak_scales, dtype=np.float32)
        self.max_g4 = max_g4_per_cre
        self.seed = seed
        self._h5 = None

    def __len__(self):
        return len(self.cre_indices)

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
            self.g_count = len(self._h5["g4/id"])
            self.g_chrom = self._h5["g4/chrom"][:].astype(str)
            self.c_chrom = self._h5["cre/chrom"][:].astype(str)
            self.g_center = (
                self._h5["g4/g4_start"][:] + self._h5["g4/g4_end"][:]
            ) / 2.0
            self.c_center = (
                self._h5["cre/start"][:] + self._h5["cre/end"][:]
            ) / 2.0
            self.overlap = self._h5["relations/overlap_pair_key"][:]
            self.far = self._h5["relations/far_pair_key"][:]
            self.far_score = self._h5["relations/far_hic_score"][:]
            names = self._h5["pretrain_feature_names"][:].astype(str).tolist()
            self.g4_peak_index = names.index("G4_peak")
        return self._h5

    @staticmethod
    def _slice_pairs(keys, cre_index, g_count):
        lo, hi = cre_index * g_count, (cre_index + 1) * g_count
        left, right = np.searchsorted(keys, [lo, hi])
        return keys[left:right] - lo, left, right

    def _select_g4(self, cre_index):
        overlap, _, _ = self._slice_pairs(self.overlap, cre_index, self.g_count)
        far, _, _ = self._slice_pairs(self.far, cre_index, self.g_count)
        required = np.unique(np.concatenate([overlap, far])).astype(np.int64)
        if self.max_g4 <= 0 or self.max_g4 >= self.g_count:
            return np.arange(self.g_count, dtype=np.int64)
        if len(required) >= self.max_g4:
            return required[:self.max_g4]
        remaining = np.setdiff1d(np.arange(self.g_count), required, assume_unique=True)
        rng = np.random.default_rng(self.seed + int(cre_index))
        sampled = rng.choice(
            remaining, min(self.max_g4 - len(required), len(remaining)), replace=False
        )
        return np.sort(np.concatenate([required, sampled])).astype(np.int64)

    def __getitem__(self, item):
        h5 = self._open()
        ci = int(self.cre_indices[item])
        gi = self._select_g4(ci)
        g_values = normalize_features(
            h5["g4/control_features"][gi], self.input_peak_mask,
            self.input_peak_scales, self.peak_normalization,
        )
        c_values = normalize_features(
            h5["cre/control_features"][ci:ci + 1], self.input_peak_mask,
            self.input_peak_scales, self.peak_normalization,
        )[0]
        target = normalize_features(
            h5["cre/perturbed_targets"][ci:ci + 1], self.target_peak_mask,
            self.target_peak_scales, self.peak_normalization,
        )[0]

        relation = np.zeros((len(gi), 3), dtype=np.float32)
        relation[:, 1] = 1.0
        overlap, _, _ = self._slice_pairs(self.overlap, ci, self.g_count)
        far, far_left, far_right = self._slice_pairs(self.far, ci, self.g_count)
        overlap_pos = np.flatnonzero(np.isin(gi, overlap))
        far_pos = np.flatnonzero(np.isin(gi, far))
        relation[overlap_pos] = (1, 0, 0)
        relation[far_pos] = (0, 0, 1)

        hic = np.zeros(len(gi), dtype=np.float32)
        if len(far):
            score_lookup = dict(zip(far, self.far_score[far_left:far_right]))
            hic[far_pos] = [score_lookup[int(index)] for index in gi[far_pos]]
        cis = (self.g_chrom[gi] == self.c_chrom[ci]).astype(np.float32)
        signed_distance = np.where(cis > 0, self.g_center[gi] - self.c_center[ci], 0.0)

        perturb = np.asarray(h5["g4/perturbation"][gi], dtype=np.float32)
        peak_scale = max(float(self.input_peak_scales[self.g4_peak_index]), 1e-8)
        perturb[:, 1] = np.sign(perturb[:, 1]) * np.log1p(
            np.abs(perturb[:, 1])
        ) / np.log1p(peak_scale)
        perturb = np.clip(perturb, -1, 1)
        return {
            "cre_index": torch.tensor(ci),
            "g4": torch.from_numpy(g_values),
            "cre": torch.from_numpy(c_values),
            "relation": torch.from_numpy(relation),
            "distance": torch.from_numpy(signed_distance.astype(np.float32)),
            "cis": torch.from_numpy(cis),
            "hic": torch.from_numpy(hic),
            "perturbation": torch.from_numpy(perturb),
            "target": torch.from_numpy(target),
        }

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()
