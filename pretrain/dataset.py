from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class FeatureSelection:
    source_indices: np.ndarray
    names: list[str]
    excluded_source_indices: np.ndarray
    excluded_names: list[str]


def read_feature_selection(
    columns_path: Path,
    excluded_terms: tuple[str, ...] = ("PolIIS5P",),
) -> FeatureSelection:
    terms = tuple(term.lower() for term in excluded_terms)
    kept_indices: list[int] = []
    kept_names: list[str] = []
    excluded_indices: list[int] = []
    excluded_names: list[str] = []
    with columns_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source_index = int(row["index"])
            name = row["name"]
            if any(term in name.lower() for term in terms):
                excluded_indices.append(source_index)
                excluded_names.append(name)
            else:
                kept_indices.append(source_index)
                kept_names.append(name)
    if not kept_names:
        raise ValueError("No features remain after exclusion")
    return FeatureSelection(
        source_indices=np.asarray(kept_indices, dtype=np.int64),
        names=kept_names,
        excluded_source_indices=np.asarray(excluded_indices, dtype=np.int64),
        excluded_names=excluded_names,
    )


def random_splits(
    row_count: int,
    validation_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(row_count)
    validation_count = max(1, int(round(row_count * validation_ratio)))
    return {
        "train": np.sort(shuffled[validation_count:]),
        "validation": np.sort(shuffled[:validation_count]),
    }


def _read_matrix_rows(
    matrix: h5py.Dataset,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    if len(rows) == 0:
        return np.empty((0, len(columns)), dtype=np.float32)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    span = int(unique_rows[-1] - unique_rows[0] + 1)
    if span <= len(unique_rows) * 2:
        block = matrix[int(unique_rows[0]) : int(unique_rows[-1]) + 1, columns]
        values = block[unique_rows - unique_rows[0]][inverse]
    else:
        values = matrix[unique_rows.tolist(), :][:, columns][inverse]
    return np.asarray(values, dtype=np.float32)


def compute_peak_scales(
    matrix_path: Path,
    training_rows: np.ndarray,
    feature_indices: np.ndarray,
    feature_names: list[str],
    quantile: float = 0.995,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    peak_mask = np.asarray(
        [name.endswith("_peak") for name in feature_names],
        dtype=bool,
    )
    peak_scales = np.ones(len(feature_names), dtype=np.float32)
    with h5py.File(matrix_path, "r") as handle:
        matrix = handle["matrix"]
        peak_values = matrix[:, feature_indices[peak_mask]]
    training_peak_values = peak_values[training_rows]
    for local_index, values in zip(
        np.flatnonzero(peak_mask),
        training_peak_values.T,
    ):
        nonzero = values[values > 0]
        if len(nonzero):
            peak_scales[local_index] = np.quantile(nonzero, quantile)
    return peak_mask, peak_scales


class H5BinDataset(Dataset):
    def __init__(
        self,
        matrix_path: Path,
        row_indices: np.ndarray,
        feature_indices: np.ndarray,
        peak_mask: np.ndarray,
        peak_scales: np.ndarray,
    ) -> None:
        self.matrix_path = Path(matrix_path)
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.peak_mask = np.asarray(peak_mask, dtype=bool)
        self.peak_log_scales = np.log1p(
            np.asarray(peak_scales, dtype=np.float32)
        )
        self._handle: h5py.File | None = None
        self._matrix: h5py.Dataset | None = None

    def __len__(self) -> int:
        return len(self.row_indices)

    def _open(self) -> h5py.Dataset:
        if self._matrix is None:
            self._handle = h5py.File(self.matrix_path, "r")
            self._matrix = self._handle["matrix"]
        return self._matrix

    def __getitem__(self, index: int) -> torch.Tensor:
        row = np.asarray([self.row_indices[index]], dtype=np.int64)
        values = _read_matrix_rows(
            self._open(),
            row,
            self.feature_indices,
        )[0]
        return torch.from_numpy(self._normalize(values))

    def __getitems__(self, indices: list[int]) -> list[torch.Tensor]:
        rows = self.row_indices[np.asarray(indices, dtype=np.int64)]
        values = _read_matrix_rows(
            self._open(),
            rows,
            self.feature_indices,
        )
        values = self._normalize(values)
        return [torch.from_numpy(row) for row in values]

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).copy()
        peaks = values[..., self.peak_mask]
        scales = self.peak_log_scales[self.peak_mask]
        values[..., self.peak_mask] = np.clip(
            np.log1p(np.maximum(peaks, 0.0)) / scales,
            0.0,
            1.0,
        )
        return values

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._matrix = None

    def __del__(self) -> None:
        self.close()


class BlockShuffleSampler(Sampler[int]):
    """Shuffle HDF5-sized row blocks while preserving locality inside each block."""

    def __init__(
        self,
        dataset_size: int,
        block_size: int = 2048,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if dataset_size <= 0 or block_size <= 0:
            raise ValueError("dataset_size and block_size must be positive")
        self.dataset_size = dataset_size
        self.block_size = block_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        block_count = math.ceil(dataset_size / block_size)
        self.blocks_per_rank = math.ceil(block_count / world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        if self.world_size == 1:
            return self.dataset_size
        return self.blocks_per_rank * self.block_size

    def __iter__(self) -> Iterator[int]:
        blocks = [
            np.arange(start, min(start + self.block_size, self.dataset_size))
            for start in range(0, self.dataset_size, self.block_size)
        ]
        generator = np.random.default_rng(self.seed + self.epoch)
        if self.world_size == 1:
            generator.shuffle(blocks)
            return iter(np.concatenate(blocks).tolist())
        if len(blocks[-1]) < self.block_size:
            blocks[-1] = np.resize(blocks[-1], self.block_size)
        original_count = len(blocks)
        while len(blocks) % self.world_size:
            blocks.append(blocks[len(blocks) % original_count].copy())
        generator.shuffle(blocks)
        selected = blocks[self.rank :: self.world_size]
        return iter(np.concatenate(selected).tolist())
