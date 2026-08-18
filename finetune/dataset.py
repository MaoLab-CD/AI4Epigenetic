from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def read_columns(path: Path) -> tuple[np.ndarray, list[str]]:
    indices, names = [], []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            indices.append(int(row["index"]))
            names.append(row["name"])
    return np.asarray(indices, dtype=np.int64), names


def select_input_and_target(
    columns_path: Path,
    target_name: str = "PolIIS5P_peak",
) -> tuple[np.ndarray, list[str], int]:
    indices, names = read_columns(columns_path)
    matches = [i for i, name in enumerate(names) if name == target_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one target column {target_name!r}, found {len(matches)}"
        )
    target_position = matches[0]
    keep = np.asarray(
        ["poliis5p" not in name.lower() for name in names], dtype=bool
    )
    return indices[keep], [name for name, flag in zip(names, keep) if flag], int(
        indices[target_position]
    )


def _read_rows(
    matrix: h5py.Dataset,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    if len(rows) == 0:
        return np.empty((0, len(columns)), dtype=np.float32)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    values = matrix[sorted_rows.tolist(), :][:, columns]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(values[inverse], dtype=np.float32)


def read_target(
    matrix_path: Path,
    target_index: int,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    with h5py.File(matrix_path, "r") as handle:
        matrix = handle["matrix"]
        if rows is None:
            return np.asarray(matrix[:, target_index], dtype=np.float32)
        return _read_rows(
            matrix,
            np.asarray(rows, dtype=np.int64),
            np.asarray([target_index], dtype=np.int64),
        )[:, 0]


def stratified_splits(
    target: np.ndarray,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if validation_ratio <= 0 or test_ratio <= 0:
        raise ValueError("validation_ratio and test_ratio must be positive")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be below 1")

    generator = np.random.default_rng(seed)
    groups = [np.flatnonzero(target == 0), np.flatnonzero(target > 0)]
    split_parts = {"train": [], "validation": [], "test": []}
    for group in groups:
        generator.shuffle(group)
        test_count = max(1, round(len(group) * test_ratio))
        validation_count = max(1, round(len(group) * validation_ratio))
        split_parts["test"].append(group[:test_count])
        split_parts["validation"].append(
            group[test_count : test_count + validation_count]
        )
        split_parts["train"].append(group[test_count + validation_count :])

    return {
        name: np.sort(np.concatenate(parts)).astype(np.int64)
        for name, parts in split_parts.items()
    }


def target_scale(
    target: np.ndarray,
    training_rows: np.ndarray,
    quantile: float,
) -> float:
    nonzero = target[training_rows]
    nonzero = nonzero[nonzero > 0]
    return float(np.quantile(nonzero, quantile)) if len(nonzero) else 1.0


class PolIIPeakDataset(Dataset):
    def __init__(
        self,
        matrix_path: Path,
        row_indices: np.ndarray,
        input_indices: np.ndarray,
        peak_mask: np.ndarray,
        peak_scales: np.ndarray,
        target_index: int,
        target_peak_scale: float,
    ):
        self.matrix_path = Path(matrix_path)
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.input_indices = np.asarray(input_indices, dtype=np.int64)
        self.peak_mask = np.asarray(peak_mask, dtype=bool)
        self.peak_log_scales = np.log1p(
            np.asarray(peak_scales, dtype=np.float32)
        )
        self.target_index = target_index
        self.target_log_scale = np.log1p(max(target_peak_scale, 1.0e-8))
        self._handle = None
        self._matrix = None

    def __len__(self):
        return len(self.row_indices)

    def _open(self):
        if self._matrix is None:
            self._handle = h5py.File(self.matrix_path, "r")
            self._matrix = self._handle["matrix"]
        return self._matrix

    def _normalize_inputs(self, values):
        values = np.asarray(values, dtype=np.float32).copy()
        peaks = values[..., self.peak_mask]
        scales = self.peak_log_scales[self.peak_mask]
        values[..., self.peak_mask] = np.clip(
            np.log1p(np.maximum(peaks, 0.0)) / scales, 0.0, 1.0
        )
        return values

    def _normalize_target(self, values):
        return np.clip(
            np.log1p(np.maximum(values, 0.0)) / self.target_log_scale,
            0.0,
            1.0,
        ).astype(np.float32)

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        rows = self.row_indices[np.asarray(indices, dtype=np.int64)]
        columns = np.concatenate(
            [self.input_indices, np.asarray([self.target_index])]
        )
        values = _read_rows(self._open(), rows, columns)
        inputs = self._normalize_inputs(values[:, :-1])
        targets = self._normalize_target(values[:, -1])
        return [
            {
                "values": torch.from_numpy(inputs[i]),
                "target": torch.tensor(targets[i]),
                "index": torch.tensor(rows[i]),
            }
            for i in range(len(rows))
        ]

    def __del__(self):
        if self._handle is not None:
            self._handle.close()
