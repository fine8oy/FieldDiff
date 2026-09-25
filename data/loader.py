from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..model.common import STBatch


VALUE_FILES = ("values_full.npy", "bike_values_full.npy")
SPATIAL_FILES = ("spatial_coords.npy", "bike_spatial_coords.npy")
TIME_FILES = ("time_features_full.npy", "bike_time_features_full.npy")


def collate_stbatch(samples: Sequence[STBatch]) -> STBatch:
    """Stack fixed-node samples produced by one processed domain."""

    values = torch.stack([sample.values for sample in samples], dim=0)
    spatial_coords = torch.stack(
        [sample.spatial_coords for sample in samples], dim=0
    )
    time_coords = torch.stack([sample.time_coords for sample in samples], dim=0)
    node_mask = (
        torch.stack([sample.node_mask for sample in samples], dim=0)
        if samples[0].node_mask is not None
        else None
    )
    obs_mask = (
        torch.stack([sample.obs_mask for sample in samples], dim=0)
        if samples[0].obs_mask is not None
        else None
    )
    metadata = {}
    for key, first in samples[0].metadata.items():
        if isinstance(first, torch.Tensor):
            metadata[key] = torch.stack(
                [sample.metadata[key] for sample in samples], dim=0
            )
    return STBatch(
        values=values,
        spatial_coords=spatial_coords,
        time_coords=time_coords,
        node_mask=node_mask,
        obs_mask=obs_mask,
        metadata=metadata,
    )


@dataclass(frozen=True)
class SplitIndices:
    train_nodes: np.ndarray
    test_nodes: np.ndarray
    train_time_stop: int
    num_time_steps: int

    def to_json(self) -> dict:
        return {
            "train_nodes": self.train_nodes.tolist(),
            "test_nodes": self.test_nodes.tolist(),
            "train_time_slice": [0, self.train_time_stop],
            "test_time_slice": [self.train_time_stop, self.num_time_steps],
        }


@dataclass(frozen=True)
class NormalizationStats:
    method: str
    value_offset: float
    value_scale: float
    finite_count: int

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Protocol:
    window_size: int = 24
    stride: int = 1
    eval_stride: int = 24
    node_train_ratio: float = 0.75
    time_train_ratio: float = 0.70
    train_val_ratio: float = 0.90
    forecasting_context_steps: int = 12
    interpolation_observed_steps: int = 12
    spatial_extrapolation_mask_ratio: float = 0.50
    spatiotemporal_imputation_mask_ratio: float = 0.50
    normalization_method: str = "standard"
    seed: int = 42


def _first_existing(root: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of {list(names)} exists under {root}")


def open_processed_arrays(root: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(root)
    values = np.load(_first_existing(root, VALUE_FILES), mmap_mode="r")
    spatial = np.load(_first_existing(root, SPATIAL_FILES), mmap_mode="r")
    time_features = np.load(_first_existing(root, TIME_FILES), mmap_mode="r")
    if values.ndim != 2:
        raise ValueError(f"{root}: values must be [T,N], got {values.shape}")
    if spatial.ndim != 2 or spatial.shape[0] != values.shape[1]:
        raise ValueError(f"{root}: spatial shape {spatial.shape} is incompatible with values {values.shape}")
    if time_features.ndim != 2 or time_features.shape[0] != values.shape[0]:
        raise ValueError(f"{root}: time shape {time_features.shape} is incompatible with values {values.shape}")
    return values, spatial, time_features


def compute_split_indices(num_nodes: int, num_time_steps: int, protocol: Protocol) -> SplitIndices:
    if not 0.0 < protocol.node_train_ratio <= 1.0:
        raise ValueError("node_train_ratio must be in (0,1]")
    if not 0.0 < protocol.time_train_ratio < 1.0:
        raise ValueError("time_train_ratio must be in (0,1)")
    if num_nodes < 2:
        raise ValueError("The four-task protocol requires at least two spatial nodes")
    if num_time_steps < 2 * protocol.window_size:
        raise ValueError(
            f"The four-task protocol requires at least {2 * protocol.window_size} time steps, got {num_time_steps}"
        )
    num_train_time = int(round(num_time_steps * protocol.time_train_ratio))
    num_train_time = min(num_time_steps - protocol.window_size, max(protocol.window_size, num_train_time))
    rng = np.random.default_rng(protocol.seed)
    nodes = rng.permutation(num_nodes).astype(np.int64)
    if protocol.node_train_ratio == 1.0:
        train_nodes = np.arange(num_nodes, dtype=np.int64)
        test_nodes = np.empty(0, dtype=np.int64)
    else:
        num_train_nodes = min(num_nodes - 1, max(1, int(round(num_nodes * protocol.node_train_ratio))))
        train_nodes = np.sort(nodes[:num_train_nodes])
        test_nodes = np.sort(nodes[num_train_nodes:])
    return SplitIndices(
        train_nodes=train_nodes,
        test_nodes=test_nodes,
        train_time_stop=num_train_time,
        num_time_steps=num_time_steps,
    )


def compute_scaled_train_time_stop(
    split: SplitIndices, protocol: Protocol, train_time_scale: float
) -> int:
    """Return a scaled train boundary without changing the held-out split."""

    if not 0.0 < train_time_scale <= 1.0:
        raise ValueError("train_time_scale must be in (0,1]")
    return min(
        split.train_time_stop,
        max(
            protocol.window_size,
            int(round(split.train_time_stop * train_time_scale)),
        ),
    )


def fit_normalization_stats(
    values: np.ndarray,
    split: SplitIndices,
    method: str,
    row_chunk_size: int = 1024,
) -> NormalizationStats:
    """Fit scalar value statistics on pre-split train nodes/time without materializing the array."""

    method = method.lower().replace("zscore", "standard")
    if method not in {"standard", "minmax"}:
        raise ValueError("normalization_method must be 'standard' or 'minmax'")
    total = 0.0
    square_total = 0.0
    finite_count = 0
    minimum = np.inf
    maximum = -np.inf
    for start in range(0, split.train_time_stop, row_chunk_size):
        stop = min(split.train_time_stop, start + row_chunk_size)
        chunk = np.asarray(values[start:stop, split.train_nodes], dtype=np.float32)
        finite = np.isfinite(chunk)
        count = int(finite.sum())
        if count == 0:
            continue
        selected = chunk[finite].astype(np.float64, copy=False)
        finite_count += count
        if method == "standard":
            total += float(selected.sum(dtype=np.float64))
            square_total += float(np.square(selected, dtype=np.float64).sum(dtype=np.float64))
        else:
            minimum = min(minimum, float(selected.min()))
            maximum = max(maximum, float(selected.max()))
    if finite_count == 0:
        raise ValueError("No finite training values were found")
    if method == "standard":
        mean = total / finite_count
        variance = max(0.0, square_total / finite_count - mean * mean)
        return NormalizationStats(method=method, value_offset=mean, value_scale=max(variance**0.5, 1e-6), finite_count=finite_count)
    # UrbanDiT's MinMax convention is [-1,1].
    return NormalizationStats(
        method=method,
        value_offset=(minimum + maximum) / 2.0,
        value_scale=max((maximum - minimum) / 2.0, 1e-6),
        finite_count=finite_count,
    )


def _window_starts(start: int, stop: int, window_size: int, stride: int) -> np.ndarray:
    last = stop - window_size
    if last < start:
        raise ValueError(f"No windows fit in time slice [{start},{stop}) with window_size={window_size}")
    return np.arange(start, last + 1, stride, dtype=np.int64)


def _nested_train_val_ids(num_windows: int, train_val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_windows < 2:
        ids = np.arange(num_windows, dtype=np.int64)
        return ids, ids
    rng = np.random.default_rng(seed)
    ids = rng.permutation(num_windows).astype(np.int64)
    count = min(num_windows - 1, max(1, int(round(num_windows * train_val_ratio))))
    return ids[:count], ids[count:]


def _select_nodes(nodes: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(nodes) <= maximum:
        return nodes
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(nodes, size=maximum, replace=False).astype(np.int64))


class ProcessedDomain:
    """A memory-mapped processed domain plus leakage-safe split statistics."""

    def __init__(
        self,
        name: str,
        root: Path | str,
        protocol: Protocol,
        stats: NormalizationStats | None = None,
        train_time_scale: float = 1.0,
    ) -> None:
        self.name = name
        self.root = Path(root)
        self.protocol = protocol
        self.train_time_scale = float(train_time_scale)
        self.values, self.spatial, self.time_features = open_processed_arrays(self.root)
        self.split = compute_split_indices(self.values.shape[1], self.values.shape[0], protocol)
        self.train_time_stop = compute_scaled_train_time_stop(
            self.split, protocol, self.train_time_scale
        )
        normalization_split = SplitIndices(
            train_nodes=self.split.train_nodes,
            test_nodes=self.split.test_nodes,
            train_time_stop=self.train_time_stop,
            num_time_steps=self.split.num_time_steps,
        )
        self.stats = stats or fit_normalization_stats(
            self.values, normalization_split, protocol.normalization_method
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.values.shape[0]), int(self.values.shape[1])

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize a clean data window and fail fast on non-finite values."""

        raw = np.asarray(values, dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(f"{self.name}: a sampled window contains NaN or Inf")
        out = (raw - self.stats.value_offset) / self.stats.value_scale
        return out.astype(np.float32, copy=False)

    def train_val_datasets(self, max_nodes: int) -> tuple["TrainWindowDataset", "TrainWindowDataset"]:
        starts = _window_starts(0, self.train_time_stop, self.protocol.window_size, self.protocol.stride)
        train_ids, val_ids = _nested_train_val_ids(len(starts), self.protocol.train_val_ratio, self.protocol.seed)
        return (
            TrainWindowDataset(self, starts[train_ids], max_nodes=max_nodes, seed=self.protocol.seed + 1009),
            TrainWindowDataset(self, starts[val_ids], max_nodes=max_nodes, seed=self.protocol.seed + 2003),
        )

    def full_train_dataset(self, max_nodes: int) -> "TrainWindowDataset":
        starts = _window_starts(0, self.train_time_stop, self.protocol.window_size, self.protocol.stride)
        return TrainWindowDataset(self, starts, max_nodes=max_nodes, seed=self.protocol.seed + 1009)

    def task_dataset(self, task: str, max_model_nodes: int) -> "TaskWindowDataset":
        return TaskWindowDataset(self, task=task, max_model_nodes=max_model_nodes)


class TrainWindowDataset(Dataset):
    def __init__(self, domain: ProcessedDomain, starts: np.ndarray, max_nodes: int, seed: int) -> None:
        self.domain = domain
        self.starts = np.asarray(starts, dtype=np.int64)
        self.max_nodes = max(1, int(max_nodes))
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.starts)

    def subset(self, ids: Iterable[int]) -> "TrainWindowDataset":
        ids = np.asarray(list(ids), dtype=np.int64)
        return TrainWindowDataset(self.domain, self.starts[ids], self.max_nodes, self.seed)

    def __getitem__(self, idx: int) -> STBatch:
        start = int(self.starts[idx])
        stop = start + self.domain.protocol.window_size
        nodes = _select_nodes(self.domain.split.train_nodes, self.max_nodes, self.seed + start)
        values = self.domain.normalize(self.domain.values[start:stop, nodes])
        observed = np.ones(values.shape, dtype=np.float32)
        return _make_batch(
            self.domain,
            start,
            nodes,
            values,
            obs_mask=observed,
            eval_mask=observed,
            mode_id=0,
        )


class TaskWindowDataset(Dataset):
    """Task evaluation windows, expanded into bounded spatial chunks."""

    TASK_TO_ID = {
        "forecasting": 1,
        "temporal_interpolation": 2,
        "spatial_extrapolation": 3,
        "spatiotemporal_imputation": 4,
    }

    def __init__(self, domain: ProcessedDomain, task: str, max_model_nodes: int) -> None:
        if task not in self.TASK_TO_ID:
            raise ValueError(f"Unsupported task: {task}")
        self.domain = domain
        self.task = task
        self.max_model_nodes = max(2, int(max_model_nodes))
        protocol = domain.protocol
        if not 0.0 < protocol.spatial_extrapolation_mask_ratio < 1.0:
            raise ValueError("spatial_extrapolation_mask_ratio must be in (0,1)")
        if not 0.0 < protocol.spatiotemporal_imputation_mask_ratio < 1.0:
            raise ValueError("spatiotemporal_imputation_mask_ratio must be in (0,1)")
        self.temporal_only_split = len(domain.split.test_nodes) == 0
        if self.temporal_only_split:
            # With no spatial holdout, every task evaluates the same nodes and
            # windows in the held-out temporal range. Only its observation mask
            # distinguishes one task from another.
            self.target_nodes = domain.split.train_nodes
            self.context_capacity = 0
            self.target_capacity = self.max_model_nodes
            time_start, time_stop = domain.split.train_time_stop, domain.split.num_time_steps
        elif task in {"forecasting", "temporal_interpolation"}:
            self.target_nodes = domain.split.train_nodes
            self.context_capacity = 0
            self.target_capacity = self.max_model_nodes
            time_start, time_stop = domain.split.train_time_stop, domain.split.num_time_steps
        else:
            self.target_nodes = domain.split.test_nodes
            self.context_capacity = min(
                len(domain.split.train_nodes),
                max(1, int(round(self.max_model_nodes * protocol.node_train_ratio))),
            )
            self.target_capacity = max(1, self.max_model_nodes - self.context_capacity)
            if task == "spatial_extrapolation":
                time_start, time_stop = 0, domain.split.train_time_stop
            else:
                time_start, time_stop = domain.split.train_time_stop, domain.split.num_time_steps
        self.starts = _window_starts(time_start, time_stop, protocol.window_size, protocol.eval_stride)
        self.num_node_chunks = int(np.ceil(len(self.target_nodes) / self.target_capacity))
        # Balanced chunks avoid a one-node tail, for which an exact 50% spatial
        # imputation mask would contain no evaluation target.
        self.node_chunks = [chunk.astype(np.int64, copy=False) for chunk in np.array_split(self.target_nodes, self.num_node_chunks)]

    def __len__(self) -> int:
        return len(self.starts) * self.num_node_chunks

    def __getitem__(self, idx: int) -> STBatch:
        window_idx, chunk_idx = divmod(idx, self.num_node_chunks)
        start = int(self.starts[window_idx])
        stop = start + self.domain.protocol.window_size
        nodes = self.node_chunks[chunk_idx]
        values = self.domain.normalize(self.domain.values[start:stop, nodes])
        obs_mask, eval_mask = self._task_masks(window_idx, chunk_idx, values.shape)
        batch = _make_batch(
            self.domain,
            start,
            nodes,
            values,
            obs_mask=obs_mask,
            eval_mask=eval_mask,
            mode_id=self.TASK_TO_ID[self.task],
        )
        if self.context_capacity:
            context_nodes = _select_nodes(
                self.domain.split.train_nodes,
                self.context_capacity,
                self.domain.protocol.seed + 7919 * window_idx,
            )
            context = self.domain.normalize(self.domain.values[start:stop, context_nodes])
            context_obs_mask = np.ones(context.shape, dtype=np.float32)
            batch.metadata.update(
                {
                    "context_values": torch.from_numpy(context[..., None]),
                    "context_spatial_coords": torch.from_numpy(
                        np.array(self.domain.spatial[context_nodes], dtype=np.float32, copy=True)
                    ),
                    "context_time_coords": torch.from_numpy(
                        np.array(self.domain.time_features[start:stop], dtype=np.float32, copy=True)
                    ),
                    "context_obs_mask": torch.from_numpy(context_obs_mask),
                    "context_node_indices": torch.from_numpy(context_nodes.astype(np.int64, copy=True)),
                }
            )
        return batch

    def _task_masks(self, window_idx: int, chunk_idx: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        time_steps, num_nodes = shape
        obs = np.ones(shape, dtype=np.float32)
        if self.task == "forecasting":
            obs[self.domain.protocol.forecasting_context_steps :, :] = 0.0
        elif self.task == "temporal_interpolation":
            obs[:, :] = 0.0
            rng = np.random.default_rng(self.domain.protocol.seed + window_idx)
            count = min(time_steps, self.domain.protocol.interpolation_observed_steps)
            obs[rng.permutation(time_steps)[:count], :] = 1.0
        elif self.task == "spatial_extrapolation":
            if self.temporal_only_split:
                rng = np.random.default_rng(
                    self.domain.protocol.seed + 32452843 * window_idx + chunk_idx
                )
                observed = max(
                    1,
                    int(round(num_nodes * (1.0 - self.domain.protocol.spatial_extrapolation_mask_ratio))),
                )
                obs[:, :] = 0.0
                obs[:, rng.permutation(num_nodes)[:observed]] = 1.0
            else:
                obs[:, :] = 0.0
        else:
            rng = np.random.default_rng(self.domain.protocol.seed + 104729 * window_idx + chunk_idx)
            obs[:, :] = 0.0
            observed = max(
                1,
                int(round(num_nodes * (1.0 - self.domain.protocol.spatiotemporal_imputation_mask_ratio))),
            )
            for time_index in range(time_steps):
                obs[time_index, rng.permutation(num_nodes)[:observed]] = 1.0
        return obs, 1.0 - obs


def _make_batch(
    domain: ProcessedDomain,
    start: int,
    nodes: np.ndarray,
    values: np.ndarray,
    obs_mask: np.ndarray,
    eval_mask: np.ndarray,
    mode_id: int,
) -> STBatch:
    stop = start + domain.protocol.window_size
    return STBatch(
        values=torch.from_numpy(values[..., None]),
        spatial_coords=torch.from_numpy(np.array(domain.spatial[nodes], dtype=np.float32, copy=True)),
        time_coords=torch.from_numpy(np.array(domain.time_features[start:stop], dtype=np.float32, copy=True)),
        node_mask=torch.ones(len(nodes), dtype=torch.float32),
        obs_mask=torch.from_numpy(obs_mask),
        metadata={
            "eval_mask": torch.from_numpy(eval_mask),
            "value_offset": torch.tensor(domain.stats.value_offset, dtype=torch.float32),
            "value_scale": torch.tensor(domain.stats.value_scale, dtype=torch.float32),
            "value_mean": torch.tensor(domain.stats.value_offset, dtype=torch.float32),
            "value_std": torch.tensor(domain.stats.value_scale, dtype=torch.float32),
            "window_start": torch.tensor(start, dtype=torch.long),
            "window_end": torch.tensor(stop, dtype=torch.long),
            "mode_id": torch.tensor(mode_id, dtype=torch.long),
            "node_indices": torch.from_numpy(nodes.astype(np.int64, copy=True)),
        },
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_stbatch,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def audit_dataset_roots(data_root: Path | str, names: Sequence[str]) -> list[dict]:
    rows: list[dict] = []
    for name in names:
        root = Path(data_root) / name
        row = {"dataset": name, "root": str(root), "exists": root.is_dir()}
        if root.is_dir():
            try:
                values, spatial, time_features = open_processed_arrays(root)
                row.update(
                    {
                        "values_shape": list(values.shape),
                        "values_dtype": str(values.dtype),
                        "spatial_shape": list(spatial.shape),
                        "time_shape": list(time_features.shape),
                        "valid": True,
                    }
                )
            except Exception as exc:  # audit should report every domain in one pass
                row.update({"valid": False, "error": repr(exc)})
        else:
            row["valid"] = False
        rows.append(row)
    return rows
