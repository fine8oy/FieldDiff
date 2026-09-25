from __future__ import annotations

"""Shared batch types and small runtime utilities."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Iterator, Mapping, Optional

import torch

class STTask(str, Enum):
    FORECASTING = "forecasting"
    TEMPORAL_INTERPOLATION = "temporal_interpolation"
    SPATIAL_EXTRAPOLATION = "spatial_extrapolation"
    SPATIOTEMPORAL_IMPUTATION = "spatiotemporal_imputation"

class ObservationMask:
    """Binary mask over node-time pairs with shape [B, T, N]."""

    mask: torch.Tensor
    task: Optional[STTask] = None

    def observed_fraction(self) -> float:
        return float(self.mask.float().mean().item())

class STBatch:
    """
    Canonical batch for one domain.

    Shapes:
    - values: [B, T, N, C]
    - spatial_coords: [B, N, Ds] or [B, T, N, Ds]
    - time_coords: [B, T, Dt] or [B, T, N, Dt]
      where Dt can be calendar-style features such as [year, month, day, hour]
    - node_mask: [B, N]
    - obs_mask: [B, T, N]
    """

    values: torch.Tensor
    spatial_coords: torch.Tensor
    time_coords: torch.Tensor
    node_mask: Optional[torch.Tensor] = None
    obs_mask: Optional[torch.Tensor] = None
    metadata: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def time_steps(self) -> int:
        return int(self.values.shape[1])

    @property
    def max_nodes(self) -> int:
        return int(self.values.shape[2])

class ModelConfig:
    """
    High-level architecture config.

    This is intentionally minimal for the first scaffold.
    """

    time_steps: int = 24
    value_dim: int = 1
    spatial_dim: int = 2
    time_coord_dim: int = 4
    use_periodic_time_features: bool = True
    coord_embed_dim: int = 64
    time_embed_dim: int = 64
    token_dim: int = 128
    latent_dim: int = 128
    num_latents: int = 32
    num_encoder_stages: int = 3
    decoder_query_dim: int = 96
    decoder_query_chunk_size: int = 65536
    cross_heads: int = 4
    cross_dim_head: int = 32
    self_heads: int = 4
    self_dim_head: int = 32
    decoder_depth: int = 2
    diffusion_width: int = 256
    diffusion_depth: int = 6
    diffusion_heads: int = 8
    diffusion_dim_head: int = 32
    diffusion_dropout: float = 0.0
    spatial_fourier_size: int = 32
    temporal_fourier_size: int = 32
    spatial_fourier_scale: float = 10.0
    temporal_fourier_scale: float = 10.0
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    p_mean: float = -1.2
    p_std: float = 1.2
    cond_drop_prob: float = 0.1
    guidance_scale: float = 1.0
    consistency_scale: float = 1.0
    sampling_steps: int = 32

class DataSplitConfig:
    window_size: int = 24
    stride: int = 1
    node_train_ratio: float = 0.75
    time_train_ratio: float = 0.7
    train_val_ratio: float = 0.9
    random_seed: int = 42
    normalization_method: str = "zscore"
    # None preserves the historical behavior where coordinates use the same
    # method as values.  Set this explicitly (for example, to ``"minmax"``)
    # when ablating value normalization without changing coordinate geometry.
    coordinate_normalization_method: Optional[str] = None
    normalize_time_coords: bool = False
    forecasting_context_steps: int = 12
    interpolation_observed_steps: int = 12
    spatiotemporal_imputation_mask_ratio: float = 0.5

def freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()

def unfreeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True
    module.train()

def move_batch_to_device(batch: STBatch, device: torch.device) -> STBatch:
    metadata = {}
    for key, value in batch.metadata.items():
        if isinstance(value, torch.Tensor):
            metadata[key] = value.to(device)
        else:
            metadata[key] = value
    return STBatch(
        values=batch.values.to(device),
        spatial_coords=batch.spatial_coords.to(device),
        time_coords=batch.time_coords.to(device),
        node_mask=batch.node_mask.to(device) if batch.node_mask is not None else None,
        obs_mask=batch.obs_mask.to(device) if batch.obs_mask is not None else None,
        metadata=metadata,
    )

class BalancedDomainIterator:
    """Deterministic round-robin domain sampler with independent loader cycling."""

    def __init__(
        self,
        loaders: Mapping[str, DomainLoaderView],
        steps: int,
        offset: int = 0,
    ) -> None:
        if not loaders:
            raise ValueError("BalancedDomainIterator needs at least one loader.")
        if steps < 1:
            raise ValueError("steps must be positive.")
        self.loaders = dict(loaders)
        self.steps = int(steps)
        self.offset = int(offset)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[tuple[str, STBatch]]:
        names = tuple(self.loaders)
        iterators = {name: iter(self.loaders[name]) for name in names}
        for step in range(self.steps):
            name = names[(self.offset + step) % len(names)]
            try:
                batch = next(iterators[name])
            except StopIteration:
                iterators[name] = iter(self.loaders[name])
                batch = next(iterators[name])
            yield name, batch

def valid_mask_for_batch(batch: STBatch) -> torch.Tensor:
    valid = batch.metadata.get("valid_mask")
    if valid is None:
        valid = torch.ones(
            batch.batch_size,
            batch.time_steps,
            batch.max_nodes,
            device=batch.values.device,
            dtype=batch.values.dtype,
        )
    if batch.node_mask is not None:
        valid = valid * batch.node_mask[:, None, :].to(valid.dtype)
    return valid
