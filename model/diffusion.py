from __future__ import annotations

"""Aligned conditional EDM used by the V2 anchor-affine experiment."""

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ModelConfig, STBatch, STTask
from .neural_field import (
    AdaptiveGlobalLocalStructuredField,
    AnchorAffineValueAnchoredFoundationField,
    FOUNDATION_DOMAIN_FEATURE_NAMES,
    VALUE_ANCHOR_DIM,
    ValueAnchoredFoundationFieldV2,
    build_full_query_coords,
    domain_features_from_batch,
    expand_spatial_coords,
    expand_time_coords,
    StructuredAxialBlock,
)

class CurrentConditionalTaskConfig:
    forecasting_context_steps: int = 12
    interpolation_observed_steps: int = 12
    spatial_observed_node_ratio: float = 0.5
    spatiotemporal_observed_node_ratio: float = 0.5

def append_dims(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
    return x.reshape(*x.shape, *((1,) * (target_ndim - x.ndim)))

def expand_spatial_coords_once(batch: STBatch) -> torch.Tensor:
    spatial = batch.spatial_coords
    if spatial.ndim == 3:
        return spatial
    if spatial.ndim == 4:
        return spatial[:, 0, :, :]
    raise ValueError(f"Unsupported spatial_coords shape: {tuple(spatial.shape)}")

def window_start_time_from_batch(batch: STBatch) -> torch.Tensor:
    time_coords = batch.time_coords
    if time_coords.ndim == 3:
        return time_coords[:, 0, :]
    if time_coords.ndim == 4:
        return time_coords[:, 0, 0, :]
    raise ValueError(f"Unsupported time_coords shape: {tuple(time_coords.shape)}")

def build_task_mask(
    task: STTask,
    batch_size: int,
    time_steps: int,
    num_nodes: int,
    device: Optional[torch.device] = None,
    observed_time_steps: int = 12,
    observed_node_ratio: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    mask = torch.zeros(batch_size, time_steps, num_nodes, device=device)
    if task == STTask.FORECASTING:
        mask[:, :observed_time_steps, :] = 1
    elif task == STTask.TEMPORAL_INTERPOLATION:
        for b in range(batch_size):
            idx = torch.randperm(time_steps, generator=generator, device=device)[:observed_time_steps]
            mask[b, idx, :] = 1
    elif task == STTask.SPATIAL_EXTRAPOLATION:
        keep = max(1, int(round(num_nodes * observed_node_ratio)))
        for b in range(batch_size):
            idx = torch.randperm(num_nodes, generator=generator, device=device)[:keep]
            mask[b, :, idx] = 1
    elif task == STTask.SPATIOTEMPORAL_IMPUTATION:
        keep = max(1, int(round(num_nodes * observed_node_ratio)))
        for b in range(batch_size):
            for t in range(time_steps):
                idx = torch.randperm(num_nodes, generator=generator, device=device)[:keep]
                mask[b, t, idx] = 1
    else:
        raise ValueError(f"Unsupported task: {task}")
    return mask

def build_condition_view(
    batch: STBatch,
    task: STTask,
    config: CurrentConditionalTaskConfig,
    generator: Optional[torch.Generator] = None,
) -> STBatch:
    if task == STTask.FORECASTING:
        mask = build_task_mask(
            task,
            batch.batch_size,
            batch.time_steps,
            batch.max_nodes,
            device=batch.values.device,
            observed_time_steps=config.forecasting_context_steps,
            generator=generator,
        )
    elif task == STTask.TEMPORAL_INTERPOLATION:
        mask = build_task_mask(
            task,
            batch.batch_size,
            batch.time_steps,
            batch.max_nodes,
            device=batch.values.device,
            observed_time_steps=config.interpolation_observed_steps,
            generator=generator,
        )
    elif task == STTask.SPATIAL_EXTRAPOLATION:
        mask = build_task_mask(
            task,
            batch.batch_size,
            batch.time_steps,
            batch.max_nodes,
            device=batch.values.device,
            observed_node_ratio=config.spatial_observed_node_ratio,
            generator=generator,
        )
    elif task == STTask.SPATIOTEMPORAL_IMPUTATION:
        mask = build_task_mask(
            task,
            batch.batch_size,
            batch.time_steps,
            batch.max_nodes,
            device=batch.values.device,
            observed_node_ratio=config.spatiotemporal_observed_node_ratio,
            generator=generator,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    if batch.node_mask is not None:
        mask = mask * batch.node_mask[:, None, :].to(mask.dtype)
    if batch.obs_mask is not None:
        mask = mask * batch.obs_mask.to(mask.dtype)
    return STBatch(
        values=batch.values,
        spatial_coords=batch.spatial_coords,
        time_coords=batch.time_coords,
        node_mask=batch.node_mask,
        obs_mask=mask.to(batch.values.dtype),
        metadata=batch.metadata,
    )

def build_condition_batch_from_metadata(batch: STBatch) -> Optional[STBatch]:
    context_values = batch.metadata.get("context_values")
    if context_values is None:
        return None
    context_spatial_coords = batch.metadata["context_spatial_coords"]
    context_time_coords = batch.metadata["context_time_coords"]
    context_obs_mask = batch.metadata.get("context_obs_mask")
    context_node_count = context_spatial_coords.shape[-2]
    return STBatch(
        values=context_values,
        spatial_coords=context_spatial_coords,
        time_coords=context_time_coords,
        node_mask=torch.ones(
            context_values.shape[0],
            context_node_count,
            device=context_values.device,
            dtype=context_values.dtype,
        ),
        obs_mask=context_obs_mask,
        metadata={},
    )

def merge_condition_batches(primary_batch: STBatch, context_batch: Optional[STBatch]) -> STBatch:
    if context_batch is None:
        return primary_batch
    if primary_batch.values.shape[0] != context_batch.values.shape[0] or primary_batch.values.shape[1] != context_batch.values.shape[1]:
        raise ValueError("Primary batch and context batch must share batch/time dimensions.")

    merged_values = torch.cat([context_batch.values, primary_batch.values], dim=2)
    merged_spatial = torch.cat([context_batch.spatial_coords, primary_batch.spatial_coords], dim=1)

    if context_batch.obs_mask is not None and primary_batch.obs_mask is not None:
        merged_obs_mask = torch.cat([context_batch.obs_mask, primary_batch.obs_mask], dim=2)
    elif context_batch.obs_mask is not None:
        merged_obs_mask = context_batch.obs_mask
    else:
        merged_obs_mask = primary_batch.obs_mask

    if context_batch.node_mask is not None and primary_batch.node_mask is not None:
        merged_node_mask = torch.cat([context_batch.node_mask, primary_batch.node_mask], dim=1)
    elif context_batch.node_mask is not None:
        merged_node_mask = context_batch.node_mask
    else:
        merged_node_mask = primary_batch.node_mask

    return STBatch(
        values=merged_values,
        spatial_coords=merged_spatial,
        time_coords=primary_batch.time_coords,
        node_mask=merged_node_mask,
        obs_mask=merged_obs_mask,
        metadata={},
    )

def masked_consistency_loss(
    prediction: torch.Tensor,
    observation: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    residual = (prediction - observation) * mask.unsqueeze(-1)
    sq = residual.square()
    if reduction == "sum":
        return sq.sum()
    if reduction == "none":
        return sq
    denom = mask.sum().clamp_min(1).float() * prediction.shape[-1]
    return sq.sum() / denom

CONDITIONAL_TASKS = (
    STTask.FORECASTING,
    STTask.TEMPORAL_INTERPOLATION,
    STTask.SPATIAL_EXTRAPOLATION,
    STTask.SPATIOTEMPORAL_IMPUTATION,
)

class StructuredConditionOutput:
    mean_latent: torch.Tensor
    condition_grid: torch.Tensor
    confidence: torch.Tensor
    global_context: torch.Tensor
    frame_observed_ratio: torch.Tensor
    previous_distance: torch.Tensor
    next_distance: torch.Tensor

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -torch.log(torch.tensor(10000.0, device=x.device, dtype=x.dtype))
            * torch.arange(half, device=x.device, dtype=x.dtype)
            / max(half - 1, 1)
        )
        args = x[:, None] * frequencies[None, :]
        out = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2:
            out = torch.cat([out, torch.zeros_like(out[:, :1])], dim=-1)
        return out

def _frame_gap_features(frame_observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distances and availability of nearest observed frames on both sides."""
    b, t = frame_observed.shape
    device = frame_observed.device
    previous_distance = torch.full((b, t), t, device=device, dtype=torch.float32)
    next_distance = torch.full((b, t), t, device=device, dtype=torch.float32)
    has_previous = torch.zeros((b, t), device=device, dtype=torch.float32)
    has_next = torch.zeros((b, t), device=device, dtype=torch.float32)
    last = torch.full((b,), -1, device=device, dtype=torch.long)
    for index in range(t):
        observed = frame_observed[:, index]
        last = torch.where(observed, torch.full_like(last, index), last)
        valid = last >= 0
        previous_distance[:, index] = torch.where(valid, index - last, t).float()
        has_previous[:, index] = valid.float()
    future = torch.full((b,), t, device=device, dtype=torch.long)
    for index in range(t - 1, -1, -1):
        observed = frame_observed[:, index]
        future = torch.where(observed, torch.full_like(future, index), future)
        valid = future < t
        next_distance[:, index] = torch.where(valid, future - index, t).float()
        has_next[:, index] = valid.float()
    scale = float(max(t - 1, 1))
    return previous_distance / scale, next_distance / scale, has_previous, has_next

class StructuredBridgeConditionEncoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        neural_field: nn.Module,
        bridge_layers: int = 2,
        bridge_axial_layers: int = 2,
        heads: int = 4,
        confidence_bandwidth: float = 0.25,
    ) -> None:
        super().__init__()
        self.config = config
        self.confidence_bandwidth = confidence_bandwidth
        object.__setattr__(self, "_neural_field", neural_field)
        self.value_proj = nn.Sequential(
            nn.Linear(config.value_dim, config.token_dim), nn.GELU(), nn.Linear(config.token_dim, config.token_dim)
        )
        self.mask_embed = nn.Linear(1, config.token_dim)
        num_slots = neural_field.num_spatial_latents
        self.missing_slots = nn.Parameter(torch.zeros(1, 1, num_slots, config.latent_dim))
        self.gap_feature_proj = nn.Sequential(
            nn.Linear(6, config.latent_dim), nn.GELU(), nn.Linear(config.latent_dim, config.latent_dim)
        )
        self.bidirectional_bridge = nn.GRU(
            config.latent_dim,
            config.latent_dim,
            num_layers=bridge_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if bridge_layers == 1 else 0.1,
        )
        self.direction_gate = nn.Sequential(nn.Linear(6, config.latent_dim), nn.Sigmoid())
        self.axial_blocks = nn.ModuleList(
            [StructuredAxialBlock(config.latent_dim, heads) for _ in range(bridge_axial_layers)]
        )
        self.condition_norm = nn.LayerNorm(config.latent_dim)
        self.mean_head = nn.Linear(config.latent_dim, config.latent_dim)
        nn.init.eye_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

    @property
    def neural_field(self):
        neural_field = self._neural_field
        if neural_field is None:
            raise RuntimeError("Structured neural field reference is unavailable.")
        return neural_field

    def _observed_point_tokens(self, batch: STBatch, obs_mask: torch.Tensor) -> torch.Tensor:
        field = self.neural_field
        spatial = expand_spatial_coords(batch)
        temporal = expand_time_coords(batch)
        rel = field.relative_time_tokens(batch.batch_size, batch.time_steps, batch.max_nodes)
        masked_values = batch.values * obs_mask[..., None]
        value = self.value_proj(masked_values) + self.mask_embed(obs_mask[..., None])
        return field.point_fuse(
            torch.cat([value, field.encode_space(spatial), field.encode_time(temporal), rel], dim=-1)
        )

    def _confidence(self, batch: STBatch, obs_mask: torch.Tensor) -> torch.Tensor:
        spatial = expand_spatial_coords(batch)
        anchors = self.neural_field.spatial_tokenizer.anchor_coords().to(spatial)
        distance_sq = (spatial[:, :, None, :, :] - anchors[None, None, :, None, :]).square().sum(dim=-1)
        weights = torch.exp(-distance_sq / max(2 * self.confidence_bandwidth**2, 1e-6))
        valid = torch.ones_like(obs_mask)
        if batch.node_mask is not None:
            valid = valid * batch.node_mask[:, None, :].to(valid.dtype)
        weights = weights * valid[:, :, None, :]
        numerator = (weights * obs_mask[:, :, None, :]).sum(dim=-1)
        denominator = weights.sum(dim=-1).clamp_min(1e-6)
        return (numerator / denominator).clamp(0, 1)

    def forward(self, batch: STBatch) -> StructuredConditionOutput:
        if batch.obs_mask is None:
            obs_mask = torch.ones(
                batch.batch_size,
                batch.time_steps,
                batch.max_nodes,
                device=batch.values.device,
                dtype=batch.values.dtype,
            )
        else:
            obs_mask = batch.obs_mask.to(batch.values.dtype)
        if batch.node_mask is not None:
            obs_mask = obs_mask * batch.node_mask[:, None, :].to(obs_mask.dtype)

        b, t, n = obs_mask.shape
        point_tokens = self._observed_point_tokens(batch, obs_mask)
        point_mask = obs_mask > 0
        all_missing = ~point_mask.any(dim=-1)
        safe_mask = point_mask.clone()
        safe_points = point_tokens.clone()
        if all_missing.any():
            # Flatten B/T explicitly so the dummy-key assignment is stable
            # across PyTorch advanced-indexing implementations.
            flat_all_missing = all_missing.reshape(b * t)
            flat_mask = safe_mask.reshape(b * t, n)
            flat_points = safe_points.reshape(b * t, n, -1)
            flat_mask[flat_all_missing, 0] = True
            flat_points[flat_all_missing, 0, :] = 0
        d = point_tokens.shape[-1]
        time_tokens = self.neural_field.time_latent_tokens(batch)
        slots = self.neural_field.spatial_tokenizer(
            safe_points.reshape(b * t, n, d),
            point_mask=safe_mask.reshape(b * t, n),
            time_tokens=time_tokens.reshape(b * t, d),
        ).reshape(b, t, -1, d)
        missing = self.missing_slots + time_tokens[:, :, None, :]
        slots = torch.where(all_missing[:, :, None, None], missing, slots)

        confidence = self._confidence(batch, obs_mask)
        valid_nodes = (
            batch.node_mask[:, None, :].to(obs_mask.dtype)
            if batch.node_mask is not None
            else torch.ones_like(obs_mask)
        )
        frame_ratio = obs_mask.sum(dim=-1) / valid_nodes.sum(dim=-1).clamp_min(1.0)
        frame_observed = frame_ratio > 0
        previous_distance, next_distance, has_previous, has_next = _frame_gap_features(frame_observed)
        gap = torch.stack(
            [
                confidence,
                frame_ratio[:, :, None].expand_as(confidence),
                previous_distance[:, :, None].expand_as(confidence),
                next_distance[:, :, None].expand_as(confidence),
                has_previous[:, :, None].expand_as(confidence),
                has_next[:, :, None].expand_as(confidence),
            ],
            dim=-1,
        )
        slots = slots + self.gap_feature_proj(gap)

        k = slots.shape[2]
        temporal = slots.permute(0, 2, 1, 3).reshape(b * k, t, d)
        directions, _ = self.bidirectional_bridge(temporal)
        forward, backward = directions.chunk(2, dim=-1)
        forward = forward.reshape(b, k, t, d).permute(0, 2, 1, 3)
        backward = backward.reshape(b, k, t, d).permute(0, 2, 1, 3)
        gate = self.direction_gate(gap)
        condition = gate * forward + (1 - gate) * backward
        for block in self.axial_blocks:
            condition = block(condition)
        condition = self.condition_norm(condition)
        mean_latent = self.mean_head(condition)
        return StructuredConditionOutput(
            mean_latent=mean_latent,
            condition_grid=condition,
            confidence=confidence,
            global_context=condition.mean(dim=(1, 2)),
            frame_observed_ratio=frame_ratio,
            previous_distance=previous_distance,
            next_distance=next_distance,
        )

def _slice_batch_rows(batch: STBatch, rows: torch.Tensor) -> STBatch:
    """Select batch rows while preserving scalar/domain metadata."""

    batch_size = batch.batch_size

    def select(value):
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            return value[rows]
        return value

    return STBatch(
        values=batch.values[rows],
        spatial_coords=select(batch.spatial_coords),
        time_coords=select(batch.time_coords),
        node_mask=select(batch.node_mask),
        obs_mask=select(batch.obs_mask),
        metadata={key: select(value) for key, value in batch.metadata.items()},
    )

def decode_neural_field(
    neural_field: nn.Module,
    latent: torch.Tensor,
    query: torch.Tensor,
    output_batch: Optional[STBatch] = None,
    condition_batch: Optional[STBatch] = None,
) -> torch.Tensor:
    """Decode with explicit geometry/context when the field supports it."""

    grounded_decode = getattr(neural_field, "decode_with_batch", None)
    if grounded_decode is not None:
        if output_batch is None:
            raise ValueError("A geometry-grounded decoder requires output_batch")
        return grounded_decode(
            latent,
            query,
            output_batch,
            condition_batch=condition_batch,
        )
    return neural_field.decode(latent, query)

class DirectConditionOutput:
    condition_grid: torch.Tensor
    raw_point_tokens: torch.Tensor
    raw_point_mask: torch.Tensor
    condition_features: torch.Tensor
    confidence: torch.Tensor
    global_context: torch.Tensor
    frame_observed_ratio: torch.Tensor
    previous_distance: torch.Tensor
    next_distance: torch.Tensor

class DirectDiffusionLossOutput:
    loss: torch.Tensor
    edm_loss: torch.Tensor
    decoded_loss: torch.Tensor
    condition_latent_loss: torch.Tensor
    condition_decoded_loss: torch.Tensor
    shuffle_loss: torch.Tensor
    condition_advantage: torch.Tensor
    shuffle_applied: torch.Tensor
    mean_sigma: torch.Tensor
    mean_confidence: torch.Tensor
    missing_confidence: torch.Tensor
    target_abs_mean: torch.Tensor
    task: str
    mean_uncertainty: Optional[torch.Tensor] = None

class DirectAlignedConditionEncoder(nn.Module):
    """Encode partial observations in the frozen Stage-1 coordinate system."""

    def __init__(
        self,
        config: ModelConfig,
        neural_field: nn.Module,
        temporal_layers: int = 2,
        axial_layers: int = 2,
        heads: int = 4,
        confidence_bandwidth: float = 0.25,
    ) -> None:
        super().__init__()
        self.config = config
        self.confidence_bandwidth = confidence_bandwidth
        object.__setattr__(self, "_neural_field", neural_field)

        # Keep the exact Stage-1 value/coordinate/fusion path as the base.  A
        # zero-initialized residual adapter learns only the partial-observation
        # correction, avoiding the old new-value-proj -> frozen-point-fuse
        # distribution mismatch.
        self.point_mask_adapter = nn.Sequential(
            nn.Linear(config.token_dim + 2, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, config.token_dim),
        )
        nn.init.zeros_(self.point_mask_adapter[-1].weight)
        nn.init.zeros_(self.point_mask_adapter[-1].bias)

        # Copy the Stage-1 tokenizer so its slot geometry is inherited while
        # its attention can adapt to sparse inputs.  Anchor and slot identity
        # stay fixed to the codec coordinate system.
        self.spatial_tokenizer = copy.deepcopy(neural_field.spatial_tokenizer)
        self.spatial_tokenizer.requires_grad_(True)
        num_slots = neural_field.num_spatial_latents
        if getattr(self.spatial_tokenizer, "requires_geometry_tokens", False):
            # Geometry-grounded tokenizers derive their slot identity from the
            # coordinates of the current domain, not from a learned [0,1]
            # anchor table.  Their missing-frame state is therefore supplied
            # separately and the shared geometry queries remain frozen in the
            # sparse conditioner.
            for name, parameter in self.spatial_tokenizer.named_parameters():
                if any(
                    identity_part in name
                    for identity_part in (
                        "geometry_queries",
                        "geometry_key",
                        "support",
                        "global_type",
                        "local_type",
                    )
                ):
                    parameter.requires_grad_(False)
            initial_missing_slots = torch.zeros(
                1, 1, num_slots, config.latent_dim
            )
        elif self._is_routed_tokenizer(self.spatial_tokenizer):
            # A routed tokenizer has separate invariant/candidate identities
            # and requires a domain descriptor to instantiate its active K
            # slots. Freeze the entire routing identity in the conditioner;
            # only its copied sparse-attention blocks may adapt.
            for name in (
                "global_anchor_logits",
                "candidate_anchor_logits",
                "global_queries",
                "candidate_queries",
                "candidate_keys",
                "router_bias",
            ):
                getattr(self.spatial_tokenizer, name).requires_grad_(False)
            self.spatial_tokenizer.router.requires_grad_(False)
            initial_missing_slots = torch.zeros(
                1, 1, num_slots, config.latent_dim
            )
        else:
            self.spatial_tokenizer.anchor_logits.requires_grad_(False)
            self.spatial_tokenizer.slot_queries.requires_grad_(False)
            with torch.no_grad():
                initial_missing_slots = (
                    self.spatial_tokenizer.initial_slots(1).unsqueeze(1).detach().clone()
                )
        self.missing_slots = nn.Parameter(initial_missing_slots)
        self.gap_feature_proj = nn.Sequential(
            nn.Linear(6, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        nn.init.zeros_(self.gap_feature_proj[-1].weight)
        nn.init.zeros_(self.gap_feature_proj[-1].bias)
        self.bidirectional_temporal = nn.GRU(
            config.latent_dim,
            config.latent_dim,
            num_layers=temporal_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if temporal_layers == 1 else 0.1,
        )
        self.direction_gate = nn.Sequential(nn.Linear(6, config.latent_dim), nn.Sigmoid())
        self.temporal_update_logit = nn.Parameter(torch.tensor(0.0))
        inherited_blocks = []
        for index in range(axial_layers):
            if index < len(neural_field.axial_blocks):
                block = copy.deepcopy(neural_field.axial_blocks[index])
                # Partial-window interpolation needs access beyond the local
                # Stage-1 radius; the attention weights themselves are reused.
                block.temporal_radius = None
            else:
                block = StructuredAxialBlock(config.latent_dim, heads)
            block.requires_grad_(True)
            inherited_blocks.append(block)
        self.axial_blocks = nn.ModuleList(inherited_blocks)
        self.out_norm = copy.deepcopy(neural_field.latent_norm)
        self.out_norm.requires_grad_(True)

    @staticmethod
    def _is_routed_tokenizer(tokenizer: nn.Module) -> bool:
        """Detect the routed parameter layout without relying on a new helper method."""

        return all(
            hasattr(tokenizer, name)
            for name in (
                "global_anchor_logits",
                "candidate_anchor_logits",
                "global_queries",
                "candidate_queries",
                "routing_logits",
                "_select_without_replacement",
            )
        )

    def _active_routed_anchors(self, routing_context: torch.Tensor) -> torch.Tensor:
        tokenizer = self.spatial_tokenizer
        helper = getattr(tokenizer, "active_anchor_coords", None)
        if helper is not None:
            return helper(routing_context)
        selection, _, _ = tokenizer._select_without_replacement(
            tokenizer.routing_logits(routing_context)
        )
        global_anchors = torch.sigmoid(tokenizer.global_anchor_logits)[None].expand(
            routing_context.shape[0], -1, -1
        )
        candidate_anchors = torch.sigmoid(tokenizer.candidate_anchor_logits)
        adaptive_anchors = torch.matmul(selection, candidate_anchors)
        return torch.cat([global_anchors, adaptive_anchors], dim=1)

    @property
    def neural_field(self):
        neural_field = self._neural_field
        if neural_field is None:
            raise RuntimeError("Structured neural field reference is unavailable.")
        return neural_field

    def _observed_point_tokens(
        self,
        batch: STBatch,
        obs_mask: torch.Tensor,
        frame_ratio: torch.Tensor,
    ) -> torch.Tensor:
        field = self.neural_field
        spatial = expand_spatial_coords(batch)
        temporal = expand_time_coords(batch)
        relative = field.relative_time_tokens(batch.batch_size, batch.time_steps, batch.max_nodes)
        masked_values = batch.values * obs_mask[..., None]
        base = field.point_fuse(
            torch.cat(
                [
                    field.value_proj(masked_values),
                    field.encode_space(spatial),
                    field.encode_time(temporal),
                    relative,
                ],
                dim=-1,
            )
        )
        status = torch.stack(
            [obs_mask, frame_ratio[:, :, None].expand_as(obs_mask)],
            dim=-1,
        )
        return base + self.point_mask_adapter(torch.cat([base, status], dim=-1))

    def _confidence(self, batch: STBatch, obs_mask: torch.Tensor) -> torch.Tensor:
        spatial = expand_spatial_coords(batch)
        if self._is_routed_tokenizer(self.spatial_tokenizer):
            routing_context = self.neural_field.domain_context(batch)
            anchors = self._active_routed_anchors(routing_context).to(spatial)
            distance_sq = (
                spatial[:, :, None, :, :] - anchors[:, None, :, None, :]
            ).square().sum(dim=-1)
        else:
            anchors = self.spatial_tokenizer.anchor_coords().to(spatial)
            distance_sq = (
                spatial[:, :, None, :, :] - anchors[None, None, :, None, :]
            ).square().sum(dim=-1)
        weights = torch.exp(-distance_sq / max(2 * self.confidence_bandwidth**2, 1e-6))
        valid = torch.ones_like(obs_mask)
        if batch.node_mask is not None:
            valid = valid * batch.node_mask[:, None, :].to(valid.dtype)
        weights = weights * valid[:, :, None, :]
        numerator = (weights * obs_mask[:, :, None, :]).sum(dim=-1)
        denominator = weights.sum(dim=-1).clamp_min(1e-6)
        return (numerator / denominator).clamp(0, 1)

    def _tokenizer_point_mask(
        self,
        batch: STBatch,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Points admitted to sparse tokenization; subclasses may include queries."""

        return obs_mask > 0

    def _uses_attention_confidence(self) -> bool:
        """Whether slot confidence comes from the tokenizer's point attention."""

        return False

    def forward(self, batch: STBatch) -> DirectConditionOutput:
        if batch.obs_mask is None:
            obs_mask = torch.ones(
                batch.batch_size,
                batch.time_steps,
                batch.max_nodes,
                device=batch.values.device,
                dtype=batch.values.dtype,
            )
        else:
            obs_mask = batch.obs_mask.to(batch.values.dtype)
        if batch.node_mask is not None:
            obs_mask = obs_mask * batch.node_mask[:, None, :].to(obs_mask.dtype)

        b, t, n = obs_mask.shape
        valid_nodes = (
            batch.node_mask[:, None, :].to(obs_mask.dtype)
            if batch.node_mask is not None
            else torch.ones_like(obs_mask)
        )
        frame_ratio = obs_mask.sum(dim=-1) / valid_nodes.sum(dim=-1).clamp_min(1.0)
        point_tokens = self._observed_point_tokens(batch, obs_mask, frame_ratio)
        raw_point_mask = obs_mask > 0
        tokenizer_point_mask = self._tokenizer_point_mask(batch, obs_mask).to(torch.bool)
        no_token_points = ~tokenizer_point_mask.any(dim=-1)
        safe_mask = tokenizer_point_mask.clone()
        safe_points = point_tokens.clone()
        if no_token_points.any():
            flat_missing = no_token_points.reshape(b * t)
            flat_mask = safe_mask.reshape(b * t, n)
            flat_points = safe_points.reshape(b * t, n, -1)
            flat_mask[flat_missing, 0] = True
            flat_points[flat_missing, 0, :] = 0

        d = point_tokens.shape[-1]
        time_tokens = self.neural_field.time_latent_tokens(batch)
        tokenizer_kwargs = {}
        if getattr(self.spatial_tokenizer, "requires_geometry_tokens", False):
            geometry_tokens, geometry_mask = self.neural_field.geometry_tokens_and_mask(batch)
            tokenizer_kwargs["geometry_tokens"] = geometry_tokens.reshape(b * t, n, d)
            tokenizer_kwargs["geometry_mask"] = geometry_mask.reshape(b * t, n)
            if getattr(self.spatial_tokenizer, "requires_metric_coords", False):
                metric_coords = self.neural_field.metric_attention_coords(batch)
                tokenizer_kwargs["metric_coords"] = metric_coords.reshape(
                    b * t, n, metric_coords.shape[-1]
                )
        elif self._is_routed_tokenizer(self.spatial_tokenizer):
            routing_context = self.neural_field.domain_context(batch)
            tokenizer_kwargs["routing_context"] = (
                routing_context[:, None, :].expand(-1, t, -1).reshape(b * t, d)
            )
        tokenizer_input = safe_points.reshape(b * t, n, d)
        tokenizer_mask = safe_mask.reshape(b * t, n)
        if self._uses_attention_confidence():
            slots, cross_attention = self.spatial_tokenizer(
                tokenizer_input,
                point_mask=tokenizer_mask,
                time_tokens=time_tokens.reshape(b * t, d),
                return_cross_attention=True,
                **tokenizer_kwargs,
            )
            k = slots.shape[1]
            # Confidence is an observation-coverage statistic, not an extra
            # trainable side channel. The attention still learns normally
            # through `slots`; only this diagnostic/conditioning branch is
            # detached so it cannot be gamed to encode arbitrary features.
            confidence = (
                cross_attention.detach().reshape(b, t, k, n)
                * obs_mask[:, :, None, :]
            ).sum(dim=-1).clamp(0, 1)
        else:
            slots = self.spatial_tokenizer(
                tokenizer_input,
                point_mask=tokenizer_mask,
                time_tokens=time_tokens.reshape(b * t, d),
                **tokenizer_kwargs,
            )
            confidence = self._confidence(batch, obs_mask)
        slots = slots.reshape(b, t, -1, d)
        missing_slots = self.missing_slots + time_tokens[:, :, None, :]
        slots = torch.where(no_token_points[:, :, None, None], missing_slots, slots)

        frame_observed = frame_ratio > 0
        previous_distance, next_distance, has_previous, has_next = _frame_gap_features(frame_observed)
        gap = torch.stack(
            [
                confidence,
                frame_ratio[:, :, None].expand_as(confidence),
                previous_distance[:, :, None].expand_as(confidence),
                next_distance[:, :, None].expand_as(confidence),
                has_previous[:, :, None].expand_as(confidence),
                has_next[:, :, None].expand_as(confidence),
            ],
            dim=-1,
        )
        slots = slots + self.gap_feature_proj(gap)

        k = slots.shape[2]
        temporal = slots.permute(0, 2, 1, 3).reshape(b * k, t, d)
        directions, _ = self.bidirectional_temporal(temporal)
        forward, backward = directions.chunk(2, dim=-1)
        forward = forward.reshape(b, k, t, d).permute(0, 2, 1, 3)
        backward = backward.reshape(b, k, t, d).permute(0, 2, 1, 3)
        gate = self.direction_gate(gap)
        temporal_update = gate * forward + (1 - gate) * backward
        condition = slots + torch.sigmoid(self.temporal_update_logit) * temporal_update
        for block in self.axial_blocks:
            condition = block(condition)
        condition = self.out_norm(condition)
        return DirectConditionOutput(
            condition_grid=condition,
            raw_point_tokens=point_tokens,
            raw_point_mask=raw_point_mask,
            condition_features=gap,
            confidence=confidence,
            global_context=condition.mean(dim=(1, 2)),
            frame_observed_ratio=frame_ratio,
            previous_distance=previous_distance,
            next_distance=next_distance,
        )

    @torch.no_grad()
    def condition_path_stats(self) -> dict[str, float]:
        return {
            "temporal_update_scale": float(torch.sigmoid(self.temporal_update_logit)),
            "point_adapter_weight_norm": float(self.point_mask_adapter[-1].weight.norm()),
            "gap_adapter_weight_norm": float(self.gap_feature_proj[-1].weight.norm()),
        }

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]

class DirectConditionedBlock(nn.Module):
    """Axial denoising block with explicit structured and raw cross-attention."""

    def __init__(
        self,
        width: int,
        heads: int,
        dropout: float = 0.0,
        use_raw_cross: bool = True,
    ) -> None:
        super().__init__()
        self.use_raw_cross = use_raw_cross
        self.temporal_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.temporal_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.spatial_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)

        self.temporal_cross_query_norm = nn.LayerNorm(width)
        self.temporal_cross_context_norm = nn.LayerNorm(width)
        self.temporal_cross_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.spatial_cross_query_norm = nn.LayerNorm(width)
        self.spatial_cross_context_norm = nn.LayerNorm(width)
        self.spatial_cross_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.raw_cross_query_norm = nn.LayerNorm(width)
        self.raw_cross_context_norm = nn.LayerNorm(width)
        self.raw_cross_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)

        self.ff_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

        # sigmoid(-1.386) ~= 0.2: condition paths are active from step one but
        # do not overwhelm the noisy latent representation.
        initial_cross_logit = torch.tensor(-1.3862944)
        self.temporal_condition_logit = nn.Parameter(initial_cross_logit.clone())
        self.spatial_condition_logit = nn.Parameter(initial_cross_logit.clone())
        self.raw_condition_logit = nn.Parameter(initial_cross_logit.clone())

    def forward(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor,
        aligned_condition: torch.Tensor,
        confidence: torch.Tensor,
        raw_points: torch.Tensor,
        raw_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_t, scale_t, shift_s, scale_s, shift_ff, scale_ff = self.modulation(global_cond).chunk(6, dim=-1)
        b, t, k, width = x.shape

        temporal_grid = _modulate(self.temporal_norm(x), shift_t, scale_t)
        temporal = temporal_grid.permute(0, 2, 1, 3).reshape(b * k, t, width)
        temporal = self.temporal_attn(temporal, temporal, temporal, need_weights=False)[0]
        x = x + temporal.reshape(b, k, t, width).permute(0, 2, 1, 3)

        spatial_grid = _modulate(self.spatial_norm(x), shift_s, scale_s).reshape(b * t, k, width)
        spatial = self.spatial_attn(spatial_grid, spatial_grid, spatial_grid, need_weights=False)[0]
        x = x + spatial.reshape(b, t, k, width)

        # Confidence softly suppresses synthesized/missing condition positions
        # without making them inaccessible as a key/value memory.
        weighted_condition = aligned_condition * (0.25 + 0.75 * confidence[..., None])
        temporal_query = self.temporal_cross_query_norm(x).permute(0, 2, 1, 3).reshape(b * k, t, width)
        temporal_context = self.temporal_cross_context_norm(weighted_condition).permute(0, 2, 1, 3).reshape(
            b * k, t, width
        )
        temporal_update = self.temporal_cross_attn(
            temporal_query,
            temporal_context,
            temporal_context,
            need_weights=False,
        )[0]
        temporal_update = temporal_update.reshape(b, k, t, width).permute(0, 2, 1, 3)
        x = x + torch.sigmoid(self.temporal_condition_logit) * temporal_update

        spatial_query = self.spatial_cross_query_norm(x).reshape(b * t, k, width)
        spatial_context = self.spatial_cross_context_norm(weighted_condition).reshape(b * t, k, width)
        spatial_update = self.spatial_cross_attn(
            spatial_query,
            spatial_context,
            spatial_context,
            need_weights=False,
        )[0]
        x = x + torch.sigmoid(self.spatial_condition_logit) * spatial_update.reshape(b, t, k, width)

        # Per-frame raw observation memory preserves physical-node information
        # that can be lost when N points are compressed into K slots.
        if self.use_raw_cross:
            _, _, n, _ = raw_points.shape
            raw_context = self.raw_cross_context_norm(raw_points).reshape(b * t, n, width)
            raw_key_mask = raw_mask.reshape(b * t, n).to(torch.bool)
            frame_observed = raw_key_mask.any(dim=-1)
            safe_mask = raw_key_mask.clone()
            safe_context = raw_context.clone()
            if (~frame_observed).any():
                safe_mask[~frame_observed, 0] = True
                safe_context[~frame_observed, 0] = 0
            raw_query = self.raw_cross_query_norm(x).reshape(b * t, k, width)
            raw_update = self.raw_cross_attn(
                raw_query,
                safe_context,
                safe_context,
                key_padding_mask=~safe_mask,
                need_weights=False,
            )[0]
            raw_update = raw_update * frame_observed[:, None, None].to(raw_update.dtype)
            x = x + torch.sigmoid(self.raw_condition_logit) * raw_update.reshape(b, t, k, width)

        return x + self.ff(_modulate(self.ff_norm(x), shift_ff, scale_ff))

class DirectConditionedDenoiser(nn.Module):
    """Direct EDM backbone whose condition routes are non-zero at initialization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        width = config.diffusion_width
        self.input_proj = nn.Linear(config.latent_dim, width)
        self.condition_proj = nn.Linear(config.latent_dim, width)
        self.raw_point_proj = nn.Linear(config.token_dim, width)
        self.condition_feature_proj = nn.Sequential(nn.Linear(6, width), nn.SiLU(), nn.Linear(width, width))
        self.global_condition_proj = nn.Linear(config.latent_dim, width)
        self.sigma_embed = SinusoidalEmbedding(width)
        self.input_condition_logit = nn.Parameter(torch.tensor(0.0))
        self.blocks = nn.ModuleList(
            [
                DirectConditionedBlock(
                    width,
                    config.diffusion_heads,
                    config.diffusion_dropout,
                    use_raw_cross=index % 2 == 0,
                )
                for index in range(config.diffusion_depth)
            ]
        )
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, config.latent_dim)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        noise_embedding: torch.Tensor,
        condition: DirectConditionOutput,
    ) -> torch.Tensor:
        aligned = self.condition_proj(condition.condition_grid)
        features = self.condition_feature_proj(condition.condition_features)
        raw_points = self.raw_point_proj(condition.raw_point_tokens)
        h = self.input_proj(noisy_latent)
        h = h + torch.sigmoid(self.input_condition_logit) * (aligned + features)
        global_cond = self.sigma_embed(noise_embedding) + self.global_condition_proj(condition.global_context)
        for block in self.blocks:
            h = block(
                h,
                global_cond,
                aligned,
                condition.confidence,
                raw_points,
                condition.raw_point_mask,
            )
        return self.out(self.out_norm(h))

    @torch.no_grad()
    def condition_path_stats(self) -> dict[str, float]:
        return {
            "input_condition_scale": float(torch.sigmoid(self.input_condition_logit)),
            "temporal_cross_scale": float(
                torch.stack([torch.sigmoid(block.temporal_condition_logit) for block in self.blocks]).mean()
            ),
            "spatial_cross_scale": float(
                torch.stack([torch.sigmoid(block.spatial_condition_logit) for block in self.blocks]).mean()
            ),
            "raw_cross_scale": float(
                torch.stack(
                    [torch.sigmoid(block.raw_condition_logit) for block in self.blocks if block.use_raw_cross]
                ).mean()
            ),
            "output_weight_norm": float(self.out.weight.norm()),
        }

class StructuredDirectLatentDiffusion(nn.Module):
    """EDM that generates the complete normalized structured latent directly."""

    def __init__(
        self,
        config: ModelConfig,
        neural_field: nn.Module,
        condition_temporal_layers: int = 2,
        condition_axial_layers: int = 2,
        confidence_bandwidth: float = 0.25,
        missing_loss_weight: float = 2.0,
        condition_dropout: float = 0.1,
        guidance_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config
        object.__setattr__(self, "_neural_field", neural_field)
        self.condition_encoder = DirectAlignedConditionEncoder(
            config,
            neural_field,
            temporal_layers=condition_temporal_layers,
            axial_layers=condition_axial_layers,
            heads=config.self_heads,
            confidence_bandwidth=confidence_bandwidth,
        )
        self.backbone = DirectConditionedDenoiser(config)
        self.condition_probe = nn.Linear(config.latent_dim, config.latent_dim)
        nn.init.eye_(self.condition_probe.weight)
        nn.init.zeros_(self.condition_probe.bias)
        self.missing_loss_weight = missing_loss_weight
        self.condition_dropout = condition_dropout
        self.guidance_scale = guidance_scale
        shape = (1, config.time_steps, neural_field.num_spatial_latents, config.latent_dim)
        # These are fixed training-set standardization statistics, not a
        # condition-dependent mean-latent predictor/bridge.
        self.register_buffer("latent_mean", torch.zeros(shape))
        self.register_buffer("latent_std", torch.ones(shape))

    @property
    def neural_field(self):
        return self._neural_field

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.latent_mean.copy_(mean.view_as(self.latent_mean))
        self.latent_std.copy_(std.view_as(self.latent_std).clamp_min(1e-6))

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / self.latent_std

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.latent_std + self.latent_mean

    @staticmethod
    def _roll_condition(condition: DirectConditionOutput) -> DirectConditionOutput:
        return DirectConditionOutput(
            condition_grid=torch.roll(condition.condition_grid, 1, 0),
            raw_point_tokens=torch.roll(condition.raw_point_tokens, 1, 0),
            raw_point_mask=torch.roll(condition.raw_point_mask, 1, 0),
            condition_features=torch.roll(condition.condition_features, 1, 0),
            confidence=torch.roll(condition.confidence, 1, 0),
            global_context=torch.roll(condition.global_context, 1, 0),
            frame_observed_ratio=torch.roll(condition.frame_observed_ratio, 1, 0),
            previous_distance=torch.roll(condition.previous_distance, 1, 0),
            next_distance=torch.roll(condition.next_distance, 1, 0),
        )

    @staticmethod
    def _null_condition(condition: DirectConditionOutput) -> DirectConditionOutput:
        return DirectConditionOutput(
            condition_grid=torch.zeros_like(condition.condition_grid),
            raw_point_tokens=torch.zeros_like(condition.raw_point_tokens),
            raw_point_mask=torch.zeros_like(condition.raw_point_mask),
            condition_features=torch.zeros_like(condition.condition_features),
            confidence=torch.zeros_like(condition.confidence),
            global_context=torch.zeros_like(condition.global_context),
            frame_observed_ratio=torch.zeros_like(condition.frame_observed_ratio),
            previous_distance=torch.zeros_like(condition.previous_distance),
            next_distance=torch.zeros_like(condition.next_distance),
        )

    def _drop_condition_samples(self, condition: DirectConditionOutput) -> DirectConditionOutput:
        if not self.training or self.condition_dropout <= 0:
            return condition
        keep = torch.rand(
            condition.condition_grid.shape[0],
            device=condition.condition_grid.device,
        ) >= self.condition_dropout

        def masked(tensor: torch.Tensor) -> torch.Tensor:
            view = keep.view(keep.shape[0], *([1] * (tensor.ndim - 1)))
            if tensor.dtype == torch.bool:
                return tensor & view
            return tensor * view.to(tensor.dtype)

        return DirectConditionOutput(
            condition_grid=masked(condition.condition_grid),
            raw_point_tokens=masked(condition.raw_point_tokens),
            raw_point_mask=masked(condition.raw_point_mask),
            condition_features=masked(condition.condition_features),
            confidence=masked(condition.confidence),
            global_context=masked(condition.global_context),
            frame_observed_ratio=masked(condition.frame_observed_ratio),
            previous_distance=masked(condition.previous_distance),
            next_distance=masked(condition.next_distance),
        )

    @staticmethod
    def sigma_to_noise_embedding(sigma: torch.Tensor) -> torch.Tensor:
        return sigma.log() / 4

    def denoise(
        self,
        noisy_latent: torch.Tensor,
        sigma: torch.Tensor,
        condition: DirectConditionOutput,
    ) -> torch.Tensor:
        sigma = sigma.to(noisy_latent.dtype)
        sigma_data = self.config.sigma_data
        c_skip = sigma_data**2 / (sigma.square() + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma.square() + sigma_data**2)
        c_in = 1 / torch.sqrt(sigma.square() + sigma_data**2)
        model_input = append_dims(c_in, noisy_latent.ndim) * noisy_latent
        model_output = self.backbone(model_input, self.sigma_to_noise_embedding(sigma), condition)
        return append_dims(c_skip, noisy_latent.ndim) * noisy_latent + append_dims(
            c_out, noisy_latent.ndim
        ) * model_output

    @staticmethod
    def _masked_value_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denominator = mask.sum().clamp_min(1.0) * target.shape[-1]
        return ((prediction - target).abs() * mask[..., None]).sum() / denominator

    def loss_for_task(
        self,
        target_latent: torch.Tensor,
        full_batch: STBatch,
        task: STTask,
        task_config: CurrentConditionalTaskConfig,
        generator: Optional[torch.Generator] = None,
        decoded_loss_weight: float = 0.05,
        condition_latent_loss_weight: float = 0.25,
        condition_decoded_loss_weight: float = 0.05,
        shuffle_loss_weight: float = 0.05,
        shuffle_margin: float = 0.05,
        compute_shuffle_loss: bool = False,
        target_logvar: Optional[torch.Tensor] = None,
    ) -> DirectDiffusionLossOutput:
        condition_batch = build_condition_view(full_batch, task, task_config, generator=generator)
        condition = self.condition_encoder(condition_batch)
        target_normalized = self.normalize(target_latent)
        latent_weight = 1.0 + self.missing_loss_weight * (1.0 - condition.confidence)

        rnd_normal = torch.randn(
            target_latent.shape[0], device=target_latent.device, dtype=target_latent.dtype
        )
        sigma = (rnd_normal * self.config.p_std + self.config.p_mean).exp()
        noisy = target_normalized + torch.randn_like(target_normalized) * append_dims(
            sigma, target_normalized.ndim
        )
        denoise_condition = condition if compute_shuffle_loss else self._drop_condition_samples(condition)
        denoised = self.denoise(noisy, sigma, denoise_condition)
        edm_weight = (sigma.square() + self.config.sigma_data**2) / (
            sigma * self.config.sigma_data
        ).square()
        edm_element = (denoised - target_normalized).square().mean(dim=-1)
        weighted = append_dims(edm_weight, edm_element.ndim) * edm_element * latent_weight
        edm_loss = weighted.sum() / latent_weight.sum().clamp_min(1.0)

        valid = torch.ones_like(condition_batch.obs_mask)
        if full_batch.node_mask is not None:
            valid = valid * full_batch.node_mask[:, None, :].to(valid.dtype)
        natural_valid = full_batch.metadata.get("valid_mask")
        if natural_valid is not None:
            valid = valid * natural_valid.to(valid.dtype)
        missing_mask = valid * (1.0 - condition_batch.obs_mask)
        low_sigma = sigma <= 1.0
        if low_sigma.any() and decoded_loss_weight > 0:
            predicted_latent = self.denormalize(denoised[low_sigma])
            query = build_full_query_coords(full_batch)[low_sigma]
            selected_full = _slice_batch_rows(full_batch, low_sigma)
            selected_condition = _slice_batch_rows(condition_batch, low_sigma)
            prediction = decode_neural_field(
                self.neural_field,
                predicted_latent,
                query,
                selected_full,
                selected_condition,
            )
            decoded_loss = self._masked_value_l1(
                prediction,
                full_batch.values[low_sigma],
                missing_mask[low_sigma],
            )
        else:
            decoded_loss = target_latent.new_zeros(())

        # Supervised representation probe. It directly prevents condition
        # collapse and supplies the centre used by anchored evaluators, while
        # leaving the EDM target itself as the complete normalized latent.
        probed_latent = self.condition_probe(condition.condition_grid)
        condition_latent_element = F.smooth_l1_loss(
            probed_latent,
            target_latent,
            reduction="none",
        ).mean(dim=-1)
        condition_latent_loss = (condition_latent_element * latent_weight).sum() / latent_weight.sum().clamp_min(1.0)
        if condition_decoded_loss_weight > 0:
            condition_prediction = decode_neural_field(
                self.neural_field,
                probed_latent,
                build_full_query_coords(full_batch),
                full_batch,
                condition_batch,
            )
            condition_decoded_loss = self._masked_value_l1(
                condition_prediction,
                full_batch.values,
                missing_mask,
            )
        else:
            condition_decoded_loss = target_latent.new_zeros(())

        shuffle_loss = target_latent.new_zeros(())
        condition_advantage = target_latent.new_zeros(())
        shuffle_applied = target_latent.new_zeros(())
        if compute_shuffle_loss and shuffle_loss_weight > 0 and target_latent.shape[0] > 1:
            shuffled_denoised = self.denoise(noisy, sigma, self._roll_condition(condition))
            correct_error = (denoised - target_normalized).square().mean(dim=(1, 2, 3))
            shuffled_error = (shuffled_denoised - target_normalized).square().mean(dim=(1, 2, 3))
            condition_advantage = (shuffled_error - correct_error).mean()
            shuffle_loss = F.relu(shuffle_margin + correct_error - shuffled_error).mean()
            shuffle_applied = target_latent.new_ones(())

        missing_latent = condition.confidence < 0.5
        missing_confidence = (
            condition.confidence[missing_latent].mean()
            if missing_latent.any()
            else condition.confidence.new_ones(())
        )
        return DirectDiffusionLossOutput(
            loss=(
                edm_loss
                + decoded_loss_weight * decoded_loss
                + condition_latent_loss_weight * condition_latent_loss
                + condition_decoded_loss_weight * condition_decoded_loss
                + shuffle_loss_weight * shuffle_loss
            ),
            edm_loss=edm_loss,
            decoded_loss=decoded_loss,
            condition_latent_loss=condition_latent_loss,
            condition_decoded_loss=condition_decoded_loss,
            shuffle_loss=shuffle_loss,
            condition_advantage=condition_advantage,
            shuffle_applied=shuffle_applied,
            mean_sigma=sigma.mean(),
            mean_confidence=condition.confidence.mean(),
            missing_confidence=missing_confidence,
            target_abs_mean=target_normalized.abs().mean(),
            task=task.value,
        )

    def karras_schedule(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        num_steps = num_steps or self.config.sampling_steps
        rho = self.config.rho
        indices = torch.arange(num_steps, device=device, dtype=dtype)
        steps = (
            self.config.sigma_max ** (1 / rho)
            + indices / max(num_steps - 1, 1)
            * (self.config.sigma_min ** (1 / rho) - self.config.sigma_max ** (1 / rho))
        ) ** rho
        return torch.cat([steps, torch.zeros_like(steps[:1])])

    @torch.no_grad()
    def sample_conditional(
        self,
        condition_batch: STBatch,
        final_query_coords: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, DirectConditionOutput]:
        condition = self.condition_encoder(condition_batch)
        null_condition = self._null_condition(condition) if self.guidance_scale != 1.0 else None
        shape = (
            condition_batch.batch_size,
            self.config.time_steps,
            self.neural_field.num_spatial_latents,
            self.config.latent_dim,
        )
        x_next = torch.randn(shape, device=condition_batch.values.device, dtype=condition_batch.values.dtype)
        x_next = x_next * self.config.sigma_max
        steps = self.karras_schedule(x_next.device, x_next.dtype, num_steps=num_steps)
        for index, (current, following) in enumerate(zip(steps[:-1], steps[1:])):
            sigma = x_next.new_full((x_next.shape[0],), current)
            denoised = self.denoise(x_next, sigma, condition)
            if null_condition is not None:
                unconditional = self.denoise(x_next, sigma, null_condition)
                denoised = unconditional + self.guidance_scale * (denoised - unconditional)
            derivative = (x_next - denoised) / current.clamp_min(self.config.sigma_min)
            euler = x_next + (following - current) * derivative
            if index < len(steps) - 2 and following > 0:
                next_sigma = x_next.new_full((x_next.shape[0],), following)
                next_denoised = self.denoise(euler, next_sigma, condition)
                if null_condition is not None:
                    next_unconditional = self.denoise(euler, next_sigma, null_condition)
                    next_denoised = next_unconditional + self.guidance_scale * (
                        next_denoised - next_unconditional
                    )
                next_derivative = (euler - next_denoised) / following.clamp_min(
                    self.config.sigma_min
                )
                x_next = x_next + 0.5 * (following - current) * (
                    derivative + next_derivative
                )
            else:
                x_next = euler
        final_latent = self.denormalize(x_next)
        query = (
            build_full_query_coords(condition_batch)
            if final_query_coords is None
            else final_query_coords
        )
        return (
            final_latent,
            decode_neural_field(
                self.neural_field,
                final_latent,
                query,
                condition_batch,
                condition_batch,
            ),
            condition,
        )

class StructuredDirectGenerator:
    def __init__(self, neural_field: nn.Module, diffusion: StructuredDirectLatentDiffusion) -> None:
        self.neural_field = neural_field
        self.diffusion = diffusion

    @torch.no_grad()
    def generate_from_batch(self, condition_batch: STBatch, num_steps: Optional[int] = None):
        context_batch = build_condition_batch_from_metadata(condition_batch)
        observed_batch = merge_condition_batches(condition_batch, context_batch)
        final_query = build_full_query_coords(condition_batch)
        latent, generated, diagnostics = self.diffusion.sample_conditional(
            observed_batch,
            final_query_coords=final_query,
            num_steps=num_steps,
        )
        if condition_batch.obs_mask is not None:
            generated = (
                condition_batch.obs_mask[..., None] * condition_batch.values
                + (1 - condition_batch.obs_mask[..., None]) * generated
            )
        return latent, generated, condition_batch.metadata.get("eval_mask"), diagnostics

class CanonicalDomainConditionEncoder(DirectAlignedConditionEncoder):
    """Single shared sparse conditioner augmented by domain context."""

    def __init__(
        self,
        config: ModelConfig,
        neural_field: CanonicalDomainStructuredField,
        domain_feature_dim: int,
        **kwargs,
    ) -> None:
        super().__init__(config, neural_field, **kwargs)
        self.domain_feature_dim = int(domain_feature_dim)
        self.domain_encoder = nn.Sequential(
            nn.LayerNorm(domain_feature_dim),
            nn.Linear(domain_feature_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.grid_adapter = nn.Linear(config.latent_dim, config.latent_dim)
        self.point_adapter = nn.Linear(config.latent_dim, config.token_dim)
        self.global_adapter = nn.Linear(config.latent_dim, config.latent_dim)
        for module in (self.grid_adapter, self.point_adapter, self.global_adapter):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, batch: STBatch) -> DirectConditionOutput:
        # Experimental metric-coordinate fields select a homogeneous-domain
        # coordinate transform before the inherited conditioner calls
        # neural_field.encode_space.  The canonical A field has no such method
        # and retains its original behavior.
        activate = getattr(self.neural_field, "activate_batch_domain", None)
        if activate is not None:
            activate(batch)
        output = super().forward(batch)
        features = domain_features_from_batch(batch, self.domain_feature_dim)
        normalize_features = getattr(self.neural_field, "normalize_domain_features", None)
        if normalize_features is not None:
            features = normalize_features(features)
        domain = self.domain_encoder(features)
        grid = output.condition_grid + self.grid_adapter(domain)[:, None, None, :]
        raw = output.raw_point_tokens + self.point_adapter(domain)[:, None, None, :]
        global_context = grid.mean(dim=(1, 2)) + self.global_adapter(domain)
        return DirectConditionOutput(
            condition_grid=grid,
            raw_point_tokens=raw,
            raw_point_mask=output.raw_point_mask,
            condition_features=output.condition_features,
            confidence=output.confidence,
            global_context=global_context,
            frame_observed_ratio=output.frame_observed_ratio,
            previous_distance=output.previous_distance,
            next_distance=output.next_distance,
        )

    @torch.no_grad()
    def condition_path_stats(self) -> dict[str, float]:
        stats = super().condition_path_stats()
        stats.update(
            {
                "domain_grid_adapter_norm": float(self.grid_adapter.weight.norm()),
                "domain_point_adapter_norm": float(self.point_adapter.weight.norm()),
                "domain_global_adapter_norm": float(self.global_adapter.weight.norm()),
            }
        )
        return stats

class FoundationAlignedConditionEncoder(CanonicalDomainConditionEncoder):
    """Foundation conditioner aligned with the frozen Stage-1 feature paths.

    Observed values use the exact Stage-1 value projection. Missing values are
    represented by one learned token, while spatial, temporal, relative-time,
    and continuous-domain encodings are shared with Stage 1. All valid query
    points participate in spatial tokenization, so the missing-value token is
    an actual input instead of an unused placeholder.
    """

    def __init__(
        self,
        config: ModelConfig,
        neural_field: AdaptiveGlobalLocalStructuredField,
        domain_feature_dim: int,
        **kwargs,
    ) -> None:
        super().__init__(config, neural_field, domain_feature_dim, **kwargs)
        reference = next(neural_field.parameters())
        with torch.no_grad():
            zero_value = torch.zeros(
                1,
                config.value_dim,
                device=reference.device,
                dtype=reference.dtype,
            )
            initial = neural_field.value_proj(zero_value).view(1, 1, 1, config.token_dim)
        self.unknown_value_embedding = nn.Parameter(initial.detach().clone())

    def _observed_point_tokens(
        self,
        batch: STBatch,
        obs_mask: torch.Tensor,
        frame_ratio: torch.Tensor,
    ) -> torch.Tensor:
        field = self.neural_field
        spatial = expand_spatial_coords(batch)
        temporal = expand_time_coords(batch)
        relative = field.relative_time_tokens(
            batch.batch_size,
            batch.time_steps,
            batch.max_nodes,
        )
        # Mask before the frozen value MLP so unavailable ground-truth values
        # are never evaluated on the condition path, even though torch.where
        # would discard that branch afterwards.
        observed_value_tokens = field.value_proj(batch.values * obs_mask[..., None])
        unknown_value_tokens = self.unknown_value_embedding.to(observed_value_tokens).expand_as(
            observed_value_tokens
        )
        value_tokens = torch.where(
            obs_mask[..., None] > 0,
            observed_value_tokens,
            unknown_value_tokens,
        )
        base = field.point_fuse(
            torch.cat(
                [
                    value_tokens,
                    field.encode_space(spatial),
                    field.encode_time(temporal),
                    relative,
                ],
                dim=-1,
            )
        )
        # CanonicalDomainStructuredField.point_tokens adds this after the
        # shared point_fuse path. Keep the conditioner in the same space.
        field_domain = field.domain_to_point(field.domain_context(batch))[:, None, None, :]
        base = base + field_domain
        status = torch.stack(
            [obs_mask, frame_ratio[:, :, None].expand_as(obs_mask)],
            dim=-1,
        )
        return base + self.point_mask_adapter(torch.cat([base, status], dim=-1))

    def _tokenizer_point_mask(
        self,
        batch: STBatch,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = torch.ones_like(obs_mask, dtype=torch.bool)
        if batch.node_mask is not None:
            valid = valid & (batch.node_mask[:, None, :] > 0)
        natural_valid = batch.metadata.get("valid_mask")
        if natural_valid is not None:
            valid = valid & natural_valid.to(torch.bool)
        return valid

    def _uses_attention_confidence(self) -> bool:
        # Confidence is the final tokenizer cross-attention mass assigned to
        # observed points. This uses the same Stage-1-aligned tokens and avoids
        # comparing raw lon/lat degrees with abstract [0,1] slot anchors.
        return True

    def forward(self, batch: STBatch) -> DirectConditionOutput:
        self.neural_field.activate_batch_geometry(batch)
        output = DirectAlignedConditionEncoder.forward(self, batch)

        # Stage 1 applies continuous-domain FiLM after latent normalization.
        # Apply the same frozen affine map before the condition-only adapter.
        field_domain = self.neural_field.domain_context(batch)
        shift, scale = self.neural_field.domain_film(field_domain).chunk(2, dim=-1)
        aligned_grid = output.condition_grid * (
            1.0 + 0.1 * torch.tanh(scale[:, None, None, :])
        ) + 0.1 * shift[:, None, None, :]

        features = domain_features_from_batch(batch, self.domain_feature_dim)
        normalize_features = getattr(self.neural_field, "normalize_domain_features", None)
        if normalize_features is not None:
            features = normalize_features(features)
        condition_domain = self.domain_encoder(features)
        grid = aligned_grid + self.grid_adapter(condition_domain)[:, None, None, :]
        raw = output.raw_point_tokens + self.point_adapter(condition_domain)[:, None, None, :]
        global_context = grid.mean(dim=(1, 2)) + self.global_adapter(condition_domain)
        return DirectConditionOutput(
            condition_grid=grid,
            raw_point_tokens=raw,
            raw_point_mask=output.raw_point_mask,
            condition_features=output.condition_features,
            confidence=output.confidence,
            global_context=global_context,
            frame_observed_ratio=output.frame_observed_ratio,
            previous_distance=output.previous_distance,
            next_distance=output.next_distance,
        )

class FoundationGlobalDirectDiffusion(StructuredDirectLatentDiffusion):
    """Direct EDM over raw Stage-1 latents with continuous domain descriptors."""

    def __init__(
        self,
        config: ModelConfig,
        neural_field: AdaptiveGlobalLocalStructuredField,
        domain_feature_dim: int,
        condition_temporal_layers: int = 2,
        condition_axial_layers: int = 2,
        confidence_bandwidth: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(
            config,
            neural_field,
            condition_temporal_layers=condition_temporal_layers,
            condition_axial_layers=condition_axial_layers,
            confidence_bandwidth=confidence_bandwidth,
            **kwargs,
        )
        self.condition_encoder = FoundationAlignedConditionEncoder(
            config,
            neural_field,
            domain_feature_dim,
            temporal_layers=condition_temporal_layers,
            axial_layers=condition_axial_layers,
            heads=config.self_heads,
            confidence_bandwidth=confidence_bandwidth,
        )

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Retain identity buffers solely for state-dict compatibility.

        Stage-2 consumes the Stage-1 latent tensor directly.  In particular,
        loading an older checkpoint cannot silently re-enable its fitted
        centering/scaling statistics.
        """

        self.latent_mean.zero_()
        self.latent_std.fill_(1.0)

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def activate_batch_domain(self, batch: STBatch) -> int:
        """Compatibility hook without a domain table or statistics switch."""

        self.neural_field.activate_batch_geometry(batch)
        return 0

class RawLatentFoundationDirectDiffusion(FoundationGlobalDirectDiffusion):
    """Foundation EDM operating directly on Stage-1 latents with no normalization."""

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        # Keep inherited buffers at identity only for state-dict compatibility.
        # They are deliberately not used by normalize/denormalize below.
        self.latent_mean.zero_()
        self.latent_std.fill_(1.0)

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

class ValueCanonicalConditionEncoder(FoundationAlignedConditionEncoder):
    """Stage-1-aligned sparse encoder without domain descriptor adapters."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for module in (
            self.domain_encoder,
            self.grid_adapter,
            self.point_adapter,
            self.global_adapter,
        ):
            module.requires_grad_(False)

    def _observed_point_tokens(
        self,
        batch: STBatch,
        obs_mask: torch.Tensor,
        frame_ratio: torch.Tensor,
    ) -> torch.Tensor:
        field = self.neural_field
        spatial = expand_spatial_coords(batch)
        temporal = expand_time_coords(batch)
        relative = field.relative_time_tokens(
            batch.batch_size, batch.time_steps, batch.max_nodes
        )
        observed = field.value_proj(batch.values * obs_mask[..., None])
        unknown = self.unknown_value_embedding.to(observed).expand_as(observed)
        value_tokens = torch.where(obs_mask[..., None] > 0, observed, unknown)
        base = field.point_fuse(
            torch.cat(
                [
                    value_tokens,
                    field.encode_space(spatial),
                    field.encode_time(temporal),
                    relative,
                ],
                dim=-1,
            )
        )
        status = torch.stack(
            [obs_mask, frame_ratio[:, :, None].expand_as(obs_mask)], dim=-1
        )
        return base + self.point_mask_adapter(torch.cat([base, status], dim=-1))

    def forward(self, batch: STBatch) -> DirectConditionOutput:
        self.neural_field.activate_batch_geometry(batch)
        # Skip FoundationAlignedConditionEncoder.forward: it deliberately adds
        # the frozen field FiLM and a second domain-conditioned adapter.
        return DirectAlignedConditionEncoder.forward(self, batch)

class ValueCanonicalDirectDiffusion(RawLatentFoundationDirectDiffusion):
    """Raw-latent control diffusion using the shortcut-free conditioner."""

    def __init__(
        self,
        config: ModelConfig,
        neural_field: ValueCanonicalFoundationFieldV1,
        domain_feature_dim: int,
        condition_temporal_layers: int = 2,
        condition_axial_layers: int = 2,
        confidence_bandwidth: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(
            config,
            neural_field,
            domain_feature_dim,
            condition_temporal_layers=condition_temporal_layers,
            condition_axial_layers=condition_axial_layers,
            confidence_bandwidth=confidence_bandwidth,
            **kwargs,
        )
        self.condition_encoder = ValueCanonicalConditionEncoder(
            config,
            neural_field,
            domain_feature_dim,
            temporal_layers=condition_temporal_layers,
            axial_layers=condition_axial_layers,
            heads=config.self_heads,
            confidence_bandwidth=confidence_bandwidth,
        )

class ValueAnchoredConditionEncoder(ValueCanonicalConditionEncoder):
    """Sparse conditioner grounded in observed value-anchor channels."""

    def _observed_point_tokens(
        self,
        batch: STBatch,
        obs_mask: torch.Tensor,
        frame_ratio: torch.Tensor,
    ) -> torch.Tensor:
        field = self.neural_field
        observed = field.value_content_tokens(
            batch.values * obs_mask[..., None], observation_mask=obs_mask
        )
        unknown = self.unknown_value_embedding.to(observed).expand_as(observed)
        base = torch.where(obs_mask[..., None] > 0, observed, unknown)
        status = torch.stack(
            [obs_mask, frame_ratio[:, :, None].expand_as(obs_mask)], dim=-1
        )
        return base + self.point_mask_adapter(torch.cat([base, status], dim=-1))

    def forward(self, batch: STBatch) -> DirectConditionOutput:
        output = super().forward(batch)
        obs_mask = (
            torch.ones_like(batch.values[..., 0])
            if batch.obs_mask is None
            else batch.obs_mask.to(batch.values.dtype)
        )
        anchors = self.neural_field.value_anchors(
            batch,
            values=batch.values * obs_mask[..., None],
            observation_mask=obs_mask,
        )
        complete_temporal = getattr(
            self.neural_field, "complete_temporal_anchors", None
        )
        if complete_temporal is not None:
            # The explicit carrier is already an observed-value-conditioned
            # estimate.  In particular, do not replace its missing-frame
            # channels with an unconstrained absolute-value network output.
            anchor_channels, _, _ = complete_temporal(anchors, batch, obs_mask)
        else:
            blend = output.confidence[..., None].clamp(0, 1)
            anchor_channels = blend * anchors + (1 - blend) * output.condition_grid[
                ..., :VALUE_ANCHOR_DIM
            ]
        grid = torch.cat(
            [anchor_channels, output.condition_grid[..., VALUE_ANCHOR_DIM:]], dim=-1
        )
        return DirectConditionOutput(
            condition_grid=grid,
            raw_point_tokens=output.raw_point_tokens,
            raw_point_mask=output.raw_point_mask,
            condition_features=output.condition_features,
            confidence=output.confidence,
            global_context=grid.mean(dim=(1, 2)),
            frame_observed_ratio=output.frame_observed_ratio,
            previous_distance=output.previous_distance,
            next_distance=output.next_distance,
        )

class ValueAnchoredDirectDiffusion(ValueCanonicalDirectDiffusion):
    """Conditional residual EDM reusing the frozen Stage-1 VAE posterior.

    The condition encoder first produces a grid in the Stage-1 latent feature
    space.  The exact posterior heads learned by Stage 1 then map that grid to
    a conditional mean and scale.  EDM operates on the standardized unresolved
    residual ``(z_full - mean_c) / scale_c`` rather than on the raw latent.

    The first four channels are deterministic analytic value anchors.  Their
    conditional centers are reused, but their diffusion scale is fixed to one;
    the deliberately tiny VAE variance assigned to those channels must not be
    used as a numerical divisor.
    """

    def __init__(self, config, neural_field, domain_feature_dim, **kwargs) -> None:
        super().__init__(config, neural_field, domain_feature_dim, **kwargs)
        self.condition_encoder = ValueAnchoredConditionEncoder(
            config,
            neural_field,
            domain_feature_dim,
            temporal_layers=int(kwargs.get("condition_temporal_layers", 2)),
            axial_layers=int(kwargs.get("condition_axial_layers", 2)),
            heads=config.self_heads,
            confidence_bandwidth=float(kwargs.get("confidence_bandwidth", 0.25)),
        )
        if not isinstance(neural_field, AnchorAffineValueAnchoredFoundationField):
            raise TypeError(
                "ValueAnchoredDirectDiffusion requires the Stage-1 anchor-affine VAE"
            )

        # Keep the legacy module in state dicts so old checkpoints remain
        # loadable, but remove the independently learned Stage-2 latent probe
        # from the active condition-to-posterior path.
        self.condition_probe.requires_grad_(False)

    def conditional_posterior(
        self, condition: DirectConditionOutput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Stage-1 posterior center, effective scale and raw logvar."""

        center, logvar = self.neural_field.posterior_parameters_from_latent(
            condition.condition_grid
        )
        residual_scale = torch.exp(
            0.5 * logvar[..., VALUE_ANCHOR_DIM:]
        ).clamp_min(1e-4)
        scale = torch.cat(
            [
                torch.ones_like(center[..., :VALUE_ANCHOR_DIM]),
                residual_scale,
            ],
            dim=-1,
        )
        return center, scale, logvar

    @staticmethod
    def compose_condition_latent(
        center: torch.Tensor,
        scale: torch.Tensor,
        standardized_residual: torch.Tensor,
    ) -> torch.Tensor:
        return center + scale * standardized_residual

    def loss_for_task(
        self,
        target_latent: torch.Tensor,
        full_batch: STBatch,
        task: STTask,
        task_config: CurrentConditionalTaskConfig,
        generator: Optional[torch.Generator] = None,
        decoded_loss_weight: float = 0.05,
        condition_latent_loss_weight: float = 0.25,
        condition_decoded_loss_weight: float = 0.05,
        shuffle_loss_weight: float = 0.05,
        shuffle_margin: float = 0.05,
        compute_shuffle_loss: bool = False,
        target_logvar: Optional[torch.Tensor] = None,
    ) -> DirectDiffusionLossOutput:
        condition_batch = build_condition_view(
            full_batch, task, task_config, generator=generator
        )
        condition = self.condition_encoder(condition_batch)
        center, scale, condition_logvar = self.conditional_posterior(condition)

        if target_logvar is None:
            # Compatibility for direct callers outside the shared Stage-2
            # trainer. The field is frozen, so this supplies a supervision
            # target without creating Stage-1 parameter gradients.
            with torch.no_grad():
                _, target_logvar = self.neural_field.posterior_parameters(full_batch)
        if target_logvar.shape != target_latent.shape:
            raise ValueError(
                "Stage-1 target posterior mean/logvar shapes must match: "
                f"{tuple(target_latent.shape)} vs {tuple(target_logvar.shape)}"
            )

        # Detaching the affine frame prevents EDM from shrinking its own target
        # by inflating scale. The conditional posterior is instead trained by
        # the explicit full-to-condition Gaussian KL below.
        target_residual = (target_latent - center.detach()) / scale.detach()
        latent_weight = 1.0 + self.missing_loss_weight * (
            1.0 - condition.confidence
        )

        rnd_normal = torch.randn(
            target_latent.shape[0],
            device=target_latent.device,
            dtype=target_latent.dtype,
        )
        sigma = (rnd_normal * self.config.p_std + self.config.p_mean).exp()
        noisy = target_residual + torch.randn_like(target_residual) * append_dims(
            sigma, target_residual.ndim
        )
        denoise_condition = (
            condition
            if compute_shuffle_loss
            else self._drop_condition_samples(condition)
        )
        denoised_residual = self.denoise(noisy, sigma, denoise_condition)
        edm_weight = (sigma.square() + self.config.sigma_data**2) / (
            sigma * self.config.sigma_data
        ).square()
        edm_element = (denoised_residual - target_residual).square().mean(dim=-1)
        weighted = (
            append_dims(edm_weight, edm_element.ndim)
            * edm_element
            * latent_weight
        )
        edm_loss = weighted.sum() / latent_weight.sum().clamp_min(1.0)

        valid = torch.ones_like(condition_batch.obs_mask)
        if full_batch.node_mask is not None:
            valid = valid * full_batch.node_mask[:, None, :].to(valid.dtype)
        natural_valid = full_batch.metadata.get("valid_mask")
        if natural_valid is not None:
            valid = valid * natural_valid.to(valid.dtype)
        missing_mask = valid * (1.0 - condition_batch.obs_mask)

        low_sigma = sigma <= 1.0
        if low_sigma.any() and decoded_loss_weight > 0:
            predicted_latent = self.compose_condition_latent(
                center[low_sigma],
                scale[low_sigma],
                denoised_residual[low_sigma],
            )
            query = build_full_query_coords(full_batch)[low_sigma]
            selected_full = _slice_batch_rows(full_batch, low_sigma)
            selected_condition = _slice_batch_rows(condition_batch, low_sigma)
            prediction = decode_neural_field(
                self.neural_field,
                predicted_latent,
                query,
                selected_full,
                selected_condition,
            )
            decoded_loss = self._masked_value_l1(
                prediction,
                full_batch.values[low_sigma],
                missing_mask[low_sigma],
            )
        else:
            decoded_loss = target_latent.new_zeros(())

        # The analytic anchors have no learned posterior variance, so supervise
        # only their centers. Learned residual channels use KL(q_full || q_c),
        # which identifies both conditional mean and conditional scale and
        # prevents a degenerate arbitrary affine frame for residual diffusion.
        anchor_center_error = F.smooth_l1_loss(
            center[..., :VALUE_ANCHOR_DIM],
            target_latent[..., :VALUE_ANCHOR_DIM],
            reduction="none",
        ).mean(dim=-1)
        full_residual_mean = target_latent[..., VALUE_ANCHOR_DIM:].detach()
        full_residual_logvar = target_logvar[
            ..., VALUE_ANCHOR_DIM:
        ].detach()
        condition_residual_mean = center[..., VALUE_ANCHOR_DIM:]
        condition_residual_logvar = condition_logvar[..., VALUE_ANCHOR_DIM:]
        posterior_kl = 0.5 * (
            condition_residual_logvar
            - full_residual_logvar
            + (
                full_residual_logvar.exp()
                + (full_residual_mean - condition_residual_mean).square()
            )
            / condition_residual_logvar.exp().clamp_min(1e-8)
            - 1.0
        )
        condition_latent_element = anchor_center_error + posterior_kl.mean(dim=-1)
        condition_latent_loss = (
            condition_latent_element * latent_weight
        ).sum() / latent_weight.sum().clamp_min(1.0)

        if condition_decoded_loss_weight > 0:
            condition_prediction = decode_neural_field(
                self.neural_field,
                center,
                build_full_query_coords(full_batch),
                full_batch,
                condition_batch,
            )
            condition_decoded_loss = self._masked_value_l1(
                condition_prediction,
                full_batch.values,
                missing_mask,
            )
        else:
            condition_decoded_loss = target_latent.new_zeros(())

        shuffle_loss = target_latent.new_zeros(())
        condition_advantage = target_latent.new_zeros(())
        shuffle_applied = target_latent.new_zeros(())
        if (
            compute_shuffle_loss
            and shuffle_loss_weight > 0
            and target_latent.shape[0] > 1
        ):
            shuffled_denoised = self.denoise(
                noisy, sigma, self._roll_condition(condition)
            )
            correct_error = (denoised_residual - target_residual).square().mean(
                dim=(1, 2, 3)
            )
            shuffled_error = (shuffled_denoised - target_residual).square().mean(
                dim=(1, 2, 3)
            )
            condition_advantage = (shuffled_error - correct_error).mean()
            shuffle_loss = F.relu(
                shuffle_margin + correct_error - shuffled_error
            ).mean()
            shuffle_applied = target_latent.new_ones(())

        missing_latent = condition.confidence < 0.5
        missing_confidence = (
            condition.confidence[missing_latent].mean()
            if missing_latent.any()
            else condition.confidence.new_ones(())
        )
        residual_scale = scale[..., VALUE_ANCHOR_DIM:]
        return DirectDiffusionLossOutput(
            loss=(
                edm_loss
                + decoded_loss_weight * decoded_loss
                + condition_latent_loss_weight * condition_latent_loss
                + condition_decoded_loss_weight * condition_decoded_loss
                + shuffle_loss_weight * shuffle_loss
            ),
            edm_loss=edm_loss,
            decoded_loss=decoded_loss,
            condition_latent_loss=condition_latent_loss,
            condition_decoded_loss=condition_decoded_loss,
            shuffle_loss=shuffle_loss,
            condition_advantage=condition_advantage,
            shuffle_applied=shuffle_applied,
            mean_sigma=sigma.mean(),
            mean_confidence=condition.confidence.mean(),
            missing_confidence=missing_confidence,
            target_abs_mean=target_residual.abs().mean(),
            task=task.value,
            mean_uncertainty=residual_scale.mean(),
        )

    @torch.no_grad()
    def sample_conditional(
        self,
        condition_batch: STBatch,
        final_query_coords: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, DirectConditionOutput]:
        condition = self.condition_encoder(condition_batch)
        center, scale, _ = self.conditional_posterior(condition)
        null_condition = (
            self._null_condition(condition) if self.guidance_scale != 1.0 else None
        )
        x_next = torch.randn_like(center) * self.config.sigma_max
        steps = self.karras_schedule(
            x_next.device, x_next.dtype, num_steps=num_steps
        )
        for index, (current, following) in enumerate(zip(steps[:-1], steps[1:])):
            sigma = x_next.new_full((x_next.shape[0],), current)
            denoised = self.denoise(x_next, sigma, condition)
            if null_condition is not None:
                unconditional = self.denoise(x_next, sigma, null_condition)
                denoised = unconditional + self.guidance_scale * (
                    denoised - unconditional
                )
            derivative = (x_next - denoised) / current.clamp_min(
                self.config.sigma_min
            )
            euler = x_next + (following - current) * derivative
            if index < len(steps) - 2 and following > 0:
                next_sigma = x_next.new_full((x_next.shape[0],), following)
                next_denoised = self.denoise(euler, next_sigma, condition)
                if null_condition is not None:
                    next_unconditional = self.denoise(
                        euler, next_sigma, null_condition
                    )
                    next_denoised = next_unconditional + self.guidance_scale * (
                        next_denoised - next_unconditional
                    )
                next_derivative = (
                    euler - next_denoised
                ) / following.clamp_min(self.config.sigma_min)
                x_next = x_next + 0.5 * (following - current) * (
                    derivative + next_derivative
                )
            else:
                x_next = euler

        final_latent = self.compose_condition_latent(center, scale, x_next)
        query = (
            build_full_query_coords(condition_batch)
            if final_query_coords is None
            else final_query_coords
        )
        return (
            final_latent,
            decode_neural_field(
                self.neural_field,
                final_latent,
                query,
                condition_batch,
                condition_batch,
            ),
            condition,
        )


def merge_test_condition_preserving_domain(batch: STBatch) -> STBatch:
    """Merge held-out-node context while retaining foundation metadata."""

    context = build_condition_batch_from_metadata(batch)
    if context is None:
        return batch
    if batch.values.shape[:2] != context.values.shape[:2]:
        raise ValueError("Primary and context batches must share batch/time dimensions")
    context_valid = batch.metadata.get("context_valid_mask")
    context_obs = context.obs_mask
    if context_valid is not None:
        context_obs = (
            context_valid
            if context_obs is None
            else context_obs * context_valid.to(context_obs.dtype)
        )
    if context_obs is not None and batch.obs_mask is not None:
        merged_obs = torch.cat([context_obs, batch.obs_mask], dim=2)
    elif context_obs is not None:
        merged_obs = context_obs
    else:
        merged_obs = batch.obs_mask
    retained_keys = {
        "domain_id",
        "domain_features",
        "delta_hours",
        "value_offset",
        "value_scale",
        "value_mean",
        "value_std",
        "geo_center_ecef",
        "geo_local_chord_scale",
        "geo_extent_km",
    }
    return STBatch(
        values=torch.cat([context.values, batch.values], dim=2),
        spatial_coords=torch.cat(
            [context.spatial_coords, batch.spatial_coords], dim=1
        ),
        time_coords=batch.time_coords,
        node_mask=torch.cat([context.node_mask, batch.node_mask], dim=1),
        obs_mask=merged_obs,
        metadata={
            key: value
            for key, value in batch.metadata.items()
            if key in retained_keys
        },
    )

__all__ = [
    "CONDITIONAL_TASKS",
    "CurrentConditionalTaskConfig",
    "DirectConditionOutput",
    "StructuredDirectLatentDiffusion",
    "ValueAnchoredDirectDiffusion",
    "build_condition_view",
    "decode_neural_field",
    "merge_test_condition_preserving_domain",
]
