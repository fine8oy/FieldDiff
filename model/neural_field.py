from __future__ import annotations

""" anchor-affine neural-field implementation."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.loader import ProcessedDomain
from .common import ModelConfig, STBatch

def get_sinusoidal_embeddings(n: int, d: int) -> torch.Tensor:
    """
    Sinusoidal latent initialization.
    """

    if d % 2 != 0:
        raise ValueError("latent_dim must be even for sinusoidal embeddings.")
    position = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / d))
    pe = torch.zeros(n, d, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def _expand_spatial_coords(batch: STBatch) -> torch.Tensor:
    spatial = batch.spatial_coords
    if spatial.ndim == 3:
        return spatial[:, None, :, :].expand(-1, batch.time_steps, -1, -1)
    if spatial.ndim == 4:
        return spatial
    raise ValueError(f"Unsupported spatial_coords shape: {tuple(spatial.shape)}")

def _expand_time_coords(batch: STBatch) -> torch.Tensor:
    time_coords = batch.time_coords
    if time_coords.ndim == 3:
        return time_coords[:, :, None, :].expand(-1, -1, batch.max_nodes, -1)
    if time_coords.ndim == 4:
        return time_coords
    raise ValueError(f"Unsupported time_coords shape: {tuple(time_coords.shape)}")

def build_full_query_coords(batch: STBatch) -> torch.Tensor:
    """
    Build dense query coordinates with shape [B, T, N, Ds + Dt].
    """

    spatial = _expand_spatial_coords(batch)
    time_coords = _expand_time_coords(batch)
    return torch.cat([spatial, time_coords], dim=-1)

def build_observation_mask(batch: STBatch) -> torch.Tensor:
    mask = torch.ones(
        batch.batch_size,
        batch.time_steps,
        batch.max_nodes,
        device=batch.values.device,
        dtype=batch.values.dtype,
    )
    if batch.node_mask is not None:
        mask = mask * batch.node_mask[:, None, :].to(mask.dtype)
    if batch.obs_mask is not None:
        mask = mask * batch.obs_mask.to(mask.dtype)
    return mask

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, context_dim: Optional[int] = None) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim) if context_dim is not None else None
        self.fn = fn

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm(x)
        if context is not None and self.context_norm is not None:
            context = self.context_norm(context)
        return self.fn(x, context=context, mask=mask)

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        del context, mask
        return self.net(x)

class GaussianFourierFeatures(nn.Module):
    """
    Gaussian random Fourier features:
    x -> [sin(xB), cos(xB)].
    """

    def __init__(self, in_features: int, mapping_size: int, scale: float = 10.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.register_buffer("B", torch.randn(in_features, mapping_size) * scale)

    @property
    def out_dim(self) -> int:
        return self.mapping_size * 2

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        projections = coords @ self.B
        return torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)

class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        context_dim = context_dim or query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, n, _ = q.shape
        h = self.heads

        q = q.view(b, n, h, -1).transpose(1, 2)
        k = k.view(b, context.shape[1], h, -1).transpose(1, 2)
        v = v.view(b, context.shape[1], h, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            key_mask = mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn = scores.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)

class CascadedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_latents: int,
        input_dim: int,
        cross_heads: int,
        cross_dim_head: int,
        self_heads: int,
        self_dim_head: int,
        residual_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(get_sinusoidal_embeddings(n_latents, dim), requires_grad=True)
        self.cross_attn = PreNorm(
            dim,
            Attention(dim, context_dim=input_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=input_dim,
        )
        self.self_attn = PreNorm(dim, Attention(dim, heads=self_heads, dim_head=self_dim_head))
        self.ff = PreNorm(dim, FeedForward(dim))
        self.residual_proj = nn.Linear(residual_dim, dim) if residual_dim is not None and residual_dim != dim else None

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = context.shape[0]
        latents = self.latents[None, :, :].expand(b, -1, -1)
        latents = latents + self.cross_attn(latents, context=context, mask=mask)
        if residual is not None:
            residual = self.residual_proj(residual) if self.residual_proj is not None else residual
            latents = latents + residual
        latents = latents + self.self_attn(latents)
        latents = latents + self.ff(latents)
        return latents

class QueryDecoder(nn.Module):
    def __init__(self, query_dim: int, latent_dim: int, output_dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        self.cross_attn = PreNorm(
            query_dim,
            Attention(query_dim, context_dim=latent_dim, heads=heads, dim_head=dim_head),
            context_dim=latent_dim,
        )
        self.ff = PreNorm(query_dim, FeedForward(query_dim))
        self.to_out = nn.Linear(query_dim, output_dim)

    def forward(self, queries: torch.Tensor, latent_tokens: torch.Tensor) -> torch.Tensor:
        x = queries + self.cross_attn(queries, context=latent_tokens)
        x = x + self.ff(x)
        return self.to_out(x)

class NeuralFieldOutput:
    latent_tokens: torch.Tensor
    reconstruction: torch.Tensor
    observation_mask: torch.Tensor

class SingleDomainNeuralFieldAE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        time_fourier_in_dim = self._time_fourier_input_dim(config)

        self.space_fourier = GaussianFourierFeatures(
            in_features=config.spatial_dim,
            mapping_size=config.spatial_fourier_size,
            scale=config.spatial_fourier_scale,
        )
        self.time_fourier = GaussianFourierFeatures(
            in_features=time_fourier_in_dim,
            mapping_size=config.temporal_fourier_size,
            scale=config.temporal_fourier_scale,
        )

        self.value_proj = nn.Sequential(
            nn.Linear(config.value_dim, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, config.token_dim),
        )
        self.space_proj = nn.Sequential(
            nn.Linear(self.space_fourier.out_dim, config.coord_embed_dim),
            nn.GELU(),
            nn.Linear(config.coord_embed_dim, config.coord_embed_dim),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_fourier.out_dim, config.time_embed_dim),
            nn.GELU(),
            nn.Linear(config.time_embed_dim, config.time_embed_dim),
        )
        token_in_dim = config.token_dim + config.coord_embed_dim + config.time_embed_dim
        self.token_fuse = nn.Sequential(
            nn.Linear(token_in_dim, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, config.token_dim),
        )

        stage_dims = [config.latent_dim] * config.num_encoder_stages
        stage_latents = [config.num_latents] * config.num_encoder_stages
        self.encoder_blocks = nn.ModuleList()
        prev_dim: Optional[int] = None
        for dim, n_latents in zip(stage_dims, stage_latents):
            self.encoder_blocks.append(
                CascadedBlock(
                    dim=dim,
                    n_latents=n_latents,
                    input_dim=config.token_dim,
                    cross_heads=config.cross_heads,
                    cross_dim_head=config.cross_dim_head,
                    self_heads=config.self_heads,
                    self_dim_head=config.self_dim_head,
                    residual_dim=prev_dim,
                )
            )
            prev_dim = dim

        self.global_to_latent = nn.ModuleList(
            [nn.Linear(config.latent_dim, config.num_latents * config.latent_dim) for _ in range(config.num_encoder_stages)]
        )
        self.latent_refiner = nn.ModuleList(
            [PreNorm(config.latent_dim, Attention(config.latent_dim, heads=config.self_heads, dim_head=config.self_dim_head))
             for _ in range(config.decoder_depth)]
        )
        self.latent_ff = nn.ModuleList([PreNorm(config.latent_dim, FeedForward(config.latent_dim)) for _ in range(config.decoder_depth)])

        query_in_dim = config.coord_embed_dim + config.time_embed_dim
        self.query_proj = nn.Sequential(
            nn.Linear(query_in_dim, config.decoder_query_dim),
            nn.GELU(),
            nn.Linear(config.decoder_query_dim, config.decoder_query_dim),
        )
        self.decoder = QueryDecoder(
            query_dim=config.decoder_query_dim,
            latent_dim=config.latent_dim,
            output_dim=config.value_dim,
            heads=config.cross_heads,
            dim_head=config.cross_dim_head,
        )

    @staticmethod
    def _time_fourier_input_dim(config: ModelConfig) -> int:
        if config.use_periodic_time_features and config.time_coord_dim == 4:
            return 9
        return config.time_coord_dim

    def _calendar_time_to_periodic(self, time_coords: torch.Tensor) -> torch.Tensor:
        if not self.config.use_periodic_time_features or time_coords.shape[-1] != 4:
            return time_coords

        year = time_coords[..., 0]
        month = time_coords[..., 1]
        day = time_coords[..., 2]
        hour = time_coords[..., 3]

        two_pi = 2.0 * torch.pi
        year_cont = ((year - 2000.0) / 100.0).unsqueeze(-1)
        month_angle = two_pi * (month - 1.0) / 12.0
        day_angle = two_pi * (day - 1.0) / 31.0
        hour_angle = two_pi * hour / 24.0

        month_lengths = time_coords.new_tensor([31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0])
        cumulative_days = torch.cat([month_lengths.new_zeros(1), month_lengths.cumsum(dim=0)[:-1]])
        month_idx = (month.round().long() - 1).clamp(0, 11)
        day_of_year = cumulative_days[month_idx] + (day - 1.0).clamp_min(0.0)
        doy_angle = two_pi * day_of_year / 365.0

        return torch.cat(
            [
                year_cont,
                torch.sin(month_angle).unsqueeze(-1),
                torch.cos(month_angle).unsqueeze(-1),
                torch.sin(day_angle).unsqueeze(-1),
                torch.cos(day_angle).unsqueeze(-1),
                torch.sin(hour_angle).unsqueeze(-1),
                torch.cos(hour_angle).unsqueeze(-1),
                torch.sin(doy_angle).unsqueeze(-1),
                torch.cos(doy_angle).unsqueeze(-1),
            ],
            dim=-1,
        )

    def _batch_to_tokens(self, batch: STBatch) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = _expand_spatial_coords(batch)
        time_coords = _expand_time_coords(batch)
        if spatial.shape[-1] != self.config.spatial_dim:
            raise ValueError(
                f"Expected spatial coords dim {self.config.spatial_dim}, got {spatial.shape[-1]}"
            )
        if time_coords.shape[-1] != self.config.time_coord_dim:
            raise ValueError(
                f"Expected time coords dim {self.config.time_coord_dim}, got {time_coords.shape[-1]}"
            )
        obs_mask = build_observation_mask(batch)

        value_tokens = self.value_proj(batch.values)
        space_tokens = self.space_proj(self.space_fourier(spatial))
        time_tokens = self.time_proj(self.time_fourier(self._calendar_time_to_periodic(time_coords)))
        tokens = self.token_fuse(torch.cat([value_tokens, space_tokens, time_tokens], dim=-1))

        b, t, n, d = tokens.shape
        tokens = tokens.view(b, t * n, d)
        token_mask = obs_mask.view(b, t * n) > 0
        return tokens, token_mask

    def _encode_query_coords(self, query_coords: torch.Tensor) -> torch.Tensor:
        spatial = query_coords[..., : self.config.spatial_dim]
        time_coords = query_coords[..., self.config.spatial_dim :]
        if time_coords.shape[-1] != self.config.time_coord_dim:
            raise ValueError(
                f"Expected query time coords dim {self.config.time_coord_dim}, got {time_coords.shape[-1]}"
            )
        return self.query_proj(
            torch.cat(
                [
                    self.space_proj(self.space_fourier(spatial)),
                    self.time_proj(self.time_fourier(self._calendar_time_to_periodic(time_coords))),
                ],
                dim=-1,
            )
        )

    def encode(self, batch: STBatch) -> torch.Tensor:
        tokens, token_mask = self._batch_to_tokens(batch)
        global_latent: Optional[torch.Tensor] = None

        for stage_idx, block in enumerate(self.encoder_blocks):
            residual = None
            if global_latent is not None:
                pooled = global_latent.mean(dim=1)
                residual = self.global_to_latent[stage_idx](pooled).view(
                    pooled.shape[0], self.config.num_latents, self.config.latent_dim
                )
            global_latent = block(context=tokens, mask=token_mask, residual=residual)

        assert global_latent is not None
        for attn, ff in zip(self.latent_refiner, self.latent_ff):
            global_latent = global_latent + attn(global_latent)
            global_latent = global_latent + ff(global_latent)
        return global_latent

    def decode(self, latent_tokens: torch.Tensor, query_coords: torch.Tensor) -> torch.Tensor:
        b = latent_tokens.shape[0]
        if query_coords.ndim == 4:
            q = self._encode_query_coords(query_coords).view(b, -1, self.config.decoder_query_dim)
            out = self.decoder(q, latent_tokens)
            return out.view(query_coords.shape[0], query_coords.shape[1], query_coords.shape[2], self.config.value_dim)
        if query_coords.ndim == 3:
            q = self._encode_query_coords(query_coords)
            return self.decoder(q, latent_tokens)
        raise ValueError(f"Unsupported query_coords shape: {tuple(query_coords.shape)}")

    def reconstruct(self, batch: STBatch, query_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        if query_coords is None:
            query_coords = build_full_query_coords(batch)
        latent_tokens = self.encode(batch)
        return self.decode(latent_tokens, query_coords)

    def forward(self, batch: STBatch, query_coords: Optional[torch.Tensor] = None) -> NeuralFieldOutput:
        if query_coords is None:
            query_coords = build_full_query_coords(batch)
        latent_tokens = self.encode(batch)
        reconstruction = self.decode(latent_tokens, query_coords)
        return NeuralFieldOutput(
            latent_tokens=latent_tokens,
            reconstruction=reconstruction,
            observation_mask=build_observation_mask(batch),
        )

def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    residual = prediction - target
    if mask is not None:
        residual = residual * mask.unsqueeze(-1)
    loss = residual.square()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    if mask is None:
        return loss.mean()
    denom = mask.sum().clamp_min(1).float() * target.shape[-1]
    return loss.sum() / denom

def expand_spatial_coords(batch: STBatch) -> torch.Tensor:
    if batch.spatial_coords.ndim == 3:
        return batch.spatial_coords[:, None, :, :].expand(-1, batch.time_steps, -1, -1)
    if batch.spatial_coords.ndim == 4:
        return batch.spatial_coords
    raise ValueError(f"Unsupported spatial_coords shape: {tuple(batch.spatial_coords.shape)}")

def expand_time_coords(batch: STBatch) -> torch.Tensor:
    if batch.time_coords.ndim == 3:
        return batch.time_coords[:, :, None, :].expand(-1, -1, batch.max_nodes, -1)
    if batch.time_coords.ndim == 4:
        return batch.time_coords
    raise ValueError(f"Unsupported time_coords shape: {tuple(batch.time_coords.shape)}")

class StructuredAxialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        temporal_radius: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.temporal_radius = temporal_radius
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.temporal_ff_norm = nn.LayerNorm(dim)
        self.temporal_ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ff_mult, dim)
        )
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.spatial_ff_norm = nn.LayerNorm(dim)
        self.spatial_ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ff_mult, dim)
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        b, t, k, d = grid.shape
        temporal = grid.permute(0, 2, 1, 3).reshape(b * k, t, d)
        q = self.temporal_norm(temporal)
        temporal_mask = None
        if self.temporal_radius is not None:
            positions = torch.arange(t, device=grid.device)
            temporal_mask = (positions[:, None] - positions[None, :]).abs() > self.temporal_radius
        temporal = temporal + self.temporal_attn(
            q, q, q, attn_mask=temporal_mask, need_weights=False
        )[0]
        temporal = temporal + self.temporal_ff(self.temporal_ff_norm(temporal))
        grid = temporal.reshape(b, k, t, d).permute(0, 2, 1, 3).contiguous()

        spatial = grid.reshape(b * t, k, d)
        q = self.spatial_norm(spatial)
        spatial = spatial + self.spatial_attn(q, q, q, need_weights=False)[0]
        spatial = spatial + self.spatial_ff(self.spatial_ff_norm(spatial))
        return spatial.reshape(b, t, k, d)

class SpatialSlotBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ff_mult, dim)
        )

    def forward(
        self,
        slots: torch.Tensor,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor],
        return_cross_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = None if point_mask is None else ~point_mask.to(torch.bool)
        q = self.query_norm(slots)
        context = self.context_norm(points)
        if return_cross_attention:
            cross_update, cross_attention = self.cross_attn(
                q,
                context,
                context,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
        else:
            cross_update = self.cross_attn(
                q,
                context,
                context,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
            cross_attention = None
        slots = slots + cross_update
        q = self.self_norm(slots)
        slots = slots + self.self_attn(q, q, q, need_weights=False)[0]
        slots = slots + self.ff(self.ff_norm(slots))
        if return_cross_attention:
            assert cross_attention is not None
            return slots, cross_attention
        return slots

class CoordinateSpatialTokenizer(nn.Module):
    """Compress every frame into K stable, coordinate-aware latent slots."""

    def __init__(
        self,
        dim: int,
        spatial_dim: int,
        num_slots: int,
        heads: int,
        layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if spatial_dim != 2:
            initial = torch.linspace(0.15, 0.85, num_slots)[:, None].repeat(1, spatial_dim)
        else:
            rows = 2
            cols = (num_slots + rows - 1) // rows
            yy, xx = torch.meshgrid(
                torch.linspace(0.2, 0.8, rows),
                torch.linspace(0.1, 0.9, cols),
                indexing="ij",
            )
            initial = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[:num_slots]
        initial = initial.clamp(1e-4, 1 - 1e-4)
        self.anchor_logits = nn.Parameter(torch.logit(initial))
        self.slot_queries = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        self.anchor_proj = nn.Sequential(nn.Linear(spatial_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([SpatialSlotBlock(dim, heads, dropout=dropout) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(dim)

    @property
    def num_slots(self) -> int:
        return int(self.slot_queries.shape[0])

    def anchor_coords(self) -> torch.Tensor:
        return torch.sigmoid(self.anchor_logits)

    def initial_slots(self, batch_frames: int, time_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        anchors = self.anchor_coords()
        slots = self.slot_queries + self.anchor_proj(anchors)
        slots = slots[None, :, :].expand(batch_frames, -1, -1)
        if time_tokens is not None:
            slots = slots + time_tokens[:, None, :]
        return slots

    def forward(
        self,
        point_tokens: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        time_tokens: Optional[torch.Tensor] = None,
        return_cross_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # point_tokens [BT,N,D].  MultiheadAttention cannot accept an all-masked
        # row, so callers may supply one safe dummy key and replace that row.
        slots = self.initial_slots(point_tokens.shape[0], time_tokens=time_tokens)
        cross_attention = None
        for index, block in enumerate(self.blocks):
            capture = return_cross_attention and index == len(self.blocks) - 1
            output = block(
                slots,
                point_tokens,
                point_mask,
                return_cross_attention=capture,
            )
            if capture:
                slots, cross_attention = output
            else:
                slots = output
        slots = self.out_norm(slots)
        if return_cross_attention:
            if cross_attention is None:
                raise RuntimeError("Cross-attention confidence requires at least one tokenizer block.")
            return slots, cross_attention
        return slots

    def repulsion_loss(self, margin: float = 0.2) -> torch.Tensor:
        anchors = self.anchor_coords()
        if anchors.shape[0] < 2:
            return anchors.new_zeros(())
        distance = torch.cdist(anchors, anchors)
        eye = torch.eye(anchors.shape[0], device=anchors.device, dtype=torch.bool)
        return F.relu(margin - distance.masked_fill(eye, margin)).square().masked_fill(eye, 0).mean()

class StructuredQueryDecoder(nn.Module):
    def __init__(self, dim: int, output_dim: int, heads: int, temporal_radius: int = 2) -> None:
        super().__init__()
        self.temporal_radius = temporal_radius
        self.local_q_norm = nn.LayerNorm(dim)
        self.local_z_norm = nn.LayerNorm(dim)
        self.local_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.neighbour_q_norm = nn.LayerNorm(dim)
        self.neighbour_z_norm = nn.LayerNorm(dim)
        self.neighbour_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.to_out = nn.Linear(dim, output_dim)

    def forward(self, queries: torch.Tensor, latent_grid: torch.Tensor) -> torch.Tensor:
        # queries [B,T,N,D], latent_grid [B,T,K,D]
        b, t, n, d = queries.shape
        k = latent_grid.shape[2]
        q = queries.reshape(b * t, n, d)
        local = latent_grid.reshape(b * t, k, d)
        x = q + self.local_attn(
            self.local_q_norm(q), self.local_z_norm(local), self.local_z_norm(local), need_weights=False
        )[0]

        offsets = torch.arange(-self.temporal_radius, self.temporal_radius + 1, device=queries.device)
        base = torch.arange(t, device=queries.device)[:, None]
        raw_indices = base + offsets[None, :]
        valid = (raw_indices >= 0) & (raw_indices < t)
        indices = raw_indices.clamp(0, t - 1)
        neighbourhood = latent_grid[:, indices, :, :]  # [B,T,W,K,D]
        neighbourhood = neighbourhood.reshape(b * t, len(offsets) * k, d)
        invalid = (~valid)[:, :, None].expand(t, len(offsets), k).reshape(t, len(offsets) * k)
        invalid = invalid[None, :, :].expand(b, -1, -1).reshape(b * t, len(offsets) * k)
        context = self.neighbour_z_norm(neighbourhood)
        x = x + self.neighbour_attn(
            self.neighbour_q_norm(x), context, context, key_padding_mask=invalid, need_weights=False
        )[0]
        x = x + self.ff(self.ff_norm(x))
        return self.to_out(x).reshape(b, t, n, -1)

class StructuredNeuralFieldDiagnostics:
    anchors: torch.Tensor
    latent_temporal_cosine: torch.Tensor
    latent_spatial_cosine: torch.Tensor

class StructuredSpatioTemporalNeuralFieldAE(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        num_spatial_latents: int = 8,
        spatial_tokenizer_layers: int = 2,
        axial_layers: int = 4,
        heads: Optional[int] = None,
        decoder_temporal_radius: int = 2,
        temporal_attention_radius: Optional[int] = 3,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_spatial_latents = num_spatial_latents
        heads = heads or config.self_heads
        if config.token_dim != config.latent_dim or config.decoder_query_dim != config.latent_dim:
            raise ValueError("The first structured experiment requires token_dim=latent_dim=decoder_query_dim.")

        time_fourier_in_dim = 9 if config.use_periodic_time_features and config.time_coord_dim == 4 else config.time_coord_dim
        self.space_fourier = GaussianFourierFeatures(
            config.spatial_dim, config.spatial_fourier_size, config.spatial_fourier_scale
        )
        self.time_fourier = GaussianFourierFeatures(
            time_fourier_in_dim, config.temporal_fourier_size, config.temporal_fourier_scale
        )
        self.value_proj = nn.Sequential(
            nn.Linear(config.value_dim, config.token_dim), nn.GELU(), nn.Linear(config.token_dim, config.token_dim)
        )
        self.space_proj = nn.Sequential(
            nn.Linear(self.space_fourier.out_dim, config.coord_embed_dim),
            nn.GELU(),
            nn.Linear(config.coord_embed_dim, config.coord_embed_dim),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_fourier.out_dim, config.time_embed_dim),
            nn.GELU(),
            nn.Linear(config.time_embed_dim, config.time_embed_dim),
        )
        self.relative_time = nn.Parameter(torch.zeros(config.time_steps, config.time_embed_dim))
        self.point_fuse = nn.Sequential(
            nn.Linear(config.token_dim + config.coord_embed_dim + config.time_embed_dim * 2, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, config.token_dim),
        )
        self.time_to_latent = nn.Linear(config.time_embed_dim * 2, config.latent_dim)
        self.spatial_tokenizer = CoordinateSpatialTokenizer(
            config.latent_dim,
            config.spatial_dim,
            num_spatial_latents,
            heads,
            spatial_tokenizer_layers,
        )
        self.axial_blocks = nn.ModuleList(
            [
                StructuredAxialBlock(
                    config.latent_dim,
                    heads,
                    dropout=0.0,
                    temporal_radius=temporal_attention_radius,
                )
                for _ in range(axial_layers)
            ]
        )
        self.latent_norm = nn.LayerNorm(config.latent_dim)
        self.query_fuse = nn.Sequential(
            nn.Linear(config.coord_embed_dim + config.time_embed_dim * 2, config.decoder_query_dim),
            nn.GELU(),
            nn.Linear(config.decoder_query_dim, config.decoder_query_dim),
        )
        self.decoder = StructuredQueryDecoder(
            config.decoder_query_dim,
            config.value_dim,
            heads=config.cross_heads,
            temporal_radius=decoder_temporal_radius,
        )

    def calendar_time_to_periodic(self, time_coords: torch.Tensor) -> torch.Tensor:
        if not self.config.use_periodic_time_features or time_coords.shape[-1] != 4:
            return time_coords
        year, month, day, hour = [time_coords[..., i] for i in range(4)]
        two_pi = 2.0 * torch.pi
        month_lengths = time_coords.new_tensor(
            [31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0]
        )
        cumulative_days = torch.cat([month_lengths.new_zeros(1), month_lengths.cumsum(0)[:-1]])
        month_idx = (month.round().long() - 1).clamp(0, 11)
        doy = cumulative_days[month_idx] + (day - 1.0).clamp_min(0.0)
        return torch.stack(
            [
                (year - 2000.0) / 100.0,
                torch.sin(two_pi * (month - 1.0) / 12.0),
                torch.cos(two_pi * (month - 1.0) / 12.0),
                torch.sin(two_pi * (day - 1.0) / 31.0),
                torch.cos(two_pi * (day - 1.0) / 31.0),
                torch.sin(two_pi * hour / 24.0),
                torch.cos(two_pi * hour / 24.0),
                torch.sin(two_pi * doy / 365.0),
                torch.cos(two_pi * doy / 365.0),
            ],
            dim=-1,
        )

    def encode_space(self, coords: torch.Tensor) -> torch.Tensor:
        return self.space_proj(self.space_fourier(coords))

    def encode_time(self, coords: torch.Tensor) -> torch.Tensor:
        return self.time_proj(self.time_fourier(self.calendar_time_to_periodic(coords)))

    def relative_time_tokens(self, batch_size: int, time_steps: int, num_nodes: int) -> torch.Tensor:
        if time_steps > self.config.time_steps:
            raise ValueError(f"time_steps={time_steps} exceeds configured time_steps={self.config.time_steps}")
        return self.relative_time[:time_steps][None, :, None, :].expand(batch_size, -1, num_nodes, -1)

    def point_tokens(self, batch: STBatch, values: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        values = batch.values if values is None else values
        spatial = expand_spatial_coords(batch)
        temporal = expand_time_coords(batch)
        rel = self.relative_time_tokens(batch.batch_size, batch.time_steps, batch.max_nodes)
        tokens = self.point_fuse(
            torch.cat([self.value_proj(values), self.encode_space(spatial), self.encode_time(temporal), rel], dim=-1)
        )
        valid = torch.ones(
            batch.batch_size, batch.time_steps, batch.max_nodes, device=values.device, dtype=torch.bool
        )
        if batch.node_mask is not None:
            valid = valid & (batch.node_mask[:, None, :] > 0)
        return tokens, valid

    def time_latent_tokens(self, batch: STBatch) -> torch.Tensor:
        temporal = batch.time_coords if batch.time_coords.ndim == 3 else batch.time_coords[:, :, 0, :]
        absolute = self.encode_time(temporal)
        relative = self.relative_time[: batch.time_steps][None, :, :].expand(batch.batch_size, -1, -1)
        return self.time_to_latent(torch.cat([absolute, relative], dim=-1))

    def encode_spatial_frames(self, batch: STBatch, values: Optional[torch.Tensor] = None) -> torch.Tensor:
        point_tokens, valid = self.point_tokens(batch, values=values)
        b, t, n, d = point_tokens.shape
        time_tokens = self.time_latent_tokens(batch).reshape(b * t, d)
        return self.spatial_tokenizer(
            point_tokens.reshape(b * t, n, d),
            point_mask=valid.reshape(b * t, n),
            time_tokens=time_tokens,
        ).reshape(b, t, self.num_spatial_latents, d)

    def encode(self, batch: STBatch) -> torch.Tensor:
        grid = self.encode_spatial_frames(batch)
        for block in self.axial_blocks:
            grid = block(grid)
        return self.latent_norm(grid)

    def encode_query_coords(self, query_coords: torch.Tensor) -> torch.Tensor:
        if query_coords.ndim != 4:
            raise ValueError(f"Expected query_coords [B,T,N,D], got {tuple(query_coords.shape)}")
        spatial = query_coords[..., : self.config.spatial_dim]
        temporal = query_coords[..., self.config.spatial_dim :]
        b, t, n, _ = query_coords.shape
        rel = self.relative_time_tokens(b, t, n)
        return self.query_fuse(torch.cat([self.encode_space(spatial), self.encode_time(temporal), rel], dim=-1))

    def decode(self, latent_tokens: torch.Tensor, query_coords: torch.Tensor) -> torch.Tensor:
        if latent_tokens.ndim != 4:
            raise ValueError(f"Structured decoder expects [B,T,K,D], got {tuple(latent_tokens.shape)}")
        if latent_tokens.shape[1] != query_coords.shape[1]:
            raise ValueError("Latent and query time dimensions must match.")
        return self.decoder(self.encode_query_coords(query_coords), latent_tokens)

    def reconstruct(self, batch: STBatch, query_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        query_coords = build_full_query_coords(batch) if query_coords is None else query_coords
        return self.decode(self.encode(batch), query_coords)

    def forward(self, batch: STBatch, query_coords: Optional[torch.Tensor] = None) -> NeuralFieldOutput:
        query_coords = build_full_query_coords(batch) if query_coords is None else query_coords
        latents = self.encode(batch)
        return NeuralFieldOutput(
            latent_tokens=latents,
            reconstruction=self.decode(latents, query_coords),
            observation_mask=build_observation_mask(batch),
        )

    @torch.no_grad()
    def diagnostics(self, latent_grid: torch.Tensor) -> StructuredNeuralFieldDiagnostics:
        temporal = F.cosine_similarity(latent_grid[:, 1:], latent_grid[:, :-1], dim=-1).mean()
        if latent_grid.shape[2] > 1:
            spatial = F.normalize(latent_grid, dim=-1)
            sim = torch.matmul(spatial, spatial.transpose(-1, -2))
            eye = torch.eye(latent_grid.shape[2], device=latent_grid.device, dtype=torch.bool)
            off_diagonal = (~eye)[None, None].expand_as(sim)
            spatial_cosine = sim.masked_select(off_diagonal).mean()
        else:
            spatial_cosine = latent_grid.new_zeros(())
        return StructuredNeuralFieldDiagnostics(
            anchors=self.spatial_tokenizer.anchor_coords().detach(),
            latent_temporal_cosine=temporal,
            latent_spatial_cosine=spatial_cosine,
        )

def structured_stage1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    anchor_repulsion: torch.Tensor,
    charbonnier_eps: float = 1e-3,
    temporal_difference_weight: float = 0.05,
    anchor_repulsion_weight: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = torch.ones_like(target[..., 0]) if valid_mask is None else valid_mask.to(target.dtype)
    residual = prediction - target
    denom = mask.sum().clamp_min(1.0) * target.shape[-1]
    charbonnier = (torch.sqrt(residual.square() + charbonnier_eps**2) * mask[..., None]).sum() / denom
    mse = (residual.square() * mask[..., None]).sum() / denom
    if target.shape[1] > 1:
        diff_mask = mask[:, 1:] * mask[:, :-1]
        pred_diff = prediction[:, 1:] - prediction[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        diff_denom = diff_mask.sum().clamp_min(1.0) * target.shape[-1]
        temporal_difference = (
            (pred_diff - target_diff).abs() * diff_mask[..., None]
        ).sum() / diff_denom
    else:
        temporal_difference = prediction.new_zeros(())
    loss = (
        charbonnier
        + 0.1 * mse
        + temporal_difference_weight * temporal_difference
        + anchor_repulsion_weight * anchor_repulsion
    )
    return loss, {
        "charbonnier": charbonnier,
        "mse": mse,
        "temporal_difference": temporal_difference,
        "anchor_repulsion": anchor_repulsion,
    }

def domain_features_from_batch(batch: STBatch, feature_dim: int) -> torch.Tensor:
    features = batch.metadata.get("domain_features")
    if features is None:
        return batch.values.new_zeros(batch.batch_size, feature_dim)
    if features.ndim == 1:
        features = features[None, :].expand(batch.batch_size, -1)
    if features.shape != (batch.batch_size, feature_dim):
        raise ValueError(
            f"Expected domain_features [{batch.batch_size},{feature_dim}], got {tuple(features.shape)}"
        )
    return features.to(device=batch.values.device, dtype=batch.values.dtype)

class CanonicalDomainStructuredField(StructuredSpatioTemporalNeuralFieldAE):
    """Shared structured codec with physical time and data-derived domain context."""

    canonical_time_dim = 11

    def __init__(
        self,
        config: ModelConfig,
        domain_feature_dim: int,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self.domain_feature_dim = int(domain_feature_dim)
        self.time_fourier = GaussianFourierFeatures(
            self.canonical_time_dim,
            config.temporal_fourier_size,
            config.temporal_fourier_scale,
        )
        self.domain_encoder = nn.Sequential(
            nn.LayerNorm(domain_feature_dim),
            nn.Linear(domain_feature_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.domain_to_point = nn.Linear(config.latent_dim, config.token_dim)
        self.domain_to_time = nn.Linear(config.latent_dim, config.latent_dim)
        self.domain_film = nn.Linear(config.latent_dim, config.latent_dim * 2)
        for module in (self.domain_to_point, self.domain_to_time, self.domain_film):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _serial_hours(time_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        year, month, day, hour = [time_coords[..., index] for index in range(4)]
        month_lengths = time_coords.new_tensor(
            [31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0]
        )
        cumulative = torch.cat([month_lengths.new_zeros(1), month_lengths.cumsum(0)[:-1]])
        month_index = (month.round().long() - 1).clamp(0, 11)
        day_of_year = cumulative[month_index] + (day - 1.0).clamp_min(0.0)
        year_int = year.round().long()
        leap = (year_int.remainder(4) == 0) & (
            (year_int.remainder(100) != 0) | (year_int.remainder(400) == 0)
        )
        day_of_year = day_of_year + (leap & (month_index >= 2)).to(day_of_year.dtype)

        def leaps_before(year_value: torch.Tensor) -> torch.Tensor:
            previous = year_value - 1
            return torch.div(previous, 4, rounding_mode="floor") - torch.div(
                previous, 100, rounding_mode="floor"
            ) + torch.div(previous, 400, rounding_mode="floor")

        base_year = torch.full_like(year_int, 2000)
        days_before_year = (
            365 * (year_int - 2000)
            + leaps_before(year_int)
            - leaps_before(base_year)
        )
        serial = (days_before_year.to(day_of_year.dtype) + day_of_year) * 24.0 + hour
        return serial, day_of_year

    def calendar_time_to_periodic(self, time_coords: torch.Tensor) -> torch.Tensor:
        if time_coords.shape[-1] != 4:
            raise ValueError(
                "CanonicalDomainStructuredField requires [year,month,day,hour_float] time coordinates."
            )
        year, month, day, hour = [time_coords[..., index] for index in range(4)]
        serial, day_of_year = self._serial_hours(time_coords)
        if serial.ndim < 2:
            raise ValueError("Time coordinates must include a batch and time dimension.")
        first = serial.select(1, 0).unsqueeze(1)
        elapsed_days = (serial - first) / 24.0
        if serial.shape[1] > 1:
            deltas = (serial[:, 1:] - serial[:, :-1]).abs()
            delta = deltas.median(dim=1).values.unsqueeze(1)
        else:
            delta = torch.ones_like(first)
        log_delta = torch.log1p(delta).expand_as(serial)
        two_pi = 2.0 * torch.pi
        return torch.stack(
            [
                (year - 2000.0) / 100.0,
                torch.sin(two_pi * (month - 1.0) / 12.0),
                torch.cos(two_pi * (month - 1.0) / 12.0),
                torch.sin(two_pi * (day - 1.0) / 31.0),
                torch.cos(two_pi * (day - 1.0) / 31.0),
                torch.sin(two_pi * hour / 24.0),
                torch.cos(two_pi * hour / 24.0),
                torch.sin(two_pi * day_of_year / 365.2425),
                torch.cos(two_pi * day_of_year / 365.2425),
                elapsed_days,
                log_delta,
            ],
            dim=-1,
        )

    def domain_context(self, batch: STBatch) -> torch.Tensor:
        return self.domain_encoder(domain_features_from_batch(batch, self.domain_feature_dim))

    def point_tokens(
        self, batch: STBatch, values: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, valid = super().point_tokens(batch, values=values)
        domain = self.domain_to_point(self.domain_context(batch))[:, None, None, :]
        natural_valid = batch.metadata.get("valid_mask")
        if natural_valid is not None:
            valid = valid & natural_valid.to(torch.bool)
        return tokens + domain, valid

    def time_latent_tokens(self, batch: STBatch) -> torch.Tensor:
        tokens = super().time_latent_tokens(batch)
        return tokens + self.domain_to_time(self.domain_context(batch))[:, None, :]

    def encode(self, batch: STBatch) -> torch.Tensor:
        grid = self.encode_spatial_frames(batch)
        for block in self.axial_blocks:
            grid = block(grid)
        grid = self.latent_norm(grid)
        shift, scale = self.domain_film(self.domain_context(batch)).chunk(2, dim=-1)
        return grid * (1.0 + 0.1 * torch.tanh(scale[:, None, None, :])) + 0.1 * shift[:, None, None, :]

EARTH_RADIUS_KM = 6371.0088

FOUNDATION_DOMAIN_FEATURE_NAMES = (
    "log_delta_hours",
    "log_num_nodes",
    "log_value_scale",
    "log_abs_value_offset",
    "log_extent_km",
    "log_median_neighbor_km",
    "nonzero_fraction",
    "log_value_cv",
)

def lonlat_to_ecef_numpy(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    lon = np.deg2rad(coords[..., 0])
    lat = np.deg2rad(coords[..., 1])
    cos_lat = np.cos(lat)
    return np.stack(
        [cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=-1
    )

def lonlat_to_ecef_torch(coords: torch.Tensor) -> torch.Tensor:
    lon = torch.deg2rad(coords[..., 0])
    lat = torch.deg2rad(coords[..., 1])
    cos_lat = torch.cos(lat)
    return torch.stack(
        [cos_lat * torch.cos(lon), cos_lat * torch.sin(lon), torch.sin(lat)], dim=-1
    )

def _serial_hours_numpy(time_features: np.ndarray) -> np.ndarray:
    time_features = np.asarray(time_features, dtype=np.float64)
    year, month, day, hour = [time_features[:, index] for index in range(4)]
    month = np.rint(month).astype(np.int64).clip(1, 12)
    month_lengths = np.asarray(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64
    )
    cumulative = np.concatenate([np.zeros(1), np.cumsum(month_lengths)[:-1]])
    day_of_year = cumulative[month - 1] + day - 1.0
    year_int = np.rint(year).astype(np.int64)
    leap = (year_int % 4 == 0) & ((year_int % 100 != 0) | (year_int % 400 == 0))
    day_of_year += (leap & (month > 2)).astype(np.float64)

    def leaps_before(values: np.ndarray | int) -> np.ndarray:
        previous = np.asarray(values, dtype=np.int64) - 1
        return previous // 4 - previous // 100 + previous // 400

    days = 365 * (year_int - 2000) + leaps_before(year_int) - leaps_before(2000)
    return (days + day_of_year) * 24.0 + hour

def _sample_value_moments(domain: ProcessedDomain, maximum_values: int = 1_000_000) -> tuple[float, float, float]:
    # Scale experiments deliberately expose only an initial fraction of the
    # pretraining interval.  Domain descriptors must obey the same boundary so
    # the unused suffix cannot leak into the model through summary features.
    time_stop = getattr(domain, "train_time_stop", domain.split.train_time_stop)
    nodes = domain.split.train_nodes
    row_count = max(1, min(time_stop, maximum_values // max(len(nodes), 1)))
    sample = np.asarray(domain.values[:row_count, nodes], dtype=np.float64)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        raise ValueError(f"{domain.name}: no finite values in train-split descriptor sample")
    mean = float(finite.mean())
    std = float(finite.std())
    nonzero = float(np.count_nonzero(finite) / finite.size)
    return mean, std, nonzero

def _median_neighbor_km(ecef: np.ndarray, maximum_points: int = 512) -> float:
    if len(ecef) < 2:
        return 0.0
    if len(ecef) > maximum_points:
        ids = np.linspace(0, len(ecef) - 1, maximum_points, dtype=np.int64)
        ecef = ecef[ids]
    chord = np.linalg.norm(ecef[:, None, :] - ecef[None, :, :], axis=-1)
    np.fill_diagonal(chord, np.inf)
    nearest = chord.min(axis=1).clip(0.0, 2.0)
    angular = 2.0 * np.arcsin(np.clip(nearest * 0.5, 0.0, 1.0))
    return float(np.median(angular) * EARTH_RADIUS_KM)

class FoundationDomainSpec:
    name: str
    features: tuple[float, ...]
    ecef_center: tuple[float, float, float]
    local_chord_scale: float
    extent_km: float
    delta_hours: float
    num_nodes: int

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "features": dict(zip(FOUNDATION_DOMAIN_FEATURE_NAMES, self.features)),
            "ecef_center": list(self.ecef_center),
            "local_chord_scale": self.local_chord_scale,
            "extent_km": self.extent_km,
            "delta_hours": self.delta_hours,
            "num_nodes": self.num_nodes,
        }

def discover_foundation_domain_spec(domain: ProcessedDomain) -> FoundationDomainSpec:
    coords = np.asarray(domain.spatial, dtype=np.float64)
    if coords.shape[-1] != 2 or not np.isfinite(coords).all():
        raise ValueError(f"{domain.name}: expected finite [N,2] longitude/latitude coordinates")
    if (np.abs(coords[:, 1]) > 90.0 + 1e-5).any() or (np.abs(coords[:, 0]) > 180.0 + 1e-5).any():
        raise ValueError(f"{domain.name}: coordinates fall outside longitude/latitude bounds")
    ecef = lonlat_to_ecef_numpy(coords)
    center = ecef.mean(axis=0)
    radii = np.linalg.norm(ecef - center[None, :], axis=-1)
    local_scale = max(float(np.quantile(radii, 0.95)), 1e-6)
    center_norm = float(np.linalg.norm(center))
    if center_norm > 1e-4:
        center_direction = center / center_norm
        angular_radius = float(
            np.quantile(
                np.arccos(np.clip(ecef @ center_direction, -1.0, 1.0)),
                0.95,
            )
        )
    else:
        # A nearly uniform global grid has no meaningful spherical centroid.
        # Record the largest possible geodesic radius instead of pretending
        # that the Euclidean mean is a point on Earth.
        angular_radius = float(np.pi)
    extent_km = max(float(angular_radius * EARTH_RADIUS_KM), 1e-3)

    serial = _serial_hours_numpy(np.asarray(domain.time_features[: min(len(domain.time_features), 512)]))
    deltas = np.diff(serial)
    deltas = deltas[deltas > 1e-8]
    if not len(deltas):
        raise ValueError(f"{domain.name}: could not infer a positive sampling interval")
    delta_hours = float(np.median(deltas))
    mean, std, nonzero = _sample_value_moments(domain)
    value_cv = std / max(abs(mean), 1e-6)
    median_neighbor_km = _median_neighbor_km(ecef)
    features = (
        float(np.log1p(delta_hours)),
        float(np.log1p(domain.shape[1])),
        float(np.log1p(abs(domain.stats.value_scale))),
        float(np.log1p(abs(domain.stats.value_offset))),
        float(np.log1p(extent_km)),
        float(np.log1p(median_neighbor_km)),
        nonzero,
        float(np.log1p(value_cv)),
    )
    return FoundationDomainSpec(
        name=domain.name,
        features=features,
        ecef_center=tuple(float(value) for value in center),
        local_chord_scale=local_scale,
        extent_km=extent_km,
        delta_hours=delta_hours,
        num_nodes=domain.shape[1],
    )

def attach_foundation_metadata(batch: STBatch, spec: FoundationDomainSpec) -> STBatch:
    metadata = dict(batch.metadata)
    dtype = batch.values.dtype
    device = batch.values.device
    batch_size = batch.batch_size
    valid = torch.isfinite(batch.values).all(dim=-1)
    values = torch.nan_to_num(batch.values, nan=0.0, posinf=0.0, neginf=0.0)
    obs_mask = batch.obs_mask
    if obs_mask is not None:
        obs_mask = obs_mask * valid.to(obs_mask.dtype)
    if "eval_mask" in metadata:
        metadata["eval_mask"] = metadata["eval_mask"] * valid.to(metadata["eval_mask"].dtype)
    metadata["valid_mask"] = valid.to(dtype)
    metadata["domain_features"] = torch.tensor(spec.features, device=device, dtype=dtype)[None].expand(
        batch_size, -1
    )
    metadata["geo_center_ecef"] = torch.tensor(
        spec.ecef_center, device=device, dtype=dtype
    )[None].expand(batch_size, -1)
    metadata["geo_local_chord_scale"] = torch.full(
        (batch_size, 1), spec.local_chord_scale, device=device, dtype=dtype
    )
    metadata["geo_extent_km"] = torch.full(
        (batch_size, 1), spec.extent_km, device=device, dtype=dtype
    )
    metadata["delta_hours"] = torch.full(
        (batch_size,), spec.delta_hours, device=device, dtype=dtype
    )
    if "context_values" in metadata:
        context = metadata["context_values"]
        context_valid = torch.isfinite(context).all(dim=-1)
        metadata["context_values"] = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        if "context_obs_mask" in metadata:
            metadata["context_obs_mask"] = metadata["context_obs_mask"] * context_valid.to(
                metadata["context_obs_mask"].dtype
            )
        metadata["context_valid_mask"] = context_valid.to(dtype)
    return STBatch(
        values=values,
        spatial_coords=batch.spatial_coords,
        time_coords=batch.time_coords,
        node_mask=batch.node_mask,
        obs_mask=obs_mask,
        metadata=metadata,
    )

class FoundationDomainLoaderView:
    def __init__(self, loader: Iterable[STBatch], spec: FoundationDomainSpec) -> None:
        self.loader = loader
        self.spec = spec

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def __iter__(self):
        for batch in self.loader:
            yield attach_foundation_metadata(batch, self.spec)

class AdaptiveGlobalLocalStructuredField(CanonicalDomainStructuredField):
    """Shared field using global ECEF and adaptive per-domain local ECEF."""

    def __init__(
        self,
        config: ModelConfig,
        domain_feature_dim: int,
        global_fourier_scale: float = 2.0,
        local_fourier_scale: float = 6.0,
        **kwargs,
    ) -> None:
        super().__init__(config, domain_feature_dim=domain_feature_dim, **kwargs)
        # The base tokenizer historically used a fixed two-row grid, which is
        # acceptable for K=8 but degenerates into a 2x16 strip at K=32.  A
        # deterministic low-discrepancy initialization spreads foundation
        # slots across both local axes without giving them domain identities.
        num_slots = self.spatial_tokenizer.num_slots
        ids = torch.arange(num_slots, dtype=self.spatial_tokenizer.anchor_logits.dtype)
        golden = (5.0**0.5 - 1.0) / 2.0
        initial_x = 0.1 + 0.8 * torch.remainder((ids + 0.5) * golden, 1.0)
        initial_y = 0.1 + 0.8 * (ids + 0.5) / max(num_slots, 1)
        initial_anchors = torch.stack([initial_x, initial_y], dim=-1).clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            self.spatial_tokenizer.anchor_logits.copy_(torch.logit(initial_anchors))
        del self.space_fourier
        mapping_size = max(8, config.spatial_fourier_size)
        self.global_space_fourier = GaussianFourierFeatures(3, mapping_size, global_fourier_scale)
        self.local_space_fourier = GaussianFourierFeatures(3, mapping_size, local_fourier_scale)
        spatial_input_dim = self.global_space_fourier.out_dim + self.local_space_fourier.out_dim + 7
        self.space_proj = nn.Sequential(
            nn.Linear(spatial_input_dim, config.coord_embed_dim),
            nn.GELU(),
            nn.Linear(config.coord_embed_dim, config.coord_embed_dim),
        )
        self.register_buffer("active_geo_center", torch.zeros(1, 3), persistent=False)
        self.register_buffer("active_geo_scale", torch.ones(1, 1), persistent=False)
        self.register_buffer("active_geo_extent", torch.ones(1, 1), persistent=False)

    def activate_batch_geometry(self, batch: STBatch) -> None:
        center = batch.metadata.get("geo_center_ecef")
        scale = batch.metadata.get("geo_local_chord_scale")
        extent = batch.metadata.get("geo_extent_km")
        if center is None or scale is None or extent is None:
            raise KeyError("Adaptive global/local field requires geo_* metadata")
        center = center.reshape(batch.batch_size, 3)
        scale = scale.reshape(batch.batch_size, 1)
        extent = extent.reshape(batch.batch_size, 1)
        if not torch.allclose(center, center[:1].expand_as(center), atol=1e-6, rtol=1e-6):
            raise ValueError("Every batch must contain one domain geometry")
        if not torch.allclose(scale, scale[:1].expand_as(scale), atol=1e-6, rtol=1e-6):
            raise ValueError("Every batch must contain one domain scale")
        self.active_geo_center.copy_(center[:1].to(self.active_geo_center))
        self.active_geo_scale.copy_(scale[:1].clamp_min(1e-6).to(self.active_geo_scale))
        self.active_geo_extent.copy_(extent[:1].clamp_min(1e-3).to(self.active_geo_extent))

    # CanonicalDomainConditionEncoder calls this optional hook.  The name is
    # retained for interface compatibility; no domain id is read or learned.
    def activate_batch_domain(self, batch: STBatch) -> int:
        self.activate_batch_geometry(batch)
        return 0

    def domain_context(self, batch: STBatch) -> torch.Tensor:
        self.activate_batch_geometry(batch)
        return super().domain_context(batch)

    def point_tokens(
        self, batch: STBatch, values: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.activate_batch_geometry(batch)
        return super().point_tokens(batch, values=values)

    def metric_coordinates(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_ecef = lonlat_to_ecef_torch(coords)
        center = self.active_geo_center.to(global_ecef.device, global_ecef.dtype)
        scale = self.active_geo_scale.to(global_ecef.device, global_ecef.dtype)
        center = center.view(*([1] * (global_ecef.ndim - 1)), 3)
        scale = scale.view(*([1] * (global_ecef.ndim - 1)), 1)
        local = (global_ecef - center) / scale.clamp_min(1e-6)
        return global_ecef, local

    def encode_space(self, coords: torch.Tensor) -> torch.Tensor:
        global_ecef, local = self.metric_coordinates(coords)
        extent = self.active_geo_extent.to(coords.device, coords.dtype)
        extent_feature = torch.log1p(extent) / float(np.log1p(np.pi * EARTH_RADIUS_KM))
        extent_feature = extent_feature.view(*([1] * (coords.ndim - 1)), 1).expand(
            *coords.shape[:-1], 1
        )
        features = torch.cat(
            [
                self.global_space_fourier(global_ecef),
                self.local_space_fourier(local),
                global_ecef,
                local,
                extent_feature,
            ],
            dim=-1,
        )
        return self.space_proj(features)

    @torch.no_grad()
    def geometry_diagnostics(self, batch: STBatch) -> dict[str, object]:
        self.activate_batch_geometry(batch)
        global_ecef, local = self.metric_coordinates(batch.spatial_coords)
        return {
            "global_ecef_min": global_ecef.amin(dim=tuple(range(global_ecef.ndim - 1))).cpu().tolist(),
            "global_ecef_max": global_ecef.amax(dim=tuple(range(global_ecef.ndim - 1))).cpu().tolist(),
            "local_min": local.amin(dim=tuple(range(local.ndim - 1))).cpu().tolist(),
            "local_max": local.amax(dim=tuple(range(local.ndim - 1))).cpu().tolist(),
            "extent_km": float(self.active_geo_extent.item()),
            "local_chord_scale": float(self.active_geo_scale.item()),
        }

class SlotRefinementBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        normalized = self.attn_norm(slots)
        slots = slots + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        return slots + self.ff(self.ff_norm(slots))

class GeometryGroundedSpatialTokenizer(nn.Module):
    """Global tokens plus local tokens supported by balanced coordinate regions."""

    requires_geometry_tokens = True

    def __init__(
        self,
        dim: int,
        total_slots: int,
        global_slots: int,
        heads: int,
        layers: int = 2,
        assignment_temperature: float = 0.25,
        sinkhorn_iterations: int = 4,
        inject_geometry_content: bool = True,
        inject_time_content: bool = True,
    ) -> None:
        super().__init__()
        if not 0 < global_slots < total_slots:
            raise ValueError("global_slots must be in [1,total_slots-1]")
        if assignment_temperature <= 0 or sinkhorn_iterations < 1:
            raise ValueError("Invalid balanced-assignment configuration")
        self.total_slots = int(total_slots)
        self.global_slots = int(global_slots)
        self.local_slots = int(total_slots - global_slots)
        self.assignment_temperature = float(assignment_temperature)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.inject_geometry_content = bool(inject_geometry_content)
        self.inject_time_content = bool(inject_time_content)

        self.global_queries = nn.Parameter(torch.randn(global_slots, dim) * 0.02)
        self.geometry_queries = nn.Parameter(torch.randn(self.local_slots, dim) * 0.02)
        self.geometry_key = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.support_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.point_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.global_blocks = nn.ModuleList(
            [SpatialSlotBlock(dim, heads) for _ in range(max(1, layers))]
        )
        self.refinement = nn.ModuleList(
            [SlotRefinementBlock(dim, heads) for _ in range(max(1, layers))]
        )
        self.global_type = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.local_type = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.out_norm = nn.LayerNorm(dim)
        self._last_assignment: Optional[torch.Tensor] = None

    @property
    def num_slots(self) -> int:
        return self.total_slots

    def anchor_coords(self) -> torch.Tensor:
        """Compatibility-only visualization; decoding never uses fake anchors."""

        ids = torch.arange(
            self.total_slots,
            device=self.geometry_queries.device,
            dtype=self.geometry_queries.dtype,
        )
        golden = (5.0**0.5 - 1.0) / 2.0
        return torch.stack(
            [torch.remainder((ids + 0.5) * golden, 1.0), (ids + 0.5) / self.total_slots],
            dim=-1,
        )

    def repulsion_loss(self, margin: float = 0.15) -> torch.Tensor:
        queries = F.normalize(self.geometry_queries, dim=-1)
        similarity = queries @ queries.transpose(0, 1)
        eye = torch.eye(self.local_slots, device=queries.device, dtype=torch.bool)
        return F.relu(similarity - (1.0 - margin)).square().masked_fill(eye, 0).mean()

    def geometry_assignment(
        self,
        geometry_tokens: torch.Tensor,
        geometry_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return balanced, geometry-only assignment weights ``[B,K_local,N]``."""

        batch, nodes, _ = geometry_tokens.shape
        if geometry_mask is None:
            geometry_mask = torch.ones(
                batch, nodes, device=geometry_tokens.device, dtype=torch.bool
            )
        else:
            geometry_mask = geometry_mask.to(torch.bool)
        if not geometry_mask.any(dim=-1).all():
            raise ValueError("Every geometry row must contain at least one valid node")

        keys = F.normalize(self.geometry_key(geometry_tokens), dim=-1)
        queries = F.normalize(self.geometry_queries, dim=-1)
        logits = torch.einsum("kd,bnd->bkn", queries, keys)
        logits = logits / self.assignment_temperature
        logits = logits.masked_fill(~geometry_mask[:, None, :], -torch.inf)
        logits = logits - logits.amax(dim=-1, keepdim=True)
        weights = logits.exp() * geometry_mask[:, None, :].to(logits.dtype)
        valid_count = geometry_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(logits.dtype)
        column_target = self.local_slots / valid_count
        for _ in range(self.sinkhorn_iterations):
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            column_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            weights = weights / column_sum * column_target[:, None, :]
            weights = weights * geometry_mask[:, None, :].to(weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def context_from_geometry(
        self,
        geometry_tokens: torch.Tensor,
        geometry_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assignment = self.geometry_assignment(geometry_tokens, geometry_mask)
        if geometry_mask is None:
            geometry_mask = torch.ones(
                geometry_tokens.shape[:2],
                device=geometry_tokens.device,
                dtype=torch.bool,
            )
        mask = geometry_mask.to(geometry_tokens.dtype)
        global_mean = (geometry_tokens * mask[..., None]).sum(dim=1)
        global_mean = global_mean / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_context = global_mean[:, None, :].expand(-1, self.global_slots, -1)
        global_context = global_context + self.global_type
        local_context = torch.einsum("bkn,bnd->bkd", assignment, geometry_tokens)
        local_context = local_context + self.local_type
        return torch.cat([global_context, local_context], dim=1), assignment

    def forward(
        self,
        point_tokens: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        time_tokens: Optional[torch.Tensor] = None,
        geometry_tokens: Optional[torch.Tensor] = None,
        geometry_mask: Optional[torch.Tensor] = None,
        return_cross_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if geometry_tokens is None:
            raise ValueError("GeometryGroundedSpatialTokenizer requires geometry_tokens")
        if point_mask is None:
            point_mask = torch.ones(
                point_tokens.shape[:2], device=point_tokens.device, dtype=torch.bool
            )
        point_mask = point_mask.to(torch.bool)
        safe_point_mask = point_mask.clone()
        safe_point_tokens = point_tokens.clone()
        empty_rows = ~safe_point_mask.any(dim=-1)
        if empty_rows.any():
            safe_point_mask[empty_rows, 0] = True
            safe_point_tokens[empty_rows, 0] = 0
        support_context, assignment = self.context_from_geometry(
            geometry_tokens, geometry_mask
        )

        global_tokens = self.global_queries[None].expand(point_tokens.shape[0], -1, -1)
        global_attention = None
        for index, block in enumerate(self.global_blocks):
            capture = return_cross_attention and index == len(self.global_blocks) - 1
            result = block(
                global_tokens,
                safe_point_tokens,
                safe_point_mask,
                return_cross_attention=capture,
            )
            if capture:
                global_tokens, global_attention = result
            else:
                global_tokens = result

        observed_weights = assignment * point_mask[:, None, :].to(assignment.dtype)
        observed_weights = observed_weights / observed_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        pooled = torch.einsum("bkn,bnd->bkd", observed_weights, point_tokens)
        local_tokens = self.geometry_queries[None] + self.point_proj(pooled)
        if self.inject_geometry_content:
            local_tokens = local_tokens + self.support_proj(
                support_context[:, self.global_slots :]
            )
            global_tokens = global_tokens + self.support_proj(
                support_context[:, : self.global_slots]
            )
        slots = torch.cat([global_tokens, local_tokens], dim=1)
        if time_tokens is not None and self.inject_time_content:
            slots = slots + time_tokens[:, None, :]
        for block in self.refinement:
            slots = block(slots)
        slots = self.out_norm(slots)
        self._last_assignment = assignment.detach()

        if return_cross_attention:
            if global_attention is None:
                valid = point_mask.to(slots.dtype)
                global_attention = valid[:, None, :] / valid.sum(
                    dim=-1, keepdim=True
                )[:, None, :].clamp_min(1.0)
                global_attention = global_attention.expand(-1, self.global_slots, -1)
            attention = torch.cat([global_attention, assignment], dim=1)
            return slots, attention
        return slots

class GeometryGroundedQueryDecoder(nn.Module):
    """Content attention augmented with spatial and physical-time metric bias."""

    def __init__(
        self,
        dim: int,
        output_dim: int,
        heads: int,
        temporal_radius: int,
        spatial_feature_dim: int = 9,
        temporal_feature_dim: int = 7,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.temporal_radius = int(temporal_radius)
        self.local_q_norm = nn.LayerNorm(dim)
        self.local_z_norm = nn.LayerNorm(dim)
        self.local_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.neighbour_q_norm = nn.LayerNorm(dim)
        self.neighbour_z_norm = nn.LayerNorm(dim)
        self.neighbour_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.context_proj = nn.Linear(dim, dim)
        self.spatial_bias = nn.Sequential(
            nn.Linear(spatial_feature_dim, max(32, dim // 2)),
            nn.SiLU(),
            nn.Linear(max(32, dim // 2), heads),
        )
        self.temporal_bias = nn.Sequential(
            nn.Linear(temporal_feature_dim, max(16, dim // 4)),
            nn.SiLU(),
            nn.Linear(max(16, dim // 4), heads),
        )
        nn.init.zeros_(self.spatial_bias[-1].weight)
        nn.init.zeros_(self.spatial_bias[-1].bias)
        nn.init.zeros_(self.temporal_bias[-1].weight)
        nn.init.zeros_(self.temporal_bias[-1].bias)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.to_out = nn.Linear(dim, output_dim)

    def forward(
        self,
        queries: torch.Tensor,
        latent_grid: torch.Tensor,
        slot_context: torch.Tensor,
        spatial_relative: torch.Tensor,
        temporal_relative: torch.Tensor,
    ) -> torch.Tensor:
        b, t, n, d = queries.shape
        k = latent_grid.shape[2]
        local = latent_grid + self.context_proj(slot_context)
        local = local.reshape(b * t, k, d)
        q = queries.reshape(b * t, n, d)
        spatial = self.spatial_bias(spatial_relative)
        spatial = spatial.permute(0, 1, 4, 2, 3).reshape(b * t * self.heads, n, k)
        x = q + self.local_attn(
            self.local_q_norm(q),
            self.local_z_norm(local),
            self.local_z_norm(local),
            attn_mask=spatial,
            need_weights=False,
        )[0]

        offsets = torch.arange(
            -self.temporal_radius,
            self.temporal_radius + 1,
            device=queries.device,
        )
        width = len(offsets)
        raw_indices = torch.arange(t, device=queries.device)[:, None] + offsets[None]
        valid = (raw_indices >= 0) & (raw_indices < t)
        indices = raw_indices.clamp(0, t - 1)
        neighbourhood = (latent_grid + self.context_proj(slot_context))[:, indices]
        neighbourhood = neighbourhood.reshape(b * t, width * k, d)

        spatial_heads = self.spatial_bias(spatial_relative).permute(0, 1, 4, 2, 3)
        time_heads = self.temporal_bias(temporal_relative).permute(0, 1, 3, 2)
        combined = spatial_heads.unsqueeze(-2) + time_heads[:, :, :, None, :, None]
        invalid = (~valid)[None, :, None, None, :, None]
        combined = combined.masked_fill(invalid, torch.finfo(combined.dtype).min)
        combined = combined.reshape(b * t * self.heads, n, width * k)
        context = self.neighbour_z_norm(neighbourhood)
        x = x + self.neighbour_attn(
            self.neighbour_q_norm(x),
            context,
            context,
            attn_mask=combined,
            need_weights=False,
        )[0]
        x = x + self.ff(self.ff_norm(x))
        return self.to_out(x).reshape(b, t, n, -1)

class CoordinateGroundedFoundationField(AdaptiveGlobalLocalStructuredField):
    """Foundation codec whose local latent slots carry explicit metric support."""

    def __init__(
        self,
        config: ModelConfig,
        domain_feature_dim: int,
        num_global_spatial_latents: int = 12,
        assignment_temperature: float = 0.25,
        sinkhorn_iterations: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(config, domain_feature_dim=domain_feature_dim, **kwargs)
        heads = kwargs.get("heads") or config.self_heads
        layers = int(kwargs.get("spatial_tokenizer_layers", 2))
        self.num_global_spatial_latents = int(num_global_spatial_latents)
        self.spatial_tokenizer = GeometryGroundedSpatialTokenizer(
            config.latent_dim,
            self.num_spatial_latents,
            self.num_global_spatial_latents,
            heads,
            layers,
            assignment_temperature,
            sinkhorn_iterations,
        )
        self.decoder = GeometryGroundedQueryDecoder(
            config.decoder_query_dim,
            config.value_dim,
            config.cross_heads,
            int(kwargs.get("decoder_temporal_radius", 2)),
        )
        self.decoder_domain_film = nn.Linear(config.latent_dim, 2 * config.decoder_query_dim)
        self.decoder_context_encoder = nn.Sequential(
            nn.Linear(4, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.decoder_output_film = nn.Linear(config.latent_dim, 2 * config.value_dim)
        self.domain_to_slot = nn.Linear(config.latent_dim, config.latent_dim)
        self.register_buffer("domain_feature_mean", torch.zeros(domain_feature_dim))
        self.register_buffer("domain_feature_std", torch.ones(domain_feature_dim))
        for module in (
            self.decoder_domain_film,
            self.decoder_output_film,
            self.domain_to_slot,
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def set_router_reference_features(self, features) -> None:
        """Fit the descriptor transform on pretraining domains only."""

        reference = torch.as_tensor(
            features,
            device=self.domain_feature_mean.device,
            dtype=self.domain_feature_mean.dtype,
        )
        if reference.ndim != 2 or reference.shape[1] != self.domain_feature_dim:
            raise ValueError(
                f"Expected domain features [domains,{self.domain_feature_dim}], got "
                f"{tuple(reference.shape)}"
            )
        self.domain_feature_mean.copy_(reference.mean(dim=0))
        self.domain_feature_std.copy_(reference.std(dim=0, unbiased=False).clamp_min(1e-4))

    def normalize_domain_features(self, features: torch.Tensor) -> torch.Tensor:
        mean = self.domain_feature_mean.to(features)
        std = self.domain_feature_std.to(features)
        return ((features - mean) / std.clamp_min(1e-4)).clamp(-6.0, 6.0)

    def domain_context(self, batch: STBatch) -> torch.Tensor:
        features = domain_features_from_batch(batch, self.domain_feature_dim)
        return self.domain_encoder(self.normalize_domain_features(features))

    @staticmethod
    def _geometry_mask(batch: STBatch) -> torch.Tensor:
        mask = torch.ones(
            batch.batch_size,
            batch.time_steps,
            batch.max_nodes,
            device=batch.values.device,
            dtype=torch.bool,
        )
        if batch.node_mask is not None:
            mask = mask & batch.node_mask[:, None, :].to(torch.bool)
        return mask

    def geometry_tokens_and_mask(self, batch: STBatch) -> tuple[torch.Tensor, torch.Tensor]:
        self.activate_batch_geometry(batch)
        tokens = self.encode_space(expand_spatial_coords(batch))
        return tokens, self._geometry_mask(batch)

    def _encode_spatial_frames(
        self,
        batch: STBatch,
        values: Optional[torch.Tensor] = None,
        observation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        point_tokens, valid = self.point_tokens(batch, values=values)
        geometry_tokens, geometry_mask = self.geometry_tokens_and_mask(batch)
        if observation_mask is not None:
            valid = valid & observation_mask.to(torch.bool)
        b, t, n, d = point_tokens.shape
        time_tokens = self.time_latent_tokens(batch).reshape(b * t, d)
        return self.spatial_tokenizer(
            point_tokens.reshape(b * t, n, d),
            point_mask=valid.reshape(b * t, n),
            time_tokens=time_tokens,
            geometry_tokens=geometry_tokens.reshape(b * t, n, d),
            geometry_mask=geometry_mask.reshape(b * t, n),
        ).reshape(b, t, self.num_spatial_latents, d)

    def encode_spatial_frames(
        self, batch: STBatch, values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self._encode_spatial_frames(batch, values=values)

    def encode_with_observation_mask(
        self, batch: STBatch, observation_mask: torch.Tensor
    ) -> torch.Tensor:
        masked_values = batch.values * observation_mask[..., None].to(batch.values.dtype)
        grid = self._encode_spatial_frames(
            batch,
            values=masked_values,
            observation_mask=observation_mask,
        )
        for block in self.axial_blocks:
            grid = block(grid)
        grid = self.latent_norm(grid)
        shift, scale = self.domain_film(self.domain_context(batch)).chunk(2, dim=-1)
        return grid * (1.0 + 0.1 * torch.tanh(scale[:, None, None, :])) + 0.1 * shift[
            :, None, None, :
        ]

    def _decoder_context(self, batch: STBatch, condition_batch: Optional[STBatch]) -> torch.Tensor:
        context_batch = batch if condition_batch is None else condition_batch
        mask = self._geometry_mask(context_batch).to(context_batch.values.dtype)
        if context_batch.obs_mask is not None:
            mask = mask * context_batch.obs_mask.to(mask.dtype)
        natural_valid = context_batch.metadata.get("valid_mask")
        if natural_valid is not None:
            mask = mask * natural_valid.to(mask.dtype)
        expanded = mask[..., None]
        count = expanded.sum(dim=(1, 2, 3)).clamp_min(1.0)
        mean = (context_batch.values * expanded).sum(dim=(1, 2, 3)) / count
        centered = (context_batch.values - mean[:, None, None, None]) * expanded
        std = (centered.square().sum(dim=(1, 2, 3)) / count).sqrt()
        abs_mean = (context_batch.values.abs() * expanded).sum(dim=(1, 2, 3)) / count
        fraction = mask.sum(dim=(1, 2)) / self._geometry_mask(context_batch).sum(
            dim=(1, 2)
        ).clamp_min(1)
        observed = torch.stack([mean, std, abs_mean, fraction], dim=-1)
        return self.domain_context(batch) + self.decoder_context_encoder(observed)

    def _slot_context(self, batch: STBatch) -> torch.Tensor:
        geometry, mask = self.geometry_tokens_and_mask(batch)
        b, t, n, d = geometry.shape
        context, _ = self.spatial_tokenizer.context_from_geometry(
            geometry.reshape(b * t, n, d), mask.reshape(b * t, n)
        )
        context = context.reshape(b, t, self.num_spatial_latents, d)
        time = self.time_latent_tokens(batch)[:, :, None, :]
        domain = self.domain_to_slot(self.domain_context(batch))[:, None, None, :]
        return context + time + domain

    def _spatial_relative_features(
        self, batch: STBatch, query_coords: torch.Tensor
    ) -> torch.Tensor:
        support_spatial = expand_spatial_coords(batch)
        query_spatial = query_coords[..., : self.config.spatial_dim]
        geometry, mask = self.geometry_tokens_and_mask(batch)
        b, t, n, d = geometry.shape
        assignment = self.spatial_tokenizer.geometry_assignment(
            geometry.reshape(b * t, n, d), mask.reshape(b * t, n)
        ).reshape(b, t, self.spatial_tokenizer.local_slots, n)
        support_global_nodes, support_local_nodes = self.metric_coordinates(support_spatial)
        query_global, query_local = self.metric_coordinates(query_spatial)
        support_global = torch.einsum("btkn,btnd->btkd", assignment, support_global_nodes)
        support_local = torch.einsum("btkn,btnd->btkd", assignment, support_local_nodes)
        delta_global = query_global[:, :, :, None, :] - support_global[:, :, None, :, :]
        delta_local = query_local[:, :, :, None, :] - support_local[:, :, None, :, :]
        chord = delta_global.square().sum(dim=-1, keepdim=True).sqrt()
        local_distance = delta_local.square().sum(dim=-1, keepdim=True).sqrt()
        local_flag = torch.ones_like(chord)
        local_features = torch.cat(
            [delta_global, delta_local, chord, local_distance, local_flag], dim=-1
        )
        query_nodes = query_spatial.shape[2]
        global_features = local_features.new_zeros(
            b,
            t,
            query_nodes,
            self.num_global_spatial_latents,
            local_features.shape[-1],
        )
        return torch.cat([global_features, local_features], dim=3)

    def _temporal_relative_features(self, batch: STBatch) -> torch.Tensor:
        temporal = batch.time_coords if batch.time_coords.ndim == 3 else batch.time_coords[:, :, 0]
        serial, _ = self._serial_hours(temporal)
        t = serial.shape[1]
        offsets = torch.arange(
            -self.decoder.temporal_radius,
            self.decoder.temporal_radius + 1,
            device=serial.device,
        )
        indices = (torch.arange(t, device=serial.device)[:, None] + offsets[None]).clamp(0, t - 1)
        neighbour = serial[:, indices]
        delta = neighbour - serial[:, :, None]
        if t > 1:
            cadence = (serial[:, 1:] - serial[:, :-1]).abs().median(dim=1).values
        else:
            cadence = torch.ones(serial.shape[0], device=serial.device, dtype=serial.dtype)
        normalized = delta / cadence[:, None, None].clamp_min(1e-6)
        absolute = normalized.abs()
        signed_log_hours = delta.sign() * torch.log1p(delta.abs())
        return torch.stack(
            [
                delta / 24.0,
                signed_log_hours,
                normalized,
                torch.exp(-0.5 * absolute.square()),
                torch.exp(-0.5 * (absolute / 3.0).square()),
                torch.exp(-0.5 * (absolute / 6.0).square()),
                torch.exp(-0.5 * (absolute / 12.0).square()),
            ],
            dim=-1,
        )

    def decode_with_batch(
        self,
        latent_tokens: torch.Tensor,
        query_coords: torch.Tensor,
        output_batch: STBatch,
        condition_batch: Optional[STBatch] = None,
    ) -> torch.Tensor:
        self.activate_batch_geometry(output_batch)
        queries = self.encode_query_coords(query_coords)
        decoder_context = self._decoder_context(output_batch, condition_batch)
        shift, scale = self.decoder_domain_film(decoder_context).chunk(2, dim=-1)
        queries = queries * (1.0 + 0.1 * torch.tanh(scale[:, None, None, :]))
        queries = queries + 0.1 * shift[:, None, None, :]
        # The latent slots describe the geometry that was visible to the
        # conditioner/encoder.  This distinction matters for held-out-node
        # evaluation, where output_batch contains only target nodes while the
        # condition batch also contains the observed support nodes.
        support_batch = output_batch if condition_batch is None else condition_batch
        prediction = self.decoder(
            queries,
            latent_tokens,
            self._slot_context(support_batch),
            self._spatial_relative_features(support_batch, query_coords),
            self._temporal_relative_features(output_batch),
        )
        output_shift, output_scale = self.decoder_output_film(decoder_context).chunk(2, dim=-1)
        return prediction * (1.0 + 0.1 * torch.tanh(output_scale[:, None, None, :])) + 0.1 * output_shift[
            :, None, None, :
        ]

    def forward(
        self, batch: STBatch, query_coords: Optional[torch.Tensor] = None
    ) -> NeuralFieldOutput:
        query_coords = build_full_query_coords(batch) if query_coords is None else query_coords
        latent = self.encode(batch)
        return NeuralFieldOutput(
            latent_tokens=latent,
            reconstruction=self.decode_with_batch(latent, query_coords, batch, batch),
            observation_mask=build_observation_mask(batch),
        )

    @torch.no_grad()
    def geometry_diagnostics(self, batch: STBatch) -> dict[str, object]:
        diagnostics = super().geometry_diagnostics(batch)
        geometry, mask = self.geometry_tokens_and_mask(batch)
        b, t, n, d = geometry.shape
        assignment = self.spatial_tokenizer.geometry_assignment(
            geometry.reshape(b * t, n, d), mask.reshape(b * t, n)
        )
        entropy = -(assignment.clamp_min(1e-8).log() * assignment).sum(dim=-1).mean()
        usage = assignment.sum(dim=1)
        valid_usage = usage[mask.reshape(b * t, n)]
        diagnostics.update(
            {
                "global_spatial_latents": self.num_global_spatial_latents,
                "local_grounded_latents": self.spatial_tokenizer.local_slots,
                "assignment_entropy": float(entropy.cpu()),
                "assignment_usage_min": float(valid_usage.min().cpu()),
                "assignment_usage_max": float(valid_usage.max().cpu()),
                "domain_feature_standardization": "pretrain_domain_mean_std_clipped_6sigma",
            }
        )
        return diagnostics

VALUE_ANCHOR_DIM = 4

VALUE_DEPENDENT_DOMAIN_FEATURES = tuple(
    FOUNDATION_DOMAIN_FEATURE_NAMES.index(name)
    for name in (
        "log_value_scale",
        "log_abs_value_offset",
        "nonzero_fraction",
        "log_value_cv",
    )
)

class ValueCanonicalFoundationFieldV1(CoordinateGroundedFoundationField):
    """Coordinate-grounded field with no domain-dependent content affine."""

    value_canonical_version = "v1_shortcut_free"

    def __init__(self, config: ModelConfig, domain_feature_dim: int, **kwargs) -> None:
        super().__init__(config, domain_feature_dim, **kwargs)
        # Retain these modules in the state-dict layout for compatibility with
        # the parent implementation, but make the ablation explicit: they are
        # neither called nor optimized.
        for module in (
            self.domain_encoder,
            self.domain_to_point,
            self.domain_to_time,
            self.domain_film,
            self.decoder_domain_film,
            self.decoder_context_encoder,
            self.decoder_output_film,
            self.domain_to_slot,
        ):
            module.requires_grad_(False)

    def normalize_domain_features(self, features: torch.Tensor) -> torch.Tensor:
        normalized = super().normalize_domain_features(features)
        normalized = normalized.clone()
        normalized[..., list(VALUE_DEPENDENT_DOMAIN_FEATURES)] = 0
        return normalized

    def point_tokens(
        self, batch: STBatch, values: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Deliberately call the non-domain base implementation. Geometry still
        # enters through encode_space and the separate grounded assignment.
        tokens, valid = StructuredSpatioTemporalNeuralFieldAE.point_tokens(
            self, batch, values=values
        )
        natural_valid = batch.metadata.get("valid_mask")
        if natural_valid is not None:
            valid = valid & natural_valid.to(torch.bool)
        return tokens, valid

    def time_latent_tokens(self, batch: STBatch) -> torch.Tensor:
        return StructuredSpatioTemporalNeuralFieldAE.time_latent_tokens(self, batch)

    def encode(self, batch: STBatch) -> torch.Tensor:
        grid = self.encode_spatial_frames(batch)
        for block in self.axial_blocks:
            grid = block(grid)
        return self.latent_norm(grid)

    def encode_with_observation_mask(
        self, batch: STBatch, observation_mask: torch.Tensor
    ) -> torch.Tensor:
        masked_values = batch.values * observation_mask[..., None].to(batch.values.dtype)
        grid = self._encode_spatial_frames(
            batch,
            values=masked_values,
            observation_mask=observation_mask,
        )
        for block in self.axial_blocks:
            grid = block(grid)
        return self.latent_norm(grid)

    def _slot_context(self, batch: STBatch) -> torch.Tensor:
        geometry, mask = self.geometry_tokens_and_mask(batch)
        b, t, n, d = geometry.shape
        context, _ = self.spatial_tokenizer.context_from_geometry(
            geometry.reshape(b * t, n, d), mask.reshape(b * t, n)
        )
        context = context.reshape(b, t, self.num_spatial_latents, d)
        return context + self.time_latent_tokens(batch)[:, :, None, :]

    def decode_with_batch(
        self,
        latent_tokens: torch.Tensor,
        query_coords: torch.Tensor,
        output_batch: STBatch,
        condition_batch: Optional[STBatch] = None,
    ) -> torch.Tensor:
        self.activate_batch_geometry(output_batch)
        queries = self.encode_query_coords(query_coords)
        support_batch = output_batch if condition_batch is None else condition_batch
        return self.decoder(
            queries,
            latent_tokens,
            self._slot_context(support_batch),
            self._spatial_relative_features(support_batch, query_coords),
            self._temporal_relative_features(output_batch),
        )

class ValueAnchoredFoundationFieldV2(ValueCanonicalFoundationFieldV1):
    """Value-only latent content with explicit, transferable slot moments."""

    value_canonical_version = "v2_value_anchored"
    value_anchor_dim = VALUE_ANCHOR_DIM

    def __init__(self, config: ModelConfig, domain_feature_dim: int, **kwargs) -> None:
        if config.value_dim != 1 or config.latent_dim <= VALUE_ANCHOR_DIM:
            raise ValueError("Value-anchored V2 requires value_dim=1 and latent_dim>4")
        super().__init__(config, domain_feature_dim, **kwargs)
        heads = kwargs.get("heads") or config.self_heads
        layers = int(kwargs.get("spatial_tokenizer_layers", 2))
        self.spatial_tokenizer = GeometryGroundedSpatialTokenizer(
            config.latent_dim,
            self.num_spatial_latents,
            self.num_global_spatial_latents,
            heads,
            layers,
            float(kwargs.get("assignment_temperature", 0.25)),
            int(kwargs.get("sinkhorn_iterations", 4)),
            inject_geometry_content=False,
            inject_time_content=False,
        )
        self.value_content_proj = nn.Sequential(
            nn.Linear(4, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, config.token_dim),
        )
        self.anchor_temperature_unconstrained = nn.Parameter(torch.tensor(-0.43275213))

    def value_content_tokens(
        self,
        values: torch.Tensor,
        observation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        delta = torch.zeros_like(values)
        delta[:, 1:] = values[:, 1:] - values[:, :-1]
        if observation_mask is not None:
            valid_pair = observation_mask[:, 1:] * observation_mask[:, :-1]
            delta[:, 1:] = delta[:, 1:] * valid_pair[..., None].to(delta.dtype)
        signed_log = values.sign() * torch.log1p(values.abs())
        features = torch.cat([values, values.abs(), signed_log, delta], dim=-1)
        return self.value_content_proj(features)

    def point_tokens(
        self, batch: STBatch, values: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = batch.values if values is None else values
        valid = torch.ones(
            batch.batch_size,
            batch.time_steps,
            batch.max_nodes,
            device=values.device,
            dtype=torch.bool,
        )
        if batch.node_mask is not None:
            valid = valid & batch.node_mask[:, None, :].to(torch.bool)
        natural_valid = batch.metadata.get("valid_mask")
        if natural_valid is not None:
            valid = valid & natural_valid.to(torch.bool)
        tokens = self.value_content_tokens(
            values, observation_mask=valid.to(values.dtype)
        )
        return tokens, valid

    def time_latent_tokens(self, batch: STBatch) -> torch.Tensor:
        return torch.zeros(
            batch.batch_size,
            batch.time_steps,
            self.config.latent_dim,
            device=batch.values.device,
            dtype=batch.values.dtype,
        )

    def value_anchors(
        self,
        batch: STBatch,
        values: Optional[torch.Tensor] = None,
        observation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        values = batch.values if values is None else values
        scalar = values[..., 0]
        valid = torch.ones_like(scalar, dtype=torch.bool)
        if batch.node_mask is not None:
            valid = valid & batch.node_mask[:, None, :].to(torch.bool)
        natural_valid = batch.metadata.get("valid_mask")
        if natural_valid is not None:
            valid = valid & natural_valid.to(torch.bool)
        if observation_mask is not None:
            valid = valid & observation_mask.to(torch.bool)

        geometry, geometry_mask = self.geometry_tokens_and_mask(batch)
        b, t, n, d = geometry.shape
        local_assignment = self.spatial_tokenizer.geometry_assignment(
            geometry.reshape(b * t, n, d), geometry_mask.reshape(b * t, n)
        ).reshape(b, t, self.spatial_tokenizer.local_slots, n)
        uniform = valid.to(values.dtype)
        uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
        global_assignment = uniform[:, :, None, :].expand(
            -1, -1, self.num_global_spatial_latents, -1
        )
        local_weights = local_assignment * valid[:, :, None, :].to(values.dtype)
        local_weights = local_weights / local_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        weights = torch.cat([global_assignment, local_weights], dim=2)
        mean = torch.einsum("btkn,btn->btk", weights, scalar)
        centered = scalar[:, :, None, :] - mean[..., None]
        variance = torch.einsum("btkn,btkn->btk", weights, centered.square())
        deviation = torch.einsum("btkn,btkn->btk", weights, centered.abs())
        delta = torch.zeros_like(mean)
        delta[:, 1:] = mean[:, 1:] - mean[:, :-1]
        # ``sqrt(variance)`` has an infinite derivative at zero. Constant
        # windows and one-observation local slots occur frequently in these
        # datasets, so the naïve expression poisoned the geometry-assignment
        # gradients after only a few optimizer steps. This shifted square root
        # is exactly zero at variance=0 and has a bounded derivative.
        variance_epsilon = 1e-4
        stable_std = torch.sqrt(variance.clamp_min(0) + variance_epsilon)
        stable_std = (stable_std - variance_epsilon**0.5).clamp_min(0)
        return torch.stack(
            [mean, torch.log1p(stable_std), deviation, delta],
            dim=-1,
        )

    @staticmethod
    def _replace_anchor_channels(
        latent: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([anchors.to(latent), latent[..., VALUE_ANCHOR_DIM:]], dim=-1)

    def encode(self, batch: STBatch) -> torch.Tensor:
        latent = super().encode(batch)
        return self._replace_anchor_channels(latent, self.value_anchors(batch))

    def encode_with_observation_mask(
        self, batch: STBatch, observation_mask: torch.Tensor
    ) -> torch.Tensor:
        latent = super().encode_with_observation_mask(batch, observation_mask)
        anchors = self.value_anchors(
            batch,
            values=batch.values * observation_mask[..., None].to(batch.values.dtype),
            observation_mask=observation_mask,
        )
        return self._replace_anchor_channels(latent, anchors)

    def stage1_latent_consistency_loss(
        self, partial_latent: torch.Tensor, full_latent: torch.Tensor
    ) -> torch.Tensor:
        # Sparse and full value moments are expected to differ.  Penalizing
        # those deterministic anchors would train the geometry assignment to
        # hide missing-value effects.  Consistency applies only to the learned
        # transferable residual representation.
        return F.smooth_l1_loss(
            partial_latent[..., VALUE_ANCHOR_DIM:],
            full_latent[..., VALUE_ANCHOR_DIM:].detach(),
        )

    def _anchor_base(
        self, latent_tokens: torch.Tensor, spatial_relative: torch.Tensor
    ) -> torch.Tensor:
        start = self.num_global_spatial_latents
        # Use squared local distance. The precomputed Euclidean distance uses
        # sqrt(sum(delta^2)), whose derivative is singular when a query lands
        # exactly on a one-node/degenerate slot support. This branch has a
        # non-zero reconstruction gradient from the first update, unlike the
        # parent's initially-zero spatial-bias MLP.
        local_delta = spatial_relative[..., start:, 3:6]
        distance_sq = local_delta.square().sum(dim=-1)
        temperature = F.softplus(self.anchor_temperature_unconstrained) + 1e-3
        weights = torch.softmax(-distance_sq / temperature.square(), dim=-1)
        local_mean = latent_tokens[:, :, start:, 0]
        return torch.einsum("btnk,btk->btn", weights, local_mean)[..., None]

    def decode_with_batch(
        self,
        latent_tokens: torch.Tensor,
        query_coords: torch.Tensor,
        output_batch: STBatch,
        condition_batch: Optional[STBatch] = None,
    ) -> torch.Tensor:
        self.activate_batch_geometry(output_batch)
        support_batch = output_batch if condition_batch is None else condition_batch
        spatial_relative = self._spatial_relative_features(support_batch, query_coords)
        residual = self.decoder(
            self.encode_query_coords(query_coords),
            latent_tokens,
            self._slot_context(support_batch),
            spatial_relative,
            self._temporal_relative_features(output_batch),
        )
        return self._anchor_base(latent_tokens, spatial_relative) + residual

class AnchorAffineDecoderMixin:
    """Decode a dimensionless residual and restore value moments analytically.

    The value-anchor channels intentionally preserve domain-specific value
    statistics, so forcing them onto the pretraining latent distribution would
    discard useful zero-shot information.  Instead, this mixin removes their
    absolute location/scale before the neural decoder and applies those moments
    back to its output:

        prediction = interpolated_mean + interpolated_scale * residual

    ``std`` and mean absolute deviation are blended into a robust slot scale.
    A small floor relative to the frame-global scale keeps singleton/constant
    local slots trainable without introducing a domain-sized learned bias.
    """

    anchor_affine_decoder = True

    def __init__(
        self,
        *args,
        anchor_scale_floor_ratio: float = 0.05,
        anchor_mad_to_std: float = 1.2533141373155001,
        **kwargs,
    ) -> None:
        if anchor_scale_floor_ratio < 0:
            raise ValueError("anchor_scale_floor_ratio must be non-negative")
        if anchor_mad_to_std <= 0:
            raise ValueError("anchor_mad_to_std must be positive")
        super().__init__(*args, **kwargs)
        self.anchor_scale_floor_ratio = float(anchor_scale_floor_ratio)
        self.anchor_mad_to_std = float(anchor_mad_to_std)

    def _anchor_affine_statistics(
        self, latent_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent_tokens.shape[-1] <= VALUE_ANCHOR_DIM:
            raise ValueError("Anchor-affine decoding requires residual latent channels")
        anchors = latent_tokens[..., :VALUE_ANCHOR_DIM]
        mean = anchors[..., 0]
        std = torch.expm1(anchors[..., 1]).clamp_min(0)
        mad_as_std = anchors[..., 2].clamp_min(0) * self.anchor_mad_to_std
        slot_scale = 0.5 * (std + mad_as_std)

        global_stop = self.num_global_spatial_latents
        if global_stop < 1:
            raise ValueError("Anchor-affine decoding requires a global spatial slot")
        frame_mean = mean[..., :global_stop].mean(dim=-1, keepdim=True)
        frame_scale = slot_scale[..., :global_stop].mean(dim=-1, keepdim=True)
        # Preserve exact scale equivariance away from the degenerate all-zero
        # case; epsilon is numerical protection, not a data-scale floor.
        stable_frame_scale = frame_scale.clamp_min(1e-6)

        canonical_anchors = torch.stack(
            [
                (mean - frame_mean) / stable_frame_scale,
                std / stable_frame_scale,
                anchors[..., 2] / stable_frame_scale,
                anchors[..., 3] / stable_frame_scale,
            ],
            dim=-1,
        )
        canonical_latent = torch.cat(
            [canonical_anchors, latent_tokens[..., VALUE_ANCHOR_DIM:]], dim=-1
        )
        return mean, slot_scale, canonical_latent

    def _anchor_affine_base_and_scale(
        self,
        mean: torch.Tensor,
        slot_scale: torch.Tensor,
        spatial_relative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.num_global_spatial_latents
        local_delta = spatial_relative[..., start:, 3:6]
        distance_sq = local_delta.square().sum(dim=-1)
        temperature = F.softplus(self.anchor_temperature_unconstrained) + 1e-3
        weights = torch.softmax(-distance_sq / temperature.square(), dim=-1)

        local_mean = mean[..., start:]
        base = torch.einsum("btnk,btk->btn", weights, local_mean)

        global_scale = slot_scale[..., :start].mean(dim=-1, keepdim=True)
        relative_floor = self.anchor_scale_floor_ratio * global_scale
        local_scale = slot_scale[..., start:]
        effective_scale = torch.sqrt(
            local_scale.square() + relative_floor.square() + 1e-12
        )
        query_scale = torch.einsum("btnk,btk->btn", weights, effective_scale)
        return base[..., None], query_scale[..., None]

    def decode_with_batch(
        self,
        latent_tokens: torch.Tensor,
        query_coords: torch.Tensor,
        output_batch: STBatch,
        condition_batch: Optional[STBatch] = None,
    ) -> torch.Tensor:
        self.activate_batch_geometry(output_batch)
        support_batch = output_batch if condition_batch is None else condition_batch
        spatial_relative = self._spatial_relative_features(support_batch, query_coords)
        mean, slot_scale, canonical_latent = self._anchor_affine_statistics(
            latent_tokens
        )
        base, scale = self._anchor_affine_base_and_scale(
            mean, slot_scale, spatial_relative
        )
        residual = self.decoder(
            self.encode_query_coords(query_coords),
            canonical_latent,
            self._slot_context(support_batch),
            spatial_relative,
            self._temporal_relative_features(output_batch),
        )
        return base + scale * residual

    @torch.no_grad()
    def geometry_diagnostics(self, batch: STBatch) -> dict[str, object]:
        diagnostics = super().geometry_diagnostics(batch)
        diagnostics.update(
            {
                "anchor_affine_decoder": True,
                "anchor_affine_base": "geometry_interpolated_local_mean",
                "anchor_affine_scale": "mean_of_std_and_mad_as_gaussian_std",
                "anchor_scale_floor_ratio": self.anchor_scale_floor_ratio,
                "anchor_mad_to_std": self.anchor_mad_to_std,
                "decoder_residual_units": "frame_anchor_standardized",
            }
        )
        return diagnostics

class AnchorAffineValueAnchoredFoundationField(
    AnchorAffineDecoderMixin, ValueAnchoredFoundationFieldV2
):
    """Anchor-affine field with a weak variational residual bottleneck.

    The first four latent channels are analytic value anchors and therefore
    stay deterministic.  The learned residual channels form a diagonal
    Gaussian posterior.  Stage 1 decodes a reparameterized sample during
    training and the posterior mean during validation, Stage 2 and inference.
    """

    value_canonical_version = "v2_anchor_affine_decoder"

    def __init__(
        self,
        *args,
        vae_free_bits: float = 0.01,
        posterior_logvar_min: float = -6.0,
        posterior_logvar_max: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if vae_free_bits < 0:
            raise ValueError("vae_free_bits must be non-negative")
        if posterior_logvar_min >= posterior_logvar_max:
            raise ValueError("posterior log-variance bounds are invalid")

        residual_dim = self.config.latent_dim - VALUE_ANCHOR_DIM
        self.posterior_mean = nn.Linear(residual_dim, residual_dim)
        self.posterior_logvar = nn.Linear(residual_dim, residual_dim)
        nn.init.eye_(self.posterior_mean.weight)
        nn.init.zeros_(self.posterior_mean.bias)
        nn.init.zeros_(self.posterior_logvar.weight)
        nn.init.constant_(self.posterior_logvar.bias, -5.0)

        self.vae_free_bits = float(vae_free_bits)
        self.posterior_logvar_min = float(posterior_logvar_min)
        self.posterior_logvar_max = float(posterior_logvar_max)
        self.vae_kl_weight = 0.0
        self._last_posterior_mean: Optional[torch.Tensor] = None
        self._last_posterior_logvar: Optional[torch.Tensor] = None

    def set_vae_kl_weight(self, weight: float) -> None:
        if weight < 0:
            raise ValueError("VAE KL weight must be non-negative")
        self.vae_kl_weight = float(weight)

    def _posterior_parameters_from_latent(
        self, base_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchors = base_latent[..., :VALUE_ANCHOR_DIM]
        residual = base_latent[..., VALUE_ANCHOR_DIM:]
        residual_mean = self.posterior_mean(residual)
        residual_logvar = self.posterior_logvar(residual).clamp(
            self.posterior_logvar_min, self.posterior_logvar_max
        )

        # Expose tensors with the same shape as the latent grid while marking
        # the analytic anchor channels as effectively deterministic.  KL is
        # computed only over the learned residual channels below.
        anchor_logvar = torch.full_like(anchors, self.posterior_logvar_min)
        return (
            torch.cat([anchors, residual_mean], dim=-1),
            torch.cat([anchor_logvar, residual_logvar], dim=-1),
        )

    def posterior_parameters_from_latent(
        self, base_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the frozen Stage-1 VAE posterior heads to an aligned grid.

        Unlike :meth:`posterior_parameters`, this entry point does not run the
        full-value encoder and does not overwrite the cached full posterior.
        Stage 2 uses it to map its Stage-1-aligned condition grid into the same
        posterior mean/log-variance coordinate system as the target latent.
        """

        return self._posterior_parameters_from_latent(base_latent)

    def posterior_parameters(
        self, batch: STBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self._posterior_parameters_from_latent(super().encode(batch))
        self._last_posterior_mean = mean
        self._last_posterior_logvar = logvar
        return mean, logvar

    def encode_posterior(
        self, batch: STBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full-grid posterior parameters and cache the full view."""

        return self.posterior_parameters(batch)

    def encode(self, batch: STBatch) -> torch.Tensor:
        mean, logvar = self.posterior_parameters(batch)
        if not self.training:
            return mean
        residual_mean = mean[..., VALUE_ANCHOR_DIM:]
        residual_logvar = logvar[..., VALUE_ANCHOR_DIM:]
        residual_sample = residual_mean + torch.exp(0.5 * residual_logvar) * (
            torch.randn_like(residual_mean)
        )
        return torch.cat([mean[..., :VALUE_ANCHOR_DIM], residual_sample], dim=-1)

    def encode_with_observation_mask(
        self, batch: STBatch, observation_mask: torch.Tensor
    ) -> torch.Tensor:
        # Partial context is a deterministic posterior center.  In particular,
        # do not overwrite the cached full posterior used by Stage-1 KL and
        # full/partial latent consistency.
        base_latent = super().encode_with_observation_mask(batch, observation_mask)
        mean, _ = self._posterior_parameters_from_latent(base_latent)
        return mean

    def cached_full_posterior(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._last_posterior_mean is None or self._last_posterior_logvar is None:
            raise RuntimeError("Encode a full batch before reading its posterior")
        return self._last_posterior_mean, self._last_posterior_logvar

    def stage1_auxiliary_loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean, logvar = self.cached_full_posterior()
        residual_mean = mean[..., VALUE_ANCHOR_DIM:]
        residual_logvar = logvar[..., VALUE_ANCHOR_DIM:]
        kl_per_dimension = 0.5 * (
            residual_mean.square()
            + residual_logvar.exp()
            - 1.0
            - residual_logvar
        )
        if self.vae_free_bits > 0:
            kl_per_dimension = kl_per_dimension.clamp_min(self.vae_free_bits)
        raw_kl = kl_per_dimension.mean()
        weighted_kl = raw_kl * self.vae_kl_weight
        return weighted_kl, {
            "vae_kl_raw": raw_kl,
            "vae_kl_weighted": weighted_kl,
            "vae_kl_beta": raw_kl.new_tensor(self.vae_kl_weight),
            "vae_posterior_std": torch.exp(0.5 * residual_logvar).mean(),
            "vae_posterior_mean_abs": residual_mean.abs().mean(),
        }

    @torch.no_grad()
    def geometry_diagnostics(self, batch: STBatch) -> dict[str, object]:
        diagnostics = super().geometry_diagnostics(batch)
        diagnostics.update(
            {
                "stage1_latent_model": "weak_variational_autoencoder",
                "vae_stochastic_channels": self.config.latent_dim - VALUE_ANCHOR_DIM,
                "vae_deterministic_anchor_channels": VALUE_ANCHOR_DIM,
                "vae_free_bits": self.vae_free_bits,
                "posterior_logvar_bounds": [
                    self.posterior_logvar_min,
                    self.posterior_logvar_max,
                ],
            }
        )
        return diagnostics

__all__ = [
    "AnchorAffineValueAnchoredFoundationField",
    "FOUNDATION_DOMAIN_FEATURE_NAMES",
    "FoundationDomainLoaderView",
    "FoundationDomainSpec",
    "NeuralFieldOutput",
    "build_full_query_coords",
    "discover_foundation_domain_spec",
    "structured_stage1_loss",
]
