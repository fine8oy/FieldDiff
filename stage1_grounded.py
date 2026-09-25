from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import training as base
from .model.common import (
    BalancedDomainIterator,
    STBatch,
    move_batch_to_device,
    valid_mask_for_batch,
)
from .model.diffusion import (
    CurrentConditionalTaskConfig,
    RawLatentFoundationDirectDiffusion,
)
from .model.neural_field import (
    CoordinateGroundedFoundationField,
    FOUNDATION_DOMAIN_FEATURE_NAMES,
    build_full_query_coords,
    structured_stage1_loss,
)


EXPERIMENT_NAME = "stfm_foundation"
DEFAULT_CONFIG = base.ROOT / "configs" / f"{EXPERIMENT_NAME}.json"
DEFAULT_LOG_DIR = base.ROOT / "logs" / EXPERIMENT_NAME
_BASE_PARSE_ARGS = base.parse_args
_BASE_CHECKPOINT_PAYLOAD = base.checkpoint_payload


def parse_args() -> argparse.Namespace:
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument("--num-global-spatial-latents", type=int, default=16)
    extra.add_argument("--assignment-temperature", type=float, default=0.25)
    extra.add_argument("--sinkhorn-iterations", type=int, default=4)
    extra.add_argument("--stage1-context-ratio-min", type=float, default=0.30)
    extra.add_argument("--stage1-context-ratio-max", type=float, default=0.70)
    extra.add_argument("--stage1-full-loss-weight", type=float, default=0.50)
    extra.add_argument("--stage1-query-loss-weight", type=float, default=1.00)
    extra.add_argument("--stage1-latent-consistency-weight", type=float, default=0.10)
    extra.add_argument("--stage1-geometry-weight", type=float, default=0.05)
    extra.add_argument("--stage1-geometry-margin", type=float, default=0.05)
    extra.add_argument("--stage1-geometry-every", type=int, default=5)
    grounded, remaining = extra.parse_known_args()
    explicit_total_slots = any(
        item == "--num-spatial-latents" or item.startswith("--num-spatial-latents=")
        for item in remaining
    )
    explicit_node_ratio = any(
        item == "--node-train-ratio" or item.startswith("--node-train-ratio=")
        for item in remaining
    )
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = _BASE_PARSE_ARGS()
    finally:
        sys.argv = original_argv
    for key, value in vars(grounded).items():
        setattr(args, key, value)

    # This experiment defines spatial extrapolation by an in-window node mask;
    # no permanent target nodes are hidden from Stage 1.
    if explicit_node_ratio and args.node_train_ratio != 1.0:
        raise ValueError("This temporal-split experiment fixes --node-train-ratio=1.0")
    args.node_train_ratio = 1.0
    if not explicit_total_slots and not args.smoke_test:
        args.num_spatial_latents = 48
    if args.smoke_test:
        args.num_global_spatial_latents = 1
        args.stage1_geometry_every = 1
    if not 0 < args.assignment_temperature:
        raise ValueError("--assignment-temperature must be positive")
    if args.sinkhorn_iterations < 1 or args.stage1_geometry_every < 1:
        raise ValueError("Sinkhorn/geometry iteration counts must be positive")
    if not 0 < args.stage1_context_ratio_min <= args.stage1_context_ratio_max < 1:
        raise ValueError("Stage-1 context ratios must satisfy 0 < min <= max < 1")
    if min(
        args.stage1_full_loss_weight,
        args.stage1_query_loss_weight,
        args.stage1_latent_consistency_weight,
        args.stage1_geometry_weight,
        args.stage1_geometry_margin,
    ) < 0:
        raise ValueError("Stage-1 grounded loss weights must be non-negative")
    return args


def task_config_from_args(args: argparse.Namespace) -> CurrentConditionalTaskConfig:
    return CurrentConditionalTaskConfig(
        forecasting_context_steps=args.forecasting_context_steps,
        interpolation_observed_steps=args.interpolation_observed_steps,
        spatial_observed_node_ratio=1.0 - args.spatial_extrapolation_mask_ratio,
        spatiotemporal_observed_node_ratio=1.0
        - args.spatiotemporal_imputation_mask_ratio,
    )


def build_field(
    args: argparse.Namespace, device: torch.device
) -> CoordinateGroundedFoundationField:
    if not 0 < args.num_global_spatial_latents < args.num_spatial_latents:
        raise ValueError(
            "--num-global-spatial-latents must be smaller than --num-spatial-latents"
        )
    return CoordinateGroundedFoundationField(
        base.make_model_config(args),
        domain_feature_dim=len(FOUNDATION_DOMAIN_FEATURE_NAMES),
        global_fourier_scale=args.global_spatial_fourier_scale,
        local_fourier_scale=args.local_spatial_fourier_scale,
        num_spatial_latents=args.num_spatial_latents,
        num_global_spatial_latents=args.num_global_spatial_latents,
        assignment_temperature=args.assignment_temperature,
        sinkhorn_iterations=args.sinkhorn_iterations,
        spatial_tokenizer_layers=args.spatial_tokenizer_layers,
        axial_layers=args.stage1_axial_layers,
        heads=args.stage1_heads,
        decoder_temporal_radius=args.decoder_temporal_radius,
        temporal_attention_radius=args.stage1_temporal_attention_radius,
    ).to(device)


def build_diffusion(args, field, device):
    return RawLatentFoundationDirectDiffusion(
        field.config,
        field,
        domain_feature_dim=len(FOUNDATION_DOMAIN_FEATURE_NAMES),
        condition_temporal_layers=args.condition_temporal_layers,
        condition_axial_layers=args.condition_axial_layers,
        confidence_bandwidth=args.confidence_bandwidth,
        missing_loss_weight=args.missing_latent_loss_weight,
        condition_dropout=args.condition_dropout,
        guidance_scale=args.guidance_scale,
    ).to(device)


def _batch_with_observation_mask(batch: STBatch, mask: torch.Tensor) -> STBatch:
    return STBatch(
        values=batch.values,
        spatial_coords=batch.spatial_coords,
        time_coords=batch.time_coords,
        node_mask=batch.node_mask,
        obs_mask=mask.to(batch.values.dtype),
        metadata=dict(batch.metadata),
    )


def _sample_context_mask(
    batch: STBatch,
    valid: torch.Tensor,
    minimum: float,
    maximum: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    b, t, n = valid.shape
    ratio = torch.empty(b, device=valid.device, dtype=batch.values.dtype)
    ratio.uniform_(minimum, maximum, generator=generator)
    node_valid = valid.any(dim=1)
    draw = torch.rand((b, n), device=valid.device, generator=generator)
    selected = (draw < ratio[:, None]) & node_valid
    for row in range(b):
        ids = torch.where(node_valid[row])[0]
        if ids.numel() == 0:
            continue
        if not selected[row].any():
            selected[row, ids[0]] = True
        if selected[row, ids].all() and ids.numel() > 1:
            selected[row, ids[-1]] = False
    return valid & selected[:, None, :]


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum().clamp_min(1.0) * target.shape[-1]
    return ((prediction - target).abs() * mask[..., None]).sum() / denominator


def _shuffled_spatial_query(batch: STBatch, generator: torch.Generator | None = None) -> torch.Tensor:
    query = build_full_query_coords(batch).clone()
    spatial_dim = batch.spatial_coords.shape[-1]
    for row in range(batch.batch_size):
        permutation = torch.randperm(batch.max_nodes, device=query.device, generator=generator)
        query[row, :, :, :spatial_dim] = query[
            row, :, permutation, :spatial_dim
        ].clone()
    return query


def grounded_stage1_objective(
    model: CoordinateGroundedFoundationField,
    batch: STBatch,
    args,
    generator: torch.Generator | None = None,
    compute_geometry: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = valid_mask_for_batch(batch).to(torch.bool)
    full = model(batch)
    full_loss, full_parts = structured_stage1_loss(
        full.reconstruction,
        batch.values,
        valid,
        model.spatial_tokenizer.repulsion_loss(),
        temporal_difference_weight=args.stage1_temporal_difference_weight,
        anchor_repulsion_weight=args.anchor_repulsion_weight,
    )

    context_mask = _sample_context_mask(
        batch,
        valid,
        args.stage1_context_ratio_min,
        args.stage1_context_ratio_max,
        generator,
    )
    query_mask = valid & ~context_mask
    context_batch = _batch_with_observation_mask(batch, context_mask)
    partial_latent = model.encode_with_observation_mask(batch, context_mask)
    query_prediction = model.decode_with_batch(
        partial_latent,
        build_full_query_coords(batch),
        batch,
        condition_batch=context_batch,
    )
    query_loss, query_parts = structured_stage1_loss(
        query_prediction,
        batch.values,
        query_mask,
        full_loss.new_zeros(()),
        temporal_difference_weight=args.stage1_temporal_difference_weight,
        anchor_repulsion_weight=0.0,
    )
    consistency_objective = getattr(model, "stage1_latent_consistency_loss", None)
    if consistency_objective is None:
        consistency = F.smooth_l1_loss(
            partial_latent, full.latent_tokens.detach()
        )
    else:
        consistency = consistency_objective(partial_latent, full.latent_tokens)
    cached_posterior = getattr(model, "cached_full_posterior", None)
    if cached_posterior is not None:
        # In the variational grounded experiment, compare the partial context
        # representation with the stable full posterior center rather than a
        # random reparameterized sample.
        full_center, _ = cached_posterior()
        if consistency_objective is None:
            consistency = F.smooth_l1_loss(
                partial_latent, full_center.detach()
            )
        else:
            consistency = consistency_objective(partial_latent, full_center)

    correct_error = _masked_l1(query_prediction, batch.values, query_mask)
    geometry_loss = full_loss.new_zeros(())
    shuffled_error = full_loss.new_zeros(())
    if compute_geometry and args.stage1_geometry_weight > 0:
        wrong_prediction = model.decode_with_batch(
            full.latent_tokens,
            _shuffled_spatial_query(batch, generator),
            batch,
            condition_batch=batch,
        )
        shuffled_error = _masked_l1(wrong_prediction, batch.values, valid)
        full_error = _masked_l1(full.reconstruction, batch.values, valid)
        geometry_loss = F.relu(args.stage1_geometry_margin + full_error - shuffled_error)

    loss = (
        args.stage1_full_loss_weight * full_loss
        + args.stage1_query_loss_weight * query_loss
        + args.stage1_latent_consistency_weight * consistency
        + args.stage1_geometry_weight * geometry_loss
    )
    auxiliary = getattr(model, "stage1_auxiliary_loss", None)
    auxiliary_parts: dict[str, torch.Tensor] = {}
    if auxiliary is not None:
        auxiliary_loss, auxiliary_parts = auxiliary()
        loss = loss + auxiliary_loss
    parts = {
        "loss": loss,
        "full_loss": full_loss,
        "query_loss": query_loss,
        "latent_consistency": consistency,
        "geometry_loss": geometry_loss,
        "correct_query_l1": correct_error,
        "shuffled_coordinate_l1": shuffled_error,
        "context_fraction": context_mask.float().sum() / valid.float().sum().clamp_min(1.0),
        **{f"full_{key}": value for key, value in full_parts.items()},
        **{f"query_{key}": value for key, value in query_parts.items()},
        **auxiliary_parts,
    }
    return loss, parts


def stage1_train_epoch(model, loaders, optimizer, args, device, epoch, logger, deadline=None):
    model.train()
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    full_steps = sum(len(loader) for loader in loaders.values())
    steps = args.steps_per_epoch or full_steps
    completed = 0
    epoch_start = time.monotonic()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 31001 * epoch)
    iterator = BalancedDomainIterator(loaders, steps, offset=(epoch - 1) % len(loaders))
    for step, (domain, batch) in enumerate(iterator):
        if deadline is not None and time.monotonic() >= deadline:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = grounded_stage1_objective(
            model,
            batch,
            args,
            generator,
            compute_geometry=step % args.stage1_geometry_every == 0,
        )
        if not bool(torch.isfinite(loss).all()):
            nonfinite_parts = sorted(
                key
                for key, value in parts.items()
                if not bool(torch.isfinite(value.detach()).all())
            )
            nonfinite_parameters = [
                name
                for name, parameter in model.named_parameters()
                if not bool(torch.isfinite(parameter.detach()).all())
            ]
            raise FloatingPointError(
                "Non-finite Stage-1 objective before backward: "
                f"epoch={epoch} step={step + 1} domain={domain} "
                f"parts={nonfinite_parts} "
                f"parameters={nonfinite_parameters[:20]}"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        if not bool(torch.isfinite(gradient_norm).all()):
            nonfinite_gradients = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad.detach()).all())
            ]
            raise FloatingPointError(
                "Non-finite Stage-1 gradient before optimizer step: "
                f"epoch={epoch} step={step + 1} domain={domain} "
                f"loss={float(loss.detach())} "
                f"gradients={nonfinite_gradients[:20]}"
            )
        optimizer.step()
        for key, value in parts.items():
            totals[domain][key].append(float(value.detach()))
        completed += 1
        if completed == 1 or (args.log_every > 0 and completed % args.log_every == 0):
            elapsed = max(time.monotonic() - epoch_start, 1e-9)
            rate = completed / elapsed
            logger.info(
                "GROUNDED_STAGE1_STEP | epoch=%d step=%d/%d domain=%s loss=%.8f "
                "query=%.8f consistency=%.8f steps_per_second=%.3f",
                epoch,
                completed,
                steps,
                domain,
                float(loss.detach()),
                float(parts["query_loss"].detach()),
                float(parts["latent_consistency"].detach()),
                rate,
            )
    per_domain = {
        name: {key: float(np.mean(values)) for key, values in metrics.items()}
        for name, metrics in totals.items()
    }
    losses = [metrics["loss"] for metrics in per_domain.values()]
    return {
        "macro_loss": float(np.mean(losses)) if losses else math.nan,
        "steps": completed,
        "requested_steps": steps,
        "time_limit_reached": completed < steps,
        "per_domain": per_domain,
    }


@torch.no_grad()
def run_stage1_validation(model, loaders, device, args):
    model.eval()
    result = {}
    validation_args = argparse.Namespace(
        stage1_temporal_difference_weight=args.stage1_temporal_difference_weight,
        anchor_repulsion_weight=args.anchor_repulsion_weight,
        stage1_context_ratio_min=getattr(model, "stage1_context_ratio_min", 0.5),
        stage1_context_ratio_max=getattr(model, "stage1_context_ratio_max", 0.5),
        stage1_full_loss_weight=getattr(model, "stage1_full_loss_weight", 0.5),
        stage1_query_loss_weight=getattr(model, "stage1_query_loss_weight", 1.0),
        stage1_latent_consistency_weight=getattr(
            model, "stage1_latent_consistency_weight", 0.1
        ),
        stage1_geometry_weight=getattr(model, "stage1_geometry_weight", 0.05),
        stage1_geometry_margin=getattr(model, "stage1_geometry_margin", 0.05),
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(99173)
    for domain, loader in loaders.items():
        totals: dict[str, list[float]] = defaultdict(list)
        for batch_index, batch in enumerate(loader):
            if args.stage1_val_max_batches > 0 and batch_index >= args.stage1_val_max_batches:
                break
            batch = move_batch_to_device(batch, device)
            _, parts = grounded_stage1_objective(
                model, batch, validation_args, generator, compute_geometry=True
            )
            for key, value in parts.items():
                totals[key].append(float(value))
        result[domain] = {
            key: float(np.mean(values)) for key, values in totals.items()
        }
    return {
        "macro_loss": float(np.mean([row["loss"] for row in result.values()])),
        "per_domain": result,
    }


def checkpoint_payload(field, diffusion, args, protocol, pretrain_names, specs, epoch, score):
    payload = _BASE_CHECKPOINT_PAYLOAD(
        field, diffusion, args, protocol, pretrain_names, specs, epoch, score
    )
    payload["experiment"] = EXPERIMENT_NAME
    payload["model_hparams"].update(
        {
            key: getattr(args, key)
            for key in (
                "num_global_spatial_latents",
                "assignment_temperature",
                "sinkhorn_iterations",
                "stage1_context_ratio_min",
                "stage1_context_ratio_max",
                "stage1_full_loss_weight",
                "stage1_query_loss_weight",
                "stage1_latent_consistency_weight",
                "stage1_geometry_weight",
                "stage1_geometry_margin",
                "stage1_geometry_every",
            )
        }
    )
    payload["zero_shot_contract"].update(
        {
            "spatial_latent_shape": "global_plus_geometry_grounded_local",
            "local_slot_assignment": "geometry_only_balanced_sinkhorn",
            "decoder_spatial_condition": "explicit_query_to_support_ecef_and_local_metric_bias",
            "decoder_temporal_condition": "physical_delta_time_rbf_bias",
            "decoder_domain_condition": "continuous_descriptor_and_observed_context_film",
            "domain_feature_transform": "pretrain_domain_mean_std_clipped_6sigma",
            "stage1_training": "full_reconstruction_plus_context_query_latent_consistency",
            "target_domain_parameter_updates": 0,
        }
    )
    payload["split_contract"] = {
        "time_train_ratio": args.time_train_ratio,
        "node_train_ratio": 1.0,
        "held_out_test_nodes": False,
        "all_tasks_test_on_held_out_time": True,
        "spatial_extrapolation_mask_ratio": args.spatial_extrapolation_mask_ratio,
    }
    return payload


def main() -> None:
    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.DEFAULT_CONFIG = DEFAULT_CONFIG
    base.DEFAULT_LOG_DIR = DEFAULT_LOG_DIR
    base.parse_args = parse_args
    base.build_field = build_field
    base.build_diffusion = build_diffusion
    base.task_config_from_args = task_config_from_args
    base.stage1_train_epoch = stage1_train_epoch
    base.run_stage1_validation = run_stage1_validation
    base.checkpoint_payload = checkpoint_payload

    # Validation receives a narrow compatibility namespace, so retain the
    # grounded Stage-1 hyperparameters on the field itself.
    original_build_field = base.build_field

    def configured_build_field(args, device):
        field = original_build_field(args, device)
        for name in (
            "stage1_context_ratio_min",
            "stage1_context_ratio_max",
            "stage1_full_loss_weight",
            "stage1_query_loss_weight",
            "stage1_latent_consistency_weight",
            "stage1_geometry_weight",
            "stage1_geometry_margin",
        ):
            setattr(field, name, getattr(args, name))
        return field

    base.build_field = configured_build_field
    base.main()


if __name__ == "__main__":
    main()
