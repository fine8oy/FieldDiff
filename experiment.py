from __future__ import annotations

import argparse
import json
import math
import sys

import torch

from . import stage1_grounded as grounded
from . import training as base
from .data.loader import make_loader
from .model.common import move_batch_to_device, valid_mask_for_batch
from .model.diffusion import (
    ValueAnchoredDirectDiffusion,
    decode_neural_field,
    merge_test_condition_preserving_domain,
)
from .model.neural_field import (
    AnchorAffineValueAnchoredFoundationField,
    FOUNDATION_DOMAIN_FEATURE_NAMES,
    FoundationDomainLoaderView,
    build_full_query_coords,
)

EXPERIMENT_NAME = "stfm_foundation"
DEFAULT_CONFIG = base.ROOT / "config.json"
_BASE_PARSE_ARGS = grounded.parse_args
_BASE_STAGE1_TRAIN_EPOCH = grounded.stage1_train_epoch
_ACTIVE_EXPERIMENT_NAME = EXPERIMENT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--stage1-zero-shot-max-batches", type=int, default=32)
    parser.add_argument("--skip-stage1-zero-shot-eval", action="store_true")
    parser.add_argument("--anchor-scale-floor-ratio", type=float, default=0.05)
    parser.add_argument("--anchor-mad-to-std", type=float, default=1.2533141373155001)
    parser.add_argument("--vae-kl-beta", type=float, default=1e-4)
    parser.add_argument("--vae-kl-warmup-epochs", type=int, default=50)
    parser.add_argument("--vae-free-bits", type=float, default=0.01)
    parser.add_argument("--posterior-logvar-min", type=float, default=-6.0)
    parser.add_argument("--posterior-logvar-max", type=float, default=2.0)
    selected, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = _BASE_PARSE_ARGS()
    finally:
        sys.argv = original
    for key, value in vars(selected).items():
        setattr(args, key, value)
    args.requested_ablation_variant = None
    args.model_scale_contract = None
    if args.sigma_data <= 0:
        raise ValueError("--sigma-data must be positive")
    if args.stage1_zero_shot_max_batches < 1:
        raise ValueError("--stage1-zero-shot-max-batches must be positive")
    if args.anchor_scale_floor_ratio < 0 or args.anchor_mad_to_std <= 0:
        raise ValueError("Invalid anchor-affine scale settings")
    if args.vae_kl_beta < 0 or args.vae_kl_warmup_epochs < 1:
        raise ValueError("Invalid VAE KL schedule")
    if args.vae_free_bits < 0:
        raise ValueError("--vae-free-bits must be non-negative")
    if args.posterior_logvar_min >= args.posterior_logvar_max:
        raise ValueError("Invalid VAE posterior log-variance bounds")
    if args.smoke_test:
        args.steps_per_epoch = max(args.steps_per_epoch, 32)
        args.stage1_zero_shot_max_batches = 1
        args.num_global_spatial_latents = 1
        args.log_every = 1
    return args


def build_field(args: argparse.Namespace, device: torch.device):
    if not 0 < args.num_global_spatial_latents < args.num_spatial_latents:
        raise ValueError(
            "--num-global-spatial-latents must be smaller than --num-spatial-latents"
        )
    config = base.make_model_config(args)
    config.sigma_data = float(args.sigma_data)
    field = AnchorAffineValueAnchoredFoundationField(
        config,
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
        anchor_scale_floor_ratio=args.anchor_scale_floor_ratio,
        anchor_mad_to_std=args.anchor_mad_to_std,
        vae_free_bits=args.vae_free_bits,
        posterior_logvar_min=args.posterior_logvar_min,
        posterior_logvar_max=args.posterior_logvar_max,
    ).to(device)
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


def stage1_train_epoch(
    model, loaders, optimizer, args, device, epoch, logger, deadline=None
):
    beta = args.vae_kl_beta * min(
        float(epoch) / float(args.vae_kl_warmup_epochs), 1.0
    )
    model.set_vae_kl_weight(beta)
    logger.info(
        "VAE_KL_SCHEDULE | epoch=%d beta=%.8g target=%.8g warmup_epochs=%d",
        epoch,
        beta,
        args.vae_kl_beta,
        args.vae_kl_warmup_epochs,
    )
    return _BASE_STAGE1_TRAIN_EPOCH(
        model, loaders, optimizer, args, device, epoch, logger, deadline
    )


def build_diffusion(args, field, device):
    return ValueAnchoredDirectDiffusion(
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

def _empty_metrics() -> dict[str, float]:
    return {
        "normalized_absolute_error": 0.0,
        "original_absolute_error": 0.0,
        "original_squared_error": 0.0,
        "count": 0.0,
    }

def _update_metrics(totals, prediction, target, mask, batch) -> None:
    expanded = mask.to(prediction.dtype)
    if expanded.ndim == prediction.ndim - 1:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(prediction)
    totals["normalized_absolute_error"] += float(
        ((prediction - target).abs() * expanded).sum()
    )

    offset = batch.metadata["value_offset"].to(prediction.device, prediction.dtype)
    scale = batch.metadata["value_scale"].to(prediction.device, prediction.dtype)
    while offset.ndim < prediction.ndim:
        offset = offset.unsqueeze(-1)
        scale = scale.unsqueeze(-1)
    prediction_original = prediction * scale + offset
    target_original = target * scale + offset
    error = (prediction_original - target_original) * expanded
    totals["original_absolute_error"] += float(error.abs().sum())
    totals["original_squared_error"] += float(error.square().sum())
    totals["count"] += float(expanded.sum())

def _finish_metrics(totals) -> dict[str, float | int]:
    count = max(totals["count"], 1.0)
    mse = totals["original_squared_error"] / count
    return {
        "normalized_mae": totals["normalized_absolute_error"] / count,
        "mae": totals["original_absolute_error"] / count,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "count": int(totals["count"]),
    }

def evaluate_stage1_zero_shot(
    field,
    store,
    zero_shot_names,
    args,
    device,
    run_dir,
    logger,
    *,
    best_stage1_epoch=None,
    best_stage1_validation=None,
):
    if args.skip_stage1_zero_shot_eval:
        return {"enabled": False, "reason": "--skip-stage1-zero-shot-eval"}
    if not zero_shot_names:
        raise ValueError("Stage-1 zero-shot evaluation received no target domains")
    field.eval()
    rows = {}
    adapter_diagnostics = {}
    macro_values = {
        "oracle_full": {"normalized_mae": [], "mae": [], "rmse": []},
        "partial_context": {"normalized_mae": [], "mae": [], "rmse": []},
    }
    for domain_index, name in enumerate(zero_shot_names):
        domain = store.get(name)
        maximum_nodes = base.effective_max_nodes(domain, args)
        rows[name] = {}
        for task_index, task in enumerate(base.TASKS):
            dataset = base.evenly_spaced_subset(
                domain.task_dataset(task, maximum_nodes),
                args.stage1_zero_shot_max_batches,
            )
            loader = FoundationDomainLoaderView(
                make_loader(
                    dataset,
                    args.eval_batch_size,
                    False,
                    args.num_workers,
                    args.seed + 410003 + 1009 * domain_index + task_index,
                ),
                store.spec(name),
            )
            totals = {
                "oracle_full": _empty_metrics(),
                "partial_context": _empty_metrics(),
            }
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                query = build_full_query_coords(batch)
                evaluation_mask = batch.metadata.get(
                    "eval_mask", valid_mask_for_batch(batch)
                )

                full_latent = field.encode(batch)
                full_prediction = decode_neural_field(
                    field, full_latent, query, batch, batch
                )
                _update_metrics(
                    totals["oracle_full"],
                    full_prediction,
                    batch.values,
                    evaluation_mask,
                    batch,
                )

                condition = merge_test_condition_preserving_domain(batch)
                if condition.obs_mask is None:
                    raise RuntimeError(f"Task {task} did not provide an observation mask")
                partial_latent = field.encode_with_observation_mask(
                    condition, condition.obs_mask
                )
                partial_prediction = decode_neural_field(
                    field,
                    partial_latent,
                    query,
                    batch,
                    condition,
                )
                _update_metrics(
                    totals["partial_context"],
                    partial_prediction,
                    batch.values,
                    evaluation_mask,
                    batch,
                )

            rows[name][task] = {
                method: _finish_metrics(values)
                for method, values in totals.items()
            }
            for method, metrics in rows[name][task].items():
                for metric in macro_values[method]:
                    macro_values[method][metric].append(float(metrics[metric]))
            logger.info(
                "STAGE1_ZERO_SHOT | domain=%s task=%s oracle_nmae=%.8f "
                "partial_nmae=%.8f",
                name,
                task,
                rows[name][task]["oracle_full"]["normalized_mae"],
                rows[name][task]["partial_context"]["normalized_mae"],
            )
            diagnostics = getattr(field, "channel_adapter_diagnostics", None)
            if diagnostics is None:
                diagnostics = getattr(field, "graph_diagnostics", None)
            if diagnostics is None:
                diagnostics = getattr(field, "covariance_diagnostics", None)
            if diagnostics is not None:
                adapter_diagnostics.setdefault(name, {})[task] = diagnostics()
    macro = {
        method: {
            metric: sum(values) / max(len(values), 1)
            for metric, values in metrics.items()
        }
        for method, metrics in macro_values.items()
    }
    result = {
        "enabled": True,
        "experiment": _ACTIVE_EXPERIMENT_NAME,
        "protocol": "held_out_time_zero_shot_before_stage2",
        "best_stage1_epoch": best_stage1_epoch,
        "best_stage1_validation": best_stage1_validation,
        "batches_per_domain_task": args.stage1_zero_shot_max_batches,
        "methods": {
            "oracle_full": "encode full target values then decode",
            "partial_context": "encode only task-observed values then decode",
        },
        "macro": macro,
        "domains": rows,
    }
    if adapter_diagnostics:
        result["channel_adapter_diagnostics"] = adapter_diagnostics
    output = run_dir / "stage1_zero_shot_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("STAGE1_ZERO_SHOT_COMPLETE | %s", json.dumps(macro))
    return {"enabled": True, "path": str(output), "macro": macro}


def checkpoint_payload(
    field, diffusion, args, protocol, pretrain_names, specs, epoch, score
):
    payload = grounded.checkpoint_payload(
        field, diffusion, args, protocol, pretrain_names, specs, epoch, score
    )
    payload["experiment"] = EXPERIMENT_NAME
    payload["model_hparams"].update(
        {
            "sigma_data": args.sigma_data,
            "stage1_zero_shot_max_batches": args.stage1_zero_shot_max_batches,
            "anchor_scale_floor_ratio": args.anchor_scale_floor_ratio,
            "anchor_mad_to_std": args.anchor_mad_to_std,
            "vae_kl_beta": args.vae_kl_beta,
            "vae_kl_warmup_epochs": args.vae_kl_warmup_epochs,
            "vae_free_bits": args.vae_free_bits,
            "posterior_logvar_min": args.posterior_logvar_min,
            "posterior_logvar_max": args.posterior_logvar_max,
        }
    )
    payload["zero_shot_contract"].update(
        {
            "latent_content": "value_payload_plus_explicit_value_moments",
            "stage1_latent_model": "weak_variational_autoencoder",
            "stage1_prior": "unit_gaussian_on_residual_channels_with_free_bits_and_kl_warmup",
            "deterministic_latent_channels": "first_four_value_anchors",
            "stochastic_latent_channels": "all_residual_channels_after_first_four",
            "evaluation_latent": "posterior_mean",
            "encoder_point_payload": "normalized_value_only",
            "encoder_coordinate_role": "geometry_only_support_assignment",
            "encoder_geometry_added_to_latent": False,
            "decoder_domain_condition": None,
            "domain_value_descriptor_condition": False,
            "decoder_form": "analytic_local_mean_plus_scale_times_dimensionless_residual",
            "decoder_anchor_canonicalization": "frame_global_location_scale",
            "condition_alignment": "frozen_stage1_vae_posterior_heads",
            "conditional_posterior": "stage1_posterior_mean_and_logvar_on_residual_channels",
            "diffusion_target": "condition_standardized_latent_residual",
            "conditional_posterior_supervision": "full_to_condition_gaussian_kl",
            "deterministic_anchor_diffusion_scale": 1.0,
            "evaluation_initialization": "zero_standardized_residual_plus_anchor_sigma_noise",
            "stage1_zero_shot_evaluation": "before_stage2_oracle_and_partial_context",
            "target_domain_parameter_updates": 0,
        }
    )
    return payload


def main() -> None:
    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.DEFAULT_CONFIG = DEFAULT_CONFIG
    base.DEFAULT_LOG_DIR = base.ROOT / "logs" / EXPERIMENT_NAME
    base.parse_args = parse_args
    base.build_field = build_field
    base.build_diffusion = build_diffusion
    base.task_config_from_args = grounded.task_config_from_args
    base.stage1_train_epoch = stage1_train_epoch
    base.run_stage1_validation = grounded.run_stage1_validation
    base.stage1_final_evaluation = evaluate_stage1_zero_shot
    base.checkpoint_payload = checkpoint_payload
    base.main()


__all__ = [
    "build_diffusion",
    "build_field",
    "evaluate_stage1_zero_shot",
    "main",
    "parse_args",
    "stage1_train_epoch",
]
