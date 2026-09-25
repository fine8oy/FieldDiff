from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Mapping, Optional

import numpy as np
import torch

from .model.common import STBatch, STTask, move_batch_to_device, valid_mask_for_batch
from .model.diffusion import (
    CONDITIONAL_TASKS,
    DirectConditionOutput,
    StructuredDirectLatentDiffusion,
    build_condition_view,
    decode_neural_field,
)
from .model.neural_field import build_full_query_coords, structured_stage1_loss


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def module_gradient_norm(module: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum())
    return squared**0.5


def denormalize_values_for_batch(
    values: torch.Tensor, batch: STBatch
) -> torch.Tensor:
    offset = batch.metadata.get("value_offset", batch.metadata.get("value_mean"))
    scale = batch.metadata.get("value_scale", batch.metadata.get("value_std"))
    if offset is None or scale is None:
        return values
    while offset.ndim < values.ndim:
        offset = offset.unsqueeze(-1)
    while scale.ndim < values.ndim:
        scale = scale.unsqueeze(-1)
    return values * scale.to(values) + offset.to(values)


def compute_original_scale_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: STBatch,
    mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    prediction = denormalize_values_for_batch(prediction, batch)
    target = denormalize_values_for_batch(target, batch)
    maes, mses, rmses = [], [], []
    for index in range(prediction.shape[0]):
        predicted = prediction[index].reshape(-1)
        expected = target[index].reshape(-1)
        if mask is not None:
            sample_mask = mask[index].reshape(-1)
            if target.shape[-1] > 1:
                sample_mask = sample_mask.repeat_interleave(target.shape[-1])
            valid = sample_mask > 0
            if not bool(valid.any()):
                continue
            predicted = predicted[valid]
            expected = expected[valid]
        if not bool(torch.isfinite(predicted).all()) or not bool(
            torch.isfinite(expected).all()
        ):
            raise ValueError("Metric inputs contain NaN or Inf")
        error = predicted - expected
        mae = error.abs().mean()
        mse = error.square().mean()
        maes.append(float(mae))
        mses.append(float(mse))
        rmses.append(float(mse.sqrt()))
    if not maes:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0}
    return {
        "mae": float(np.mean(maes)),
        "mse": float(np.mean(mses)),
        "rmse": float(np.mean(rmses)),
    }


def _mean_dict(values: Mapping[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(items)) for key, items in values.items() if items}


@torch.no_grad()
def run_stage1_validation(model, loaders, device, args) -> dict:
    model.eval()
    result = {}
    for domain, loader in loaders.items():
        totals: dict[str, list[float]] = defaultdict(list)
        for batch_index, batch in enumerate(loader):
            if args.stage1_val_max_batches > 0 and batch_index >= args.stage1_val_max_batches:
                break
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            loss, parts = structured_stage1_loss(
                output.reconstruction,
                batch.values,
                valid_mask_for_batch(batch),
                model.spatial_tokenizer.repulsion_loss(),
                temporal_difference_weight=args.stage1_temporal_difference_weight,
                anchor_repulsion_weight=args.anchor_repulsion_weight,
            )
            totals["loss"].append(float(loss))
            for key, value in parts.items():
                totals[key].append(float(value))
            metrics = compute_original_scale_metrics(
                output.reconstruction,
                batch.values,
                batch,
                valid_mask_for_batch(batch),
            )
            for key, value in metrics.items():
                totals[f"original_{key}"].append(value)
        result[domain] = _mean_dict(totals)
    return {
        "macro_loss": float(np.mean([values["loss"] for values in result.values()])),
        "per_domain": result,
    }


def clamp_observed(
    prediction: torch.Tensor,
    values: torch.Tensor,
    obs_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if obs_mask is None:
        return prediction
    return obs_mask[..., None] * values + (1 - obs_mask[..., None]) * prediction


def _schedule(
    sigma_start: float,
    sigma_min: float,
    rho: float,
    steps: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    count = max(int(steps), 2)
    indices = torch.arange(count, device=reference.device, dtype=reference.dtype)
    start = max(float(sigma_start), float(sigma_min))
    schedule = (
        start ** (1 / rho)
        + indices / (count - 1)
        * (sigma_min ** (1 / rho) - start ** (1 / rho))
    ) ** rho
    return torch.cat([schedule, torch.zeros_like(schedule[:1])])


def _noise_set(
    reference: torch.Tensor, count: int, seed: int, antithetic: bool
) -> list[torch.Tensor]:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(seed)
    noises = []
    while len(noises) < count:
        noise = torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        noises.append(noise)
        if antithetic and len(noises) < count:
            noises.append(-noise)
    return noises


@torch.no_grad()
def _sample_direct(
    diffusion: StructuredDirectLatentDiffusion,
    condition: DirectConditionOutput,
    base_normalized: torch.Tensor,
    noise: torch.Tensor,
    sigma_start: float,
    steps: int,
) -> torch.Tensor:
    state = base_normalized + noise * sigma_start
    schedule = _schedule(
        sigma_start, diffusion.config.sigma_min, diffusion.config.rho, steps, state
    )
    null_condition = (
        diffusion._null_condition(condition)
        if diffusion.guidance_scale != 1.0
        else None
    )
    for index, (current, following) in enumerate(zip(schedule[:-1], schedule[1:])):
        sigma = state.new_full((state.shape[0],), current)
        denoised = diffusion.denoise(state, sigma, condition)
        if null_condition is not None:
            unconditional = diffusion.denoise(state, sigma, null_condition)
            denoised = unconditional + diffusion.guidance_scale * (
                denoised - unconditional
            )
        derivative = (state - denoised) / current.clamp_min(
            diffusion.config.sigma_min
        )
        euler = state + (following - current) * derivative
        if index < len(schedule) - 2 and following > 0:
            next_sigma = state.new_full((state.shape[0],), following)
            next_denoised = diffusion.denoise(euler, next_sigma, condition)
            if null_condition is not None:
                next_unconditional = diffusion.denoise(
                    euler, next_sigma, null_condition
                )
                next_denoised = next_unconditional + diffusion.guidance_scale * (
                    next_denoised - next_unconditional
                )
            next_derivative = (euler - next_denoised) / following.clamp_min(
                diffusion.config.sigma_min
            )
            state = state + 0.5 * (following - current) * (
                derivative + next_derivative
            )
        else:
            state = euler
    return diffusion.denormalize(state)


def _sampling_sigma(diffusion, args) -> float:
    source = getattr(args, "sampling_sigma_source", "anchor")
    values = {
        "anchor": args.anchor_sigma,
        "sigma_data": diffusion.config.sigma_data,
        "sigma_max": diffusion.config.sigma_max,
        "explicit": getattr(args, "sampling_sigma", None),
    }
    value = float(values[source])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid sampling sigma: {value}")
    return value


@torch.no_grad()
def anchored_predictions(
    model,
    diffusion,
    condition_batch: STBatch,
    output_batch: STBatch,
    args,
    seed: int,
    observed_mask: Optional[torch.Tensor],
):
    diffusion.activate_batch_domain(condition_batch)
    condition = diffusion.condition_encoder(condition_batch)
    center_latent = diffusion.condition_probe(condition.condition_grid)
    center_normalized = diffusion.normalize(center_latent)
    query = build_full_query_coords(output_batch)
    center = clamp_observed(
        decode_neural_field(
            model, center_latent, query, output_batch, condition_batch
        ),
        output_batch.values,
        observed_mask,
    )
    initialization = getattr(args, "sampling_initialization", "center_noise")
    sigma_start = _sampling_sigma(diffusion, args)
    base = center_normalized if initialization != "pure_noise" else torch.zeros_like(center_normalized)
    noises = (
        [torch.zeros_like(center_normalized)]
        if initialization == "center_zero"
        else _noise_set(
            center_normalized,
            args.ensemble_samples,
            seed,
            antithetic=not args.independent_noise,
        )
    )
    predictions = [
        clamp_observed(
            decode_neural_field(
                model,
                _sample_direct(
                    diffusion,
                    condition,
                    base,
                    noise,
                    sigma_start,
                    args.sampling_steps,
                ),
                query,
                output_batch,
                condition_batch,
            ),
            output_batch.values,
            observed_mask,
        )
        for noise in noises
    ]
    stack = torch.stack(predictions)
    return center, stack, stack.mean(dim=0), float(condition.confidence.mean())


@torch.no_grad()
def evaluate_validation(
    model, diffusion, loaders, task_config, device, args, logger
) -> tuple[float, dict]:
    model.eval()
    diffusion.eval()
    batches = int(getattr(args, "val_eval_batches", 2))
    details = {}
    normalized_scores = []
    mask_generator = torch.Generator(device=device)
    mask_generator.manual_seed(args.seed + 800000)
    for domain_index, (domain, loader) in enumerate(loaders.items()):
        details[domain] = {}
        for task_index, task in enumerate(CONDITIONAL_TASKS):
            totals: dict[str, list[float]] = defaultdict(list)
            for batch_index, batch in enumerate(loader):
                if batch_index >= batches:
                    break
                batch = move_batch_to_device(batch, device)
                condition = build_condition_view(
                    batch, task, task_config, generator=mask_generator
                )
                mask = valid_mask_for_batch(batch) * (1.0 - condition.obs_mask)
                center, stack, ensemble, confidence = anchored_predictions(
                    model,
                    diffusion,
                    condition,
                    batch,
                    args,
                    args.seed
                    + 810000
                    + domain_index * 100000
                    + task_index * 10000
                    + batch_index * 100,
                    condition.obs_mask,
                )
                for method, prediction in (
                    ("center", center),
                    ("anchored_s1", stack[0]),
                    ("anchored_ensemble", ensemble),
                ):
                    metrics = compute_original_scale_metrics(
                        prediction, batch.values, batch, mask
                    )
                    for key, value in metrics.items():
                        totals[f"{method}_{key}"].append(value)
                    scale = batch.metadata.get("value_scale")
                    scale_value = 1.0 if scale is None else max(float(scale.float().mean()), 1e-8)
                    totals[f"{method}_nmae"].append(metrics["mae"] / scale_value)
                totals["confidence"].append(confidence)
            summary = _mean_dict(totals)
            details[domain][task.value] = summary
            normalized_scores.append(summary["anchored_ensemble_nmae"])
            logger.info(
                "VALIDATION_CELL | domain=%s task=%s anchored_nmae=%.8f anchored_mae=%.8f",
                domain,
                task.value,
                summary["anchored_ensemble_nmae"],
                summary["anchored_ensemble_mae"],
            )
    score = float(np.mean(normalized_scores))
    details["_selection"] = {
        "macro_nmae": score,
        "weighted_nmae": score,
        "batches_per_cell": batches,
        "evenly_spaced": False,
    }
    return score, details
