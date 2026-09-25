from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Subset

from .training_helpers import (
    anchored_predictions,
    count_parameters,
    evaluate_validation,
    module_gradient_norm,
    run_stage1_validation,
    set_global_seed,
)
from .model.common import (
    BalancedDomainIterator,
    ModelConfig,
    STBatch,
    STTask,
    freeze_module,
    move_batch_to_device,
    valid_mask_for_batch,
)
from .model.diffusion import (
    CONDITIONAL_TASKS,
    CurrentConditionalTaskConfig,
    FoundationGlobalDirectDiffusion,
    merge_test_condition_preserving_domain,
)
from .model.neural_field import (
    AdaptiveGlobalLocalStructuredField,
    FOUNDATION_DOMAIN_FEATURE_NAMES,
    FoundationDomainLoaderView,
    FoundationDomainSpec,
    build_full_query_coords,
    discover_foundation_domain_spec,
    structured_stage1_loss,
)
from .data.loader import (
    ProcessedDomain,
    Protocol,
    TrainWindowDataset,
    audit_dataset_roots,
    compute_scaled_train_time_stop,
    compute_split_indices,
    make_loader,
)

try:
    import setproctitle
except ModuleNotFoundError:  # Optional on lightweight/local smoke environments.
    setproctitle = None

if setproctitle is not None:
    setproctitle.setproctitle('FieldDiff')

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
EXPERIMENT_NAME = "stfm_foundation"
DEFAULT_LOG_DIR = ROOT / "logs" / EXPERIMENT_NAME
DEFAULT_PRETRAIN_TIME_SCALE = 1.0
FIXED_PRETRAIN_TIME_SCALE: Optional[float] = None
DEFAULT_FEW_SHOT_EPOCHS = 1000
DEFAULT_FEW_SHOT_STEPS_PER_EPOCH = 0
TASKS = tuple(task.value for task in CONDITIONAL_TASKS)


def stage1_final_evaluation(
    field,
    store,
    zero_shot_names,
    args,
    device,
    run_dir,
    logger,
    **kwargs,
):
    """Optional experiment hook run after Stage 1 and before Stage 2."""

    return None


def parse_csv(value: str | None) -> list[str]:
    return [] if not value else [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str | None) -> list[float]:
    return [] if not value else [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("all", "audit", "pretrain", "evaluate", "few-shot"), default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data_processed")
    parser.add_argument("--pretrain-datasets", default=None)
    parser.add_argument("--zero-shot-datasets", default=None)
    parser.add_argument("--few-shot-ratios", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=None,
        help=(
            "Load a best_stage1.pt field checkpoint, skip Stage 1, and train "
            "Stage 2 from scratch directly in the raw Stage-1 latent space."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--max-model-nodes",
        type=int,
        default=0,
        help="0 keeps every available node; positive values are debug/ablation-only node caps.",
    )
    parser.add_argument(
        "--batch-node-budget",
        type=int,
        default=4096,
        help="Maximum total full-node samples per training batch; changes batch size, never drops nodes.",
    )

    parser.add_argument("--stage1-epochs", type=int, default=1000)
    parser.add_argument("--stage2-epochs", type=int, default=1000)
    parser.add_argument(
        "--few-shot-epochs", type=int, default=DEFAULT_FEW_SHOT_EPOCHS
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help=(
            "Stop after the best Stage-1 checkpoint and its final evaluation; "
            "do not construct or train Stage 2."
        ),
    )
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument(
        "--few-shot-steps-per-epoch",
        type=int,
        default=DEFAULT_FEW_SHOT_STEPS_PER_EPOCH,
    )
    parser.add_argument(
        "--pretrain-time-scale",
        type=float,
        default=DEFAULT_PRETRAIN_TIME_SCALE,
        help=(
            "Chronological fraction of each pretraining domain's original train "
            "time interval to expose. The held-out test interval is unchanged."
        ),
    )
    parser.add_argument("--max-pretrain-hours", type=float, default=None)
    parser.add_argument("--stage1-time-fraction", type=float, default=None)
    parser.add_argument("--lr-stage1", type=float, default=3e-4)
    parser.add_argument("--lr-stage2", type=float, default=3e-4)
    parser.add_argument("--few-shot-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-6)
    parser.add_argument("--stage1-patience", type=int, default=250)
    parser.add_argument("--stage2-patience", type=int, default=250)
    parser.add_argument("--few-shot-patience", type=int, default=100)

    parser.add_argument("--model-dim", type=int, default=128)
    # The reference runs under logs/stfm_foundation_v2_anchor_affine_temporal_split
    # use 48 total slots (16 global + 32 local).
    parser.add_argument("--num-spatial-latents", type=int, default=48)
    parser.add_argument("--spatial-tokenizer-layers", type=int, default=2)
    parser.add_argument("--stage1-axial-layers", type=int, default=4)
    parser.add_argument("--stage1-heads", type=int, default=4)
    parser.add_argument("--stage1-temporal-attention-radius", type=int, default=3)
    parser.add_argument("--decoder-temporal-radius", type=int, default=2)
    parser.add_argument("--stage1-temporal-difference-weight", type=float, default=0.05)
    parser.add_argument("--anchor-repulsion-weight", type=float, default=1e-4)
    parser.add_argument("--global-spatial-fourier-scale", type=float, default=2.0)
    parser.add_argument("--local-spatial-fourier-scale", type=float, default=6.0)

    parser.add_argument("--condition-temporal-layers", type=int, default=2)
    parser.add_argument("--condition-axial-layers", type=int, default=2)
    parser.add_argument("--confidence-bandwidth", type=float, default=0.25)
    parser.add_argument("--missing-latent-loss-weight", type=float, default=2.0)
    parser.add_argument("--decoded-loss-weight", type=float, default=0.05)
    parser.add_argument("--condition-latent-loss-weight", type=float, default=0.25)
    parser.add_argument("--condition-decoded-loss-weight", type=float, default=0.05)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--shuffle-loss-weight", type=float, default=0.05)
    parser.add_argument("--shuffle-margin", type=float, default=0.05)
    parser.add_argument("--shuffle-every", type=int, default=5)
    parser.add_argument("--diffusion-width", type=int, default=256)
    parser.add_argument("--diffusion-depth", type=int, default=8)
    parser.add_argument("--diffusion-heads", type=int, default=8)
    parser.add_argument("--latent-stats-max-batches", type=int, default=64)

    parser.add_argument("--anchor-sigma", type=float, default=2.0)
    parser.add_argument("--ensemble-samples", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=12)
    parser.add_argument("--independent-noise", action="store_true")
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--validation-batches-per-domain", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument("--save-few-shot-checkpoints", action="store_true")

    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--train-stride", type=int, default=None)
    parser.add_argument("--eval-stride", type=int, default=None)
    parser.add_argument("--node-train-ratio", type=float, default=None)
    parser.add_argument("--time-train-ratio", type=float, default=None)
    parser.add_argument("--train-val-ratio", type=float, default=None)
    parser.add_argument("--forecasting-context-steps", type=int, default=None)
    parser.add_argument("--interpolation-observed-steps", type=int, default=None)
    parser.add_argument("--spatial-extrapolation-mask-ratio", type=float, default=None)
    parser.add_argument("--spatiotemporal-imputation-mask-ratio", type=float, default=None)
    parser.add_argument("--normalization-method", choices=("standard", "minmax"), default=None)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_experiment_config(args: argparse.Namespace) -> tuple[list[str], list[str], list[float], dict]:
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = raw.get("protocol", {})
    training = raw.get("training", {})
    mapping = {
        "window_size": "window_size",
        "stride": "train_stride",
        "eval_stride": "eval_stride",
        "node_train_ratio": "node_train_ratio",
        "time_train_ratio": "time_train_ratio",
        "train_val_ratio": "train_val_ratio",
        "forecasting_context_steps": "forecasting_context_steps",
        "interpolation_observed_steps": "interpolation_observed_steps",
        "spatial_extrapolation_mask_ratio": "spatial_extrapolation_mask_ratio",
        "spatiotemporal_imputation_mask_ratio": "spatiotemporal_imputation_mask_ratio",
        "normalization_method": "normalization_method",
    }
    fallbacks = {
        "window_size": 24,
        "train_stride": 1,
        "eval_stride": 24,
        "node_train_ratio": 0.75,
        "time_train_ratio": 0.70,
        "train_val_ratio": 0.90,
        "forecasting_context_steps": 12,
        "interpolation_observed_steps": 12,
        "spatial_extrapolation_mask_ratio": 0.50,
        "spatiotemporal_imputation_mask_ratio": 0.50,
        "normalization_method": "standard",
    }
    for config_key, arg_key in mapping.items():
        if getattr(args, arg_key) is None:
            setattr(args, arg_key, protocol.get(config_key, fallbacks[arg_key]))
    if args.max_pretrain_hours is None:
        args.max_pretrain_hours = float(training.get("max_pretrain_hours", 48.0))
    if args.stage1_time_fraction is None:
        args.stage1_time_fraction = float(training.get("stage1_time_fraction", 0.35))
    pretrain = parse_csv(args.pretrain_datasets) or list(raw["pretrain_datasets"])
    zero = parse_csv(args.zero_shot_datasets) or list(raw["zero_shot_datasets"])
    ratios = parse_float_csv(args.few_shot_ratios) or [float(value) for value in raw["few_shot_ratios"]]
    if set(pretrain) & set(zero):
        raise ValueError(f"Pretrain and zero-shot datasets overlap: {sorted(set(pretrain) & set(zero))}")
    if len(pretrain) != len(set(pretrain)) or len(zero) != len(set(zero)):
        raise ValueError("Dataset lists contain duplicates")
    if any(not 0.0 < ratio <= 1.0 for ratio in ratios):
        raise ValueError("Few-shot ratios must be in (0,1]")
    return pretrain, zero, sorted(set(ratios)), raw


def make_protocol(args: argparse.Namespace) -> Protocol:
    return Protocol(
        window_size=args.window_size,
        stride=args.train_stride,
        eval_stride=args.eval_stride,
        node_train_ratio=args.node_train_ratio,
        time_train_ratio=args.time_train_ratio,
        train_val_ratio=args.train_val_ratio,
        forecasting_context_steps=args.forecasting_context_steps,
        interpolation_observed_steps=args.interpolation_observed_steps,
        spatial_extrapolation_mask_ratio=args.spatial_extrapolation_mask_ratio,
        spatiotemporal_imputation_mask_ratio=args.spatiotemporal_imputation_mask_ratio,
        normalization_method=args.normalization_method,
        seed=args.seed,
    )


def validate_args(args: argparse.Namespace) -> torch.device:
    if args.smoke_test:
        args.stage1_epochs = args.stage2_epochs = args.few_shot_epochs = 1
        args.steps_per_epoch = args.few_shot_steps_per_epoch = 2
        args.validation_interval = 1
        args.validation_batches_per_domain = 1
        args.max_eval_batches = 2
        args.latent_stats_max_batches = 1
        args.batch_size = 2
        args.eval_batch_size = 1
        args.max_model_nodes = 16 if args.max_model_nodes == 0 else min(args.max_model_nodes, 16)
        args.batch_node_budget = 32
        args.model_dim = 32
        args.num_spatial_latents = 4
        args.spatial_tokenizer_layers = 1
        args.stage1_axial_layers = 1
        args.stage1_heads = 4
        args.condition_temporal_layers = 1
        args.condition_axial_layers = 1
        args.diffusion_width = 64
        args.diffusion_depth = 2
        args.diffusion_heads = 4
        args.ensemble_samples = 2
        args.sampling_steps = 2
        args.max_pretrain_hours = 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {args.device}")
    return device


def setup_run(log_dir: Path) -> tuple[Path, logging.Logger]:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(EXPERIMENT_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(run_dir / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return run_dir, logger


class DomainStore:
    def __init__(
        self,
        data_root: Path,
        protocol: Protocol,
        logger: logging.Logger,
        train_time_scales: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.data_root = data_root
        self.protocol = protocol
        self.logger = logger
        self.train_time_scales = dict(train_time_scales or {})
        self.domains: dict[str, ProcessedDomain] = {}
        self.specs: dict[str, FoundationDomainSpec] = {}

    def get(self, name: str) -> ProcessedDomain:
        if name not in self.domains:
            domain = ProcessedDomain(
                name,
                self.data_root / name,
                self.protocol,
                train_time_scale=self.train_time_scales.get(name, 1.0),
            )
            spec = discover_foundation_domain_spec(domain)
            self.domains[name] = domain
            self.specs[name] = spec
            self.logger.info("DOMAIN_SPEC | %s", json.dumps(spec.to_json()))
            self.logger.info(
                "DOMAIN_TIME_SCALE | domain=%s scale=%.2f train_time_slice=[0,%d) "
                "original_train_time_slice=[0,%d) test_time_slice=[%d,%d)",
                name,
                domain.train_time_scale,
                domain.train_time_stop,
                domain.split.train_time_stop,
                domain.split.train_time_stop,
                domain.split.num_time_steps,
            )
        return self.domains[name]

    def spec(self, name: str) -> FoundationDomainSpec:
        self.get(name)
        return self.specs[name]


def make_loaders(store: DomainStore, names: Sequence[str], args: argparse.Namespace):
    train, val = {}, {}
    for index, name in enumerate(names):
        domain = store.get(name)
        max_nodes = effective_max_nodes(domain, args)
        train_batch_size = effective_train_batch_size(domain, args)
        train_dataset, val_dataset = domain.train_val_datasets(max_nodes)
        train[name] = FoundationDomainLoaderView(
            make_loader(train_dataset, train_batch_size, True, args.num_workers, args.seed + 1009 + index),
            store.spec(name),
        )
        val[name] = FoundationDomainLoaderView(
            make_loader(val_dataset, train_batch_size, False, args.num_workers, args.seed + 2003 + index),
            store.spec(name),
        )
        store.logger.info(
            "DOMAIN_LOADER | domain=%s total_nodes=%d train_nodes=%d model_nodes=%d batch_size=%d",
            name, domain.shape[1], len(domain.split.train_nodes),
            min(len(domain.split.train_nodes), max_nodes), train_batch_size,
        )
    return train, val


def effective_max_nodes(domain: ProcessedDomain, args: argparse.Namespace) -> int:
    return domain.shape[1] if args.max_model_nodes == 0 else min(args.max_model_nodes, domain.shape[1])


def effective_train_batch_size(domain: ProcessedDomain, args: argparse.Namespace) -> int:
    nodes = min(len(domain.split.train_nodes), effective_max_nodes(domain, args))
    return min(args.batch_size, max(1, args.batch_node_budget // max(nodes, 1)))


def make_model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        time_steps=args.window_size,
        time_coord_dim=4,
        use_periodic_time_features=True,
        coord_embed_dim=args.model_dim,
        time_embed_dim=args.model_dim,
        token_dim=args.model_dim,
        latent_dim=args.model_dim,
        decoder_query_dim=args.model_dim,
        diffusion_width=args.diffusion_width,
        diffusion_depth=args.diffusion_depth,
        diffusion_heads=args.diffusion_heads,
        sampling_steps=args.sampling_steps,
        self_heads=args.stage1_heads,
        cross_heads=args.stage1_heads,
    )


def build_field(args: argparse.Namespace, device: torch.device) -> AdaptiveGlobalLocalStructuredField:
    return AdaptiveGlobalLocalStructuredField(
        make_model_config(args),
        domain_feature_dim=len(FOUNDATION_DOMAIN_FEATURE_NAMES),
        global_fourier_scale=args.global_spatial_fourier_scale,
        local_fourier_scale=args.local_spatial_fourier_scale,
        num_spatial_latents=args.num_spatial_latents,
        spatial_tokenizer_layers=args.spatial_tokenizer_layers,
        axial_layers=args.stage1_axial_layers,
        heads=args.stage1_heads,
        decoder_temporal_radius=args.decoder_temporal_radius,
        temporal_attention_radius=args.stage1_temporal_attention_radius,
    ).to(device)


def build_diffusion(
    args: argparse.Namespace,
    field: AdaptiveGlobalLocalStructuredField,
    device: torch.device,
) -> FoundationGlobalDirectDiffusion:
    return FoundationGlobalDirectDiffusion(
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


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def mean_dict(values: Mapping[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(items)) for key, items in values.items() if items}


def stage1_train_epoch(model, loaders, optimizer, args, device, epoch, logger, deadline=None) -> dict:
    model.train()
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    full_steps = sum(len(loader) for loader in loaders.values())
    steps = args.steps_per_epoch or full_steps
    completed = 0
    epoch_start = time.monotonic()
    for domain, batch in BalancedDomainIterator(loaders, steps, offset=(epoch - 1) % len(loaders)):
        if deadline is not None and time.monotonic() >= deadline:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, parts = structured_stage1_loss(
            output.reconstruction,
            batch.values,
            valid_mask_for_batch(batch),
            model.spatial_tokenizer.repulsion_loss(),
            temporal_difference_weight=args.stage1_temporal_difference_weight,
            anchor_repulsion_weight=args.anchor_repulsion_weight,
        )
        auxiliary = getattr(model, "stage1_auxiliary_loss", None)
        if auxiliary is not None:
            auxiliary_loss, auxiliary_parts = auxiliary()
            loss = loss + auxiliary_loss
            parts = {**parts, "adaptive_router_loss": auxiliary_loss, **auxiliary_parts}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        totals[domain]["loss"].append(float(loss.detach()))
        for key, value in parts.items():
            totals[domain][key].append(float(value.detach()))
        completed += 1
        if completed == 1 or (args.log_every > 0 and completed % args.log_every == 0):
            elapsed = max(time.monotonic() - epoch_start, 1e-9)
            rate = completed / elapsed
            logger.info(
                "STAGE1_STEP | epoch=%d step=%d/%d domain=%s loss=%.8f steps_per_second=%.3f eta_hours=%.2f",
                epoch, completed, steps, domain, float(loss.detach()), rate,
                (steps - completed) / max(rate, 1e-9) / 3600.0,
            )
    per_domain = {name: mean_dict(values) for name, values in totals.items()}
    losses = [row["loss"] for row in per_domain.values()]
    return {
        "macro_loss": float(np.mean(losses)) if losses else float("nan"),
        "steps": completed,
        "requested_steps": steps,
        "time_limit_reached": completed < steps,
        "per_domain": per_domain,
    }


def stage2_train_epoch(
    model, diffusion, loaders, optimizer, task_config, args, device, epoch, logger,
    deadline=None, steps_override=None, phase="STAGE2",
):
    model.eval()
    diffusion.train()
    totals: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    full_steps = sum(len(loader) for loader in loaders.values())
    steps = steps_override or args.steps_per_epoch or full_steps
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 10000 + epoch)
    condition_gradients, backbone_gradients = [], []
    completed = 0
    epoch_start = time.monotonic()
    num_domains = len(loaders)
    iterator = BalancedDomainIterator(loaders, steps, offset=(epoch - 1) % num_domains)
    for step, (domain, batch) in enumerate(iterator):
        if deadline is not None and time.monotonic() >= deadline:
            break
        batch = move_batch_to_device(batch, device)
        task = CONDITIONAL_TASKS[((step // num_domains) + epoch - 1) % len(CONDITIONAL_TASKS)]
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            target_latent = model.encode(batch)
        output = diffusion.loss_for_task(
            target_latent,
            batch,
            task,
            task_config,
            generator=generator,
            decoded_loss_weight=args.decoded_loss_weight,
            condition_latent_loss_weight=args.condition_latent_loss_weight,
            condition_decoded_loss_weight=args.condition_decoded_loss_weight,
            shuffle_loss_weight=args.shuffle_loss_weight,
            shuffle_margin=args.shuffle_margin,
            compute_shuffle_loss=args.shuffle_every > 0 and step % args.shuffle_every == 0,
        )
        output.loss.backward()
        if step == 0:
            condition_gradients.append(module_gradient_norm(diffusion.condition_encoder))
            backbone_gradients.append(module_gradient_norm(diffusion.backbone))
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.grad_clip)
        optimizer.step()
        for key in (
            "loss", "edm_loss", "decoded_loss", "condition_latent_loss",
            "condition_decoded_loss", "shuffle_loss", "condition_advantage",
            "mean_sigma", "mean_confidence",
        ):
            totals[domain][task.value][key].append(float(getattr(output, key).detach()))
        if output.mean_uncertainty is not None:
            totals[domain][task.value]["mean_uncertainty"].append(
                float(output.mean_uncertainty.detach())
            )
        completed += 1
        if completed == 1 or (args.log_every > 0 and completed % args.log_every == 0):
            elapsed = max(time.monotonic() - epoch_start, 1e-9)
            rate = completed / elapsed
            logger.info(
                "%s_STEP | epoch=%d step=%d/%d domain=%s task=%s loss=%.8f steps_per_second=%.3f eta_hours=%.2f",
                phase, epoch, completed, steps, domain, task.value, float(output.loss.detach()), rate,
                (steps - completed) / max(rate, 1e-9) / 3600.0,
            )
    per_domain = {
        name: {task: mean_dict(values) for task, values in tasks.items()}
        for name, tasks in totals.items()
    }
    losses = [row["loss"] for tasks in per_domain.values() for row in tasks.values()]
    return {
        "macro_loss": float(np.mean(losses)) if losses else float("nan"),
        "steps": completed,
        "requested_steps": steps,
        "time_limit_reached": completed < steps,
        "condition_gradient_norm": float(np.mean(condition_gradients)) if condition_gradients else 0.0,
        "backbone_gradient_norm": float(np.mean(backbone_gradients)) if backbone_gradients else 0.0,
        "per_domain_task": per_domain,
    }


def task_config_from_args(args: argparse.Namespace) -> CurrentConditionalTaskConfig:
    return CurrentConditionalTaskConfig(
        forecasting_context_steps=args.forecasting_context_steps,
        interpolation_observed_steps=args.interpolation_observed_steps,
        spatial_observed_node_ratio=args.node_train_ratio,
        spatiotemporal_observed_node_ratio=1.0 - args.spatiotemporal_imputation_mask_ratio,
    )


def foundation_validation_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build arguments accepted by both old and new canonical evaluators."""

    validation_args = copy.copy(args)
    validation_args.val_eval_batches = args.validation_batches_per_domain
    # Foundation checkpoint selection historically uses the first N batches
    # and an unweighted macro over all four tasks. Set these explicitly so a
    # server with the older shared evaluator never depends on missing fields.
    validation_args.evenly_spaced_validation = False
    validation_args.spatial_validation_weight = 1.0
    validation_args.spatiotemporal_validation_weight = 1.0
    return validation_args


def checkpoint_payload(field, diffusion, args, protocol, pretrain_names, specs, epoch, score) -> dict:
    model_hparams = {
        key: getattr(args, key)
        for key in (
            "model_dim", "num_spatial_latents", "spatial_tokenizer_layers",
            "stage1_axial_layers", "stage1_heads", "stage1_temporal_attention_radius",
            "decoder_temporal_radius", "global_spatial_fourier_scale",
            "local_spatial_fourier_scale", "condition_temporal_layers",
            "condition_axial_layers", "confidence_bandwidth", "missing_latent_loss_weight",
            "condition_dropout", "guidance_scale", "diffusion_width", "diffusion_depth",
            "diffusion_heads", "sampling_steps",
        )
    }
    payload = {
        "format_version": 2,
        "experiment": EXPERIMENT_NAME,
        "field": cpu_state_dict(field),
        "diffusion": cpu_state_dict(diffusion),
        "model_hparams": model_hparams,
        "protocol": asdict(protocol),
        "pretrain_datasets": list(pretrain_names),
        "pretrain_time_scale": args.pretrain_time_scale,
        "pretrain_domain_specs": {name: specs[name].to_json() for name in pretrain_names},
        "epoch": epoch,
        "validation_score": score,
        "zero_shot_contract": {
            "learned_domain_id": False,
            "latent_normalization": False,
            "latent_whitening": None,
            "geometry": "unit_ecef_plus_adaptive_domain_centered_ecef",
            "missing_value_condition": "learned_embedding",
            "condition_spatial_confidence": "stage1_tokenizer_attention_mass_on_observed_points",
        },
    }
    if getattr(args, "model_scale_contract", None) is not None:
        payload["model_scale_contract"] = args.model_scale_contract
    return payload


def stage1_checkpoint_payload(
    field, args, protocol, pretrain_names, specs, epoch, score
) -> dict:
    """Self-describing field-only checkpoint for Stage-1 ablations."""

    names = (
        "model_dim",
        "num_spatial_latents",
        "num_global_spatial_latents",
        "spatial_tokenizer_layers",
        "stage1_axial_layers",
        "stage1_heads",
        "stage1_temporal_attention_radius",
        "decoder_temporal_radius",
        "global_spatial_fourier_scale",
        "local_spatial_fourier_scale",
        "assignment_temperature",
        "sinkhorn_iterations",
        "value_canonical_variant",
        "sigma_data",
        "coverage_moment_weight",
        "coverage_covariance_weight",
        "coverage_swd_weight",
        "coverage_projection_count",
        "coverage_max_tokens",
        "coverage_warmup_epochs",
        "coverage_ramp_epochs",
        "anchor_scale_floor_ratio",
        "anchor_mad_to_std",
        "vae_kl_beta",
        "vae_kl_warmup_epochs",
        "vae_free_bits",
        "posterior_logvar_min",
        "posterior_logvar_max",
        "temporal_carrier_hidden_dim",
        "temporal_carrier_heads",
        "temporal_carrier_layers",
        "temporal_carrier_max_mean_correction",
        "temporal_carrier_max_log_scale_correction",
        "temporal_anchor_mean_loss_weight",
        "temporal_anchor_scale_loss_weight",
        "cama_fingerprint_max_nodes",
        "cama_temporal_bins",
        "cama_hidden_dim",
        "cama_metric_dim",
        "cama_topk",
        "cama_gate_bias",
        "cama_metric_log_scale",
    )
    payload = {
        "format_version": 2,
        "checkpoint_stage": "stage1",
        "experiment": EXPERIMENT_NAME,
        "field": cpu_state_dict(field),
        "model_hparams": {
            name: getattr(args, name) for name in names if hasattr(args, name)
        },
        "protocol": asdict(protocol),
        "pretrain_datasets": list(pretrain_names),
        "pretrain_time_scale": args.pretrain_time_scale,
        "pretrain_domain_specs": {
            name: specs[name].to_json() for name in pretrain_names
        },
        "epoch": epoch,
        "score": score,
        "zero_shot_contract": {
            "target_domain_parameter_updates": 0,
            "stage1_only_checkpoint": True,
        },
    }
    if hasattr(field, "posterior_mean"):
        payload["zero_shot_contract"].update(
            {
                "stage1_latent_model": "weak_variational_autoencoder",
                "evaluation_latent": "posterior_mean",
                "deterministic_value_anchor_channels": 4,
            }
        )
    if getattr(args, "model_scale_contract", None) is not None:
        payload["model_scale_contract"] = args.model_scale_contract
    cama_initialization = getattr(field, "cama_initialization", None)
    if cama_initialization is not None:
        payload["cama_initialization"] = cama_initialization
    temporal_carrier_initialization = getattr(
        field, "temporal_carrier_initialization", None
    )
    if temporal_carrier_initialization is not None:
        payload["temporal_carrier_initialization"] = temporal_carrier_initialization
    return payload


def apply_checkpoint_settings(args: argparse.Namespace, checkpoint: dict) -> None:
    for key, value in checkpoint.get("model_hparams", {}).items():
        setattr(args, key, value)
    mapping = {
        "window_size": "window_size", "stride": "train_stride", "eval_stride": "eval_stride",
        "node_train_ratio": "node_train_ratio", "time_train_ratio": "time_train_ratio",
        "train_val_ratio": "train_val_ratio", "forecasting_context_steps": "forecasting_context_steps",
        "interpolation_observed_steps": "interpolation_observed_steps",
        "spatial_extrapolation_mask_ratio": "spatial_extrapolation_mask_ratio",
        "spatiotemporal_imputation_mask_ratio": "spatiotemporal_imputation_mask_ratio",
        "normalization_method": "normalization_method",
    }
    for source, target in mapping.items():
        if source in checkpoint.get("protocol", {}):
            setattr(args, target, checkpoint["protocol"][source])
    if "pretrain_time_scale" in checkpoint:
        args.pretrain_time_scale = float(checkpoint["pretrain_time_scale"])


def load_foundation(checkpoint: dict, args, device):
    format_version = int(checkpoint.get("format_version", 1))
    if format_version < 2:
        raise ValueError(
            "This Stage-2 checkpoint predates the Stage-1-aligned condition encoder "
            "(learned unknown-value token and tokenizer-attention confidence). Reuse "
            "its best_stage1.pt with --stage1-checkpoint and retrain Stage 2."
        )
    field = build_field(args, device)
    field.load_state_dict(checkpoint["field"], strict=True)
    freeze_module(field)
    diffusion = build_diffusion(args, field, device)
    diffusion.load_state_dict(checkpoint["diffusion"], strict=True)
    return field, diffusion


def pretrain_foundation(names, store, args, device, run_dir, logger):
    train, val = make_loaders(store, names, args)
    field = build_field(args, device)
    set_router_reference_features = getattr(field, "set_router_reference_features", None)
    if set_router_reference_features is not None:
        set_router_reference_features([store.spec(name).features for name in names])
    logger.info("STAGE1_PARAMETERS | total=%d trainable=%d", *count_parameters(field))
    start = time.monotonic()
    total_deadline = start + args.max_pretrain_hours * 3600 if args.max_pretrain_hours > 0 else None
    # Keep the same Stage-1 wall-clock allocation in Stage-1-only ablations so
    # their optimization budget remains directly comparable with full runs.
    stage1_deadline = (
        start + args.max_pretrain_hours * args.stage1_time_fraction * 3600
        if args.max_pretrain_hours > 0
        else None
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_stage1, best_stage1_epoch, bad = math.inf, 0, 0
    stage1_history = []
    if args.stage1_checkpoint is not None:
        source_stage1 = torch.load(
            args.stage1_checkpoint, map_location=device, weights_only=False
        )
        if not isinstance(source_stage1, dict) or "field" not in source_stage1:
            raise ValueError(
                "--stage1-checkpoint must be a best_stage1.pt payload containing 'field'."
            )
        if hasattr(field, "posterior_mean") and not any(
            key.startswith("posterior_mean.") for key in source_stage1["field"]
        ):
            raise ValueError(
                "This Stage-1 checkpoint predates the VAE encoder and has no "
                "posterior_mean parameters. Retrain Stage 1 with the current model."
            )
        if (
            getattr(args, "requested_ablation_variant", None)
            == "v2_ablation_independent_condition"
            and source_stage1.get("model_hparams", {}).get(
                "value_canonical_variant"
            )
            != "v2_anchor_affine"
        ):
            raise ValueError(
                "The independent-condition control must reuse a Full V2 "
                "anchor-affine best_stage1.pt checkpoint"
            )
        field.load_state_dict(source_stage1["field"], strict=True)
        best_stage1_epoch = int(source_stage1.get("epoch", 0))
        best_stage1 = float(source_stage1.get("score", math.nan))
        stage1_history.append(
            {
                "skipped": True,
                "source_checkpoint": str(args.stage1_checkpoint),
                "source_epoch": best_stage1_epoch,
                "source_score": best_stage1,
            }
        )
        loaded_payload = stage1_checkpoint_payload(
            field,
            args,
            store.protocol,
            names,
            store.specs,
            best_stage1_epoch,
            best_stage1,
        )
        loaded_payload["source_checkpoint"] = str(args.stage1_checkpoint)
        torch.save(loaded_payload, checkpoint_dir / "best_stage1.pt")
        (run_dir / "stage1_history.json").write_text(
            json.dumps(stage1_history, indent=2), encoding="utf-8"
        )
        logger.info(
            "STAGE1_LOADED | path=%s epoch=%d score=%s; Stage 1 training skipped",
            args.stage1_checkpoint,
            best_stage1_epoch,
            best_stage1,
        )
    else:
        optimizer = torch.optim.AdamW(
            field.parameters(), lr=args.lr_stage1, weight_decay=args.weight_decay
        )
        for epoch in range(1, args.stage1_epochs + 1):
            train_metrics = stage1_train_epoch(
                field, train, optimizer, args, device, epoch, logger, stage1_deadline
            )
            validation = run_stage1_validation(field, val, device, argparse.Namespace(
                stage1_val_max_batches=args.validation_batches_per_domain,
                stage1_temporal_difference_weight=args.stage1_temporal_difference_weight,
                anchor_repulsion_weight=args.anchor_repulsion_weight,
            ))
            score = validation["macro_loss"]
            if not math.isfinite(score):
                raise FloatingPointError(
                    "Stage-1 validation macro loss is non-finite; refusing to "
                    "continue without a valid best_stage1 checkpoint. Inspect "
                    "the final finite/non-finite training step in run.log."
                )
            row = {"epoch": epoch, "train": train_metrics, "validation": validation}
            stage1_history.append(row)
            logger.info("STAGE1_EPOCH | epoch=%d train=%s validation_macro=%.8f", epoch, json.dumps(train_metrics), score)
            if score < best_stage1 - args.early_stop_min_delta:
                best_stage1, best_stage1_epoch, bad = score, epoch, 0
                torch.save(
                    stage1_checkpoint_payload(
                        field,
                        args,
                        store.protocol,
                        names,
                        store.specs,
                        epoch,
                        score,
                    ),
                    checkpoint_dir / "best_stage1.pt",
                )
            else:
                bad += 1
            (run_dir / "stage1_history.json").write_text(json.dumps(stage1_history, indent=2), encoding="utf-8")
            if train_metrics["time_limit_reached"] or (args.stage1_patience > 0 and bad >= args.stage1_patience):
                break
        best_stage1_path = checkpoint_dir / "best_stage1.pt"
        if not best_stage1_path.is_file():
            raise RuntimeError(
                "Stage 1 finished without producing best_stage1.pt. This means "
                "no finite validation score was accepted."
            )
        stage1_checkpoint = torch.load(
            best_stage1_path, map_location=device, weights_only=False
        )
        field.load_state_dict(stage1_checkpoint["field"], strict=True)
        del optimizer

    stage1_final = stage1_final_evaluation(
        field,
        store,
        getattr(args, "resolved_zero_shot_datasets", ()),
        args,
        device,
        run_dir,
        logger,
        best_stage1_epoch=best_stage1_epoch,
        best_stage1_validation=best_stage1,
    )

    diagnostic_names = names if hasattr(field, "stage1_auxiliary_loss") else names[:1]
    for diagnostic_name in diagnostic_names:
        diagnostic_batch = move_batch_to_device(next(iter(train[diagnostic_name])), device)
        logger.info(
            "GEOMETRY_PREFLIGHT | domain=%s diagnostics=%s",
            diagnostic_name,
            json.dumps(field.geometry_diagnostics(diagnostic_batch)),
        )
    if args.stage1_only:
        return field, None, None, {
            "stage1_only": True,
            "best_stage1_epoch": best_stage1_epoch,
            "best_stage1_validation": best_stage1,
            "best_stage2_epoch": None,
            "best_stage2_validation": None,
            "elapsed_hours": (time.monotonic() - start) / 3600,
            "latent_stats": None,
            "stage1_final_evaluation": stage1_final,
        }
    freeze_module(field)

    diffusion = build_diffusion(args, field, device)
    fit_pretraining_latent_stats = getattr(
        diffusion, "fit_pretraining_latent_stats", None
    )
    if fit_pretraining_latent_stats is None:
        diffusion.set_latent_stats(
            torch.zeros_like(diffusion.latent_mean),
            torch.ones_like(diffusion.latent_std),
        )
        latent_stats = {
            "enabled": False,
            "mode": "raw_stage1_latents",
            "statistics_batches": 0,
            "mean": 0.0,
            "std": 1.0,
        }
        logger.info("RAW_LATENT_CONTRACT | %s", json.dumps(latent_stats))
    else:
        latent_stats = fit_pretraining_latent_stats(
            train, device, args.latent_stats_max_batches
        )
        logger.info("LATENT_STATS | %s", json.dumps(latent_stats))
    logger.info("STAGE2_PARAMETERS | total=%d trainable=%d", *count_parameters(diffusion))
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr_stage2, weight_decay=args.weight_decay)
    task_config = task_config_from_args(args)
    best_stage2, best_stage2_epoch, bad = math.inf, 0, 0
    stage2_history = []
    for epoch in range(1, args.stage2_epochs + 1):
        train_metrics = stage2_train_epoch(
            field, diffusion, train, optimizer, task_config, args, device, epoch, logger, total_deadline
        )
        row = {"epoch": epoch, "train": train_metrics}
        should_eval = (
            train_metrics["time_limit_reached"] or epoch == 1 or epoch == args.stage2_epochs
            or epoch % args.validation_interval == 0
        )
        if should_eval:
            validation_args = foundation_validation_args(args)
            score, details = evaluate_validation(
                field, diffusion, val, task_config, device, validation_args, logger
            )
            row.update({"validation_macro_nmae": score, "validation": details})
            logger.info("STAGE2_VALIDATION | epoch=%d macro_nmae=%.8f", epoch, score)
            if score < best_stage2 - args.early_stop_min_delta:
                best_stage2, best_stage2_epoch, bad = score, epoch, 0
                torch.save(
                    checkpoint_payload(field, diffusion, args, store.protocol, names, store.specs, epoch, score),
                    checkpoint_dir / "best_foundation.pt",
                )
            else:
                bad += args.validation_interval if epoch > 1 else 1
        stage2_history.append(row)
        (run_dir / "pretrain_history.json").write_text(
            json.dumps({"stage1": stage1_history, "stage2": stage2_history}, indent=2), encoding="utf-8"
        )
        logger.info("STAGE2_EPOCH | epoch=%d train=%s", epoch, json.dumps(train_metrics))
        if train_metrics["time_limit_reached"] or (args.stage2_patience > 0 and bad >= args.stage2_patience):
            break
    checkpoint = torch.load(checkpoint_dir / "best_foundation.pt", map_location="cpu", weights_only=False)
    field, diffusion = load_foundation(checkpoint, args, device)
    return field, diffusion, checkpoint, {
        "best_stage1_epoch": best_stage1_epoch,
        "best_stage1_validation": best_stage1,
        "best_stage2_epoch": best_stage2_epoch,
        "best_stage2_validation": best_stage2,
        "elapsed_hours": (time.monotonic() - start) / 3600,
        "latent_stats": latent_stats,
        "stage1_final_evaluation": stage1_final,
    }


def evenly_spaced_subset(dataset, maximum: int):
    if maximum <= 0 or len(dataset) <= maximum:
        return dataset
    ids = np.linspace(0, len(dataset) - 1, maximum, dtype=np.int64)
    return Subset(dataset, ids.tolist())


def new_totals():
    return {"absolute_error": 0.0, "squared_error": 0.0, "count": 0.0}


def update_totals(totals, prediction, target, mask, batch):
    offset = batch.metadata["value_offset"].to(prediction.device, prediction.dtype)
    scale = batch.metadata["value_scale"].to(prediction.device, prediction.dtype)
    while offset.ndim < prediction.ndim:
        offset = offset.unsqueeze(-1)
        scale = scale.unsqueeze(-1)
    prediction = prediction * scale + offset
    target = target * scale + offset
    expanded = mask.to(prediction.dtype)
    if expanded.ndim == prediction.ndim - 1:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(prediction)
    error = (prediction - target) * expanded
    totals["absolute_error"] += float(error.abs().sum())
    totals["squared_error"] += float(error.square().sum())
    totals["count"] += float(expanded.sum())


@torch.no_grad()
def evaluate_task(field, diffusion, domain, spec, task, args, device):
    dataset = evenly_spaced_subset(
        domain.task_dataset(task, effective_max_nodes(domain, args)), args.max_eval_batches
    )
    loader = FoundationDomainLoaderView(
        make_loader(dataset, args.eval_batch_size, False, args.num_workers, args.seed + 9001), spec
    )
    ensemble_method = (
        "standard_noise_ensemble"
        if getattr(args, "sampling_initialization", "center_noise") == "pure_noise"
        else "anchored_ensemble"
    )
    totals = {"deterministic_center": new_totals(), ensemble_method: new_totals()}
    field.eval(); diffusion.eval()
    task_index = TASKS.index(task)
    for batch_index, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        condition = merge_test_condition_preserving_domain(batch)
        eval_mask = batch.metadata.get("eval_mask", valid_mask_for_batch(batch))
        center, _, ensemble, _ = anchored_predictions(
            field, diffusion, condition, batch, args,
            args.seed + 900000 + task_index * 10000 + batch_index * 100,
            batch.obs_mask,
        )
        update_totals(totals["deterministic_center"], center, batch.values, eval_mask, batch)
        update_totals(totals[ensemble_method], ensemble, batch.values, eval_mask, batch)
    result = {}
    for method, values in totals.items():
        count = max(values["count"], 1.0)
        mse = values["squared_error"] / count
        result[method] = {
            "mae": values["absolute_error"] / count,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "count": int(values["count"]),
        }
    return result


def append_result(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_group(field, diffusion, names, regime, store, args, device, path, logger, ratio=None, extra=None):
    rows = []
    for name in names:
        domain, spec = store.get(name), store.spec(name)
        for task in TASKS:
            methods = evaluate_task(field, diffusion, domain, spec, task, args, device)
            primary_method = (
                "standard_noise_ensemble"
                if "standard_noise_ensemble" in methods
                else "anchored_ensemble"
            )
            primary = methods[primary_method]
            row = {
                "regime": regime, "few_shot_ratio": ratio, "dataset": name, "task": task,
                **primary, "primary_method": primary_method, "methods": methods,
            }
            if extra:
                row.update(extra)
            append_result(path, row)
            rows.append(row)
            logger.info("EVALUATION | %s", json.dumps(row))
    return rows


def nested_few_shot_datasets(base: TrainWindowDataset, ratios: Sequence[float], seed: int):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(base)).astype(np.int64)
    output = {}
    for ratio in sorted(ratios):
        budget = min(len(base), max(2, int(round(len(base) * ratio))))
        selected = permutation[:budget]
        train_count = min(budget - 1, max(1, int(round(budget * 0.9))))
        output[ratio] = (base.subset(selected[:train_count]), base.subset(selected[train_count:]), budget)
    return output


def few_shot_adapt(checkpoint, names, ratios, store, args, device, run_dir, path, logger):
    rows = []
    task_config = task_config_from_args(args)
    for domain_index, name in enumerate(names):
        domain, spec = store.get(name), store.spec(name)
        subsets = nested_few_shot_datasets(
            domain.full_train_dataset(effective_max_nodes(domain, args)),
            ratios,
            args.seed + 50021 * domain_index,
        )
        for ratio in ratios:
            train_dataset, val_dataset, budget = subsets[ratio]
            field, diffusion = load_foundation(checkpoint, args, device)
            train_loader = FoundationDomainLoaderView(
                make_loader(
                    train_dataset,
                    effective_train_batch_size(domain, args),
                    True,
                    args.num_workers,
                    args.seed + domain_index,
                ),
                spec,
            )
            val_loader = FoundationDomainLoaderView(
                make_loader(
                    val_dataset,
                    effective_train_batch_size(domain, args),
                    False,
                    args.num_workers,
                    args.seed + 1000 + domain_index,
                ),
                spec,
            )
            optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.few_shot_lr, weight_decay=args.weight_decay)
            best_score, best_state, bad = math.inf, cpu_state_dict(diffusion), 0
            loader_map, val_map = {name: train_loader}, {name: val_loader}
            for epoch in range(1, args.few_shot_epochs + 1):
                metrics = stage2_train_epoch(
                    field, diffusion, loader_map, optimizer, task_config, args, device, epoch, logger,
                    steps_override=args.few_shot_steps_per_epoch or len(train_loader),
                    phase=f"FEW_SHOT[{name}:{ratio:.2f}]",
                )
                if epoch == 1 or epoch == args.few_shot_epochs or epoch % args.validation_interval == 0:
                    val_args = foundation_validation_args(args)
                    score, _ = evaluate_validation(field, diffusion, val_map, task_config, device, val_args, logger)
                    if score < best_score - args.early_stop_min_delta:
                        best_score, best_state, bad = score, cpu_state_dict(diffusion), 0
                    else:
                        bad += args.validation_interval if epoch > 1 else 1
                    logger.info(
                        "FEW_SHOT | dataset=%s ratio=%.2f budget=%d epoch=%d train=%.8f val=%.8f",
                        name, ratio, budget, epoch, metrics["macro_loss"], score,
                    )
                if args.few_shot_patience > 0 and bad >= args.few_shot_patience:
                    break
            diffusion.load_state_dict(best_state, strict=True)
            if args.save_few_shot_checkpoints:
                target = run_dir / "few_shot_checkpoints" / name
                target.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"diffusion": best_state, "dataset": name, "ratio": ratio, "budget": budget, "score": best_score},
                    target / f"ratio_{ratio:.2f}.pt",
                )
            rows.extend(evaluate_group(
                field, diffusion, [name], f"few_shot_{int(round(ratio * 100))}pct",
                store, args, device, path, logger, ratio, {"few_shot_budget": budget},
            ))
            del field, diffusion, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def write_csv(jsonl_path: Path, csv_path: Path) -> list[dict]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = [
        "regime", "few_shot_ratio", "dataset", "task", "primary_method",
        "mae", "mse", "rmse", "count", "few_shot_budget",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return rows


def macro_summary(rows: Sequence[dict]) -> dict:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key in ("mae", "mse", "rmse"):
            grouped[row["regime"]][key].append(float(row[key]))
    return {
        regime: {key: float(np.mean(values)) for key, values in metrics.items()}
        for regime, metrics in grouped.items()
    }


def main() -> None:
    args = parse_args()
    pretrain_names, zero_names, ratios, raw_config = load_experiment_config(args)
    args.resolved_zero_shot_datasets = list(zero_names)
    checkpoint = None
    if args.stage in {"evaluate", "few-shot"}:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        apply_checkpoint_settings(args, checkpoint)
    device = validate_args(args)
    set_global_seed(args.seed)
    run_dir, logger = setup_run(args.log_dir)
    protocol = make_protocol(args)
    audit = audit_dataset_roots(args.data_root, [*pretrain_names, *zero_names])
    pretrain_set = set(pretrain_names)
    for row in audit:
        if not row.get("valid"):
            continue
        total_time, total_nodes = row["values_shape"]
        split = compute_split_indices(total_nodes, total_time, protocol)
        scale = args.pretrain_time_scale if row["dataset"] in pretrain_set else 1.0
        scaled_stop = compute_scaled_train_time_stop(split, protocol, scale)
        row["time_scale_contract"] = {
            "scale": scale,
            "scaled_train_time_slice": [0, scaled_stop],
            "original_train_time_slice": [0, split.train_time_stop],
            "test_time_slice": [split.train_time_stop, split.num_time_steps],
        }
    (run_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    invalid = [row for row in audit if not row.get("valid")]
    if invalid:
        raise RuntimeError(f"Dataset audit failed: {invalid}")
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pretrain_datasets": pretrain_names,
        "zero_shot_datasets": zero_names,
        "few_shot_ratios": ratios,
        "tasks": TASKS,
        "protocol": asdict(protocol),
        "pretrain_scale_contract": {
            "scale": args.pretrain_time_scale,
            "scope": "pretraining_domains_only",
            "selection": "chronological_prefix_of_original_train_interval",
            "normalization": "fit_on_scaled_pretraining_interval",
            "held_out_test_interval": "unchanged",
            "zero_shot_and_few_shot_domains": "unscaled",
        },
        "source_config": raw_config,
        "comparability": {
            "zero_shot_target_updates": 0,
            "few_shot_subsets": "nested_5pct_10pct_with_10pct_adaptation_validation",
        },
    }
    if getattr(args, "model_scale_contract", None) is not None:
        manifest["model_scale_contract"] = args.model_scale_contract
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "PROTOCOL | pretrain=%d zero_shot=%d ratios=%s tasks=%s device=%s",
        len(pretrain_names), len(zero_names), ratios, TASKS, device,
    )
    if args.stage == "audit":
        return
    store = DomainStore(
        args.data_root,
        protocol,
        logger,
        train_time_scales={
            name: args.pretrain_time_scale for name in pretrain_names
        },
    )
    results_path = run_dir / "metrics.jsonl"
    pretraining = None
    if args.stage in {"all", "pretrain"}:
        field, diffusion, checkpoint, pretraining = pretrain_foundation(
            pretrain_names, store, args, device, run_dir, logger
        )
    else:
        assert checkpoint is not None
        field, diffusion = load_foundation(checkpoint, args, device)
    if args.stage1_only:
        summary = {
            "experiment": EXPERIMENT_NAME,
            "pretraining": pretraining,
            "rows": 0,
            "macro": {},
            "checkpoint": str(run_dir / "checkpoints" / "best_stage1.pt"),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        logger.info("STAGE1_ONLY_COMPLETE | %s", json.dumps(summary))
        return
    if args.stage in {"all", "evaluate"}:
        evaluate_group(field, diffusion, pretrain_names, "pretrain_test", store, args, device, results_path, logger)
        evaluate_group(field, diffusion, zero_names, "zero_shot", store, args, device, results_path, logger)
    if args.stage in {"all", "few-shot"}:
        assert checkpoint is not None
        few_shot_adapt(checkpoint, zero_names, ratios, store, args, device, run_dir, results_path, logger)
    rows = write_csv(results_path, run_dir / "metrics.csv") if results_path.exists() else []
    summary = {
        "experiment": EXPERIMENT_NAME,
        "pretraining": pretraining,
        "rows": len(rows),
        "macro": macro_summary(rows),
        "checkpoint": str(args.checkpoint) if args.checkpoint else str(run_dir / "checkpoints" / "best_foundation.pt"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("RUN_COMPLETE | %s", json.dumps(summary))


if __name__ == "__main__":
    main()
