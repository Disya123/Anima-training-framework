from __future__ import annotations

import contextlib
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from .checkpoints import export_full_model, export_lora, load_training_checkpoint, save_training_checkpoint
from .concepts import policy_for
from .config import TrainerConfig, torch_dtype
from .data import BucketBatchSampler, CachedConceptDataset, collate_cached
from .lora import lora_disabled
from .loader import load_anima_dit
from .objectives import bell_timestep_weights, make_flow_batch, sample_sigmas, weighted_flow_mse
from .scopes import ScopeReport, apply_train_scope
from .validation import ValidationRunner


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EndlessLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator: Iterator[dict[str, Any]] = iter(loader)

    def next(self) -> dict[str, Any]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def make_loader(
    dataset: CachedConceptDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    sampler = BucketBatchSampler(
        dataset,
        batch_size,
        shuffle=shuffle,
        drop_last=False,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        collate_fn=collate_cached,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def build_optimizer(config: TrainerConfig, named_parameters):
    from .scopes import component_of

    base_lr = config.train.learning_rate or policy_for(config.concept.mode).default_learning_rate
    overrides = dict(config.train.scope.lr_overrides) if hasattr(config.train.scope, "lr_overrides") else {}
    kwargs = dict(
        betas=config.train.betas,
        eps=config.train.eps,
        weight_decay=config.train.weight_decay,
    )
    if overrides:
        buckets: dict[float, list[torch.nn.Parameter]] = {}
        for name, parameter in named_parameters:
            lr = overrides.get(component_of(name) or "", base_lr)
            buckets.setdefault(lr, []).append(parameter)
        param_groups = [{"params": params, "lr": lr} for lr, params in sorted(buckets.items())]
        optimizer_params = param_groups
    else:
        kwargs["lr"] = base_lr
        optimizer_params = [parameter for _, parameter in named_parameters]
    if config.train.optimizer == "adamw":
        return torch.optim.AdamW(optimizer_params, **kwargs)
    try:
        import bitsandbytes as bnb
    except ImportError as error:
        raise RuntimeError(
            "train.optimizer=adamw8bit requires bitsandbytes; install the `[eightbit]` extra "
            "or set train.optimizer=adamw"
        ) from error
    return bnb.optim.AdamW8bit(optimizer_params, **kwargs)


def build_scheduler(config: TrainerConfig, optimizer: torch.optim.Optimizer):
    warmup = config.train.warmup_steps
    total = config.train.steps

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        if config.train.lr_scheduler == "constant":
            return 1.0
        if config.train.lr_scheduler == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class AnimaTrainer:
    def _maybe_apply_quantization(self, config: TrainerConfig) -> dict | None:
        quant_mode = getattr(config.model, "quantization", None)
        if not quant_mode:
            return None
        from .quantization import CONVROT_QUANT_TYPES, apply_convrot

        if quant_mode not in CONVROT_QUANT_TYPES:
            raise ValueError(f"model.quantization must be one of {sorted(CONVROT_QUANT_TYPES)}")
        scope = config.train.scope
        below_block = None
        if config.model.quantize_extent == "below_trainable":
            blocks = getattr(scope, "blocks", None)
            trainable_blocks = [b for b in (blocks or ()) if b is not None]
            if not trainable_blocks:
                raise ValueError(
                    "quantize_extent=below_trainable requires train.scope.blocks; use extent=all for unscoped runs"
                )
            below_block = min(trainable_blocks)
        return apply_convrot(
            self.model,
            tuple(scope.components),
            extent=config.model.quantize_extent,
            below_block=below_block,
        )

    def __init__(self, config: TrainerConfig, *, resume: str | Path | None = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("native Anima training requires a CUDA GPU")
        self.model_dtype = torch_dtype(config.model.dtype)
        if self.model_dtype == torch.float16:
            raise ValueError("Anima fp16 training is unstable; use model.dtype=bfloat16")
        seed_everything(config.project.seed)

        self.output_dir = config.project.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "artifacts").mkdir(exist_ok=True)
        shutil.copy2(config.source, self.output_dir / "run_config.yaml")
        self.events_path = self.output_dir / "events.jsonl"

        self.model = load_anima_dit(config.model.checkpoint, device=self.device, dtype=self.model_dtype)
        # Conditioning is precomputed after the frozen adapter. Keeping the
        # adapter on GPU during training would waste VRAM and cannot affect loss.
        self.model.llm_adapter.to("cpu")
        self.quant_report = self._maybe_apply_quantization(config)
        self.scope_report: ScopeReport = apply_train_scope(self.model, config.train.scope)
        # checkpointing mode resolution: explicit enum wins; legacy boolean
        # gradient_checkpointing maps to block/off when mode is unset.
        tr = config.train
        mode = getattr(tr, "checkpoint_mode", None)
        if mode is None:
            mode = "block" if getattr(tr, "gradient_checkpointing", True) else "off"
        self._ckpt_fn = None
        self._ckpt_comps = None
        self._ckpt_group = 1
        if mode == "block":
            self._ckpt_fn = checkpoint
        elif mode == "group":
            self._ckpt_group = int(tr.checkpoint_group_size)
            if self._ckpt_group < 2:
                raise ValueError("checkpoint_mode=group requires checkpoint_group_size >= 2")
        elif mode == "selective":
            comps = tuple(tr.checkpoint_components or ())
            if not comps:
                raise ValueError("checkpoint_mode=selective requires non-empty checkpoint_components")
            self._ckpt_comps = comps
        elif mode != "off":
            raise ValueError(f"unknown checkpoint_mode: {mode}")
        # gradient transport policy: raw patterns ("mlp", "blocks.25-27.mlp")
        # resolved per block by model.TransportPolicy.
        self._transport = frozenset(
            p for p, pol in getattr(tr, "gradient_transport", {}).items() if pol == "local"
        )
        # Selective Activation Checkpointing inside whole-block regions:
        # exact gradients, op-level save/recompute policy. sdpa saves the
        # attention kernel output (most expensive recompute, +~0.9 GiB).
        sac = getattr(tr, "checkpoint_sac", None)
        self._ckpt_ctx = None
        if sac is not None and mode == "block":
            from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

            def _sac_policy(ctx, func, *args, **kwargs):
                name = str(func)
                if sac == "sdpa" and "_scaled_dot_product" in name:
                    return CheckpointPolicy.MUST_SAVE
                if sac == "gelu" and ("gelu" in name or "silu" in name):
                    return CheckpointPolicy.MUST_SAVE
                if sac == "mm" and (".mm." in name or "addmm" in name):
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            def _sac_context():
                return create_selective_checkpoint_contexts(_sac_policy)

            self._ckpt_ctx = _sac_context
        self.trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = build_optimizer(config, ((name, p) for name, p in self.model.named_parameters() if p.requires_grad))
        self.scheduler = build_scheduler(config, self.optimizer)

        target_dataset = CachedConceptDataset(config.data.cache_dir, ("train",))
        self.target_loader = EndlessLoader(
            make_loader(
                target_dataset,
                batch_size=config.train.micro_batch_size,
                workers=config.data.num_workers,
                shuffle=True,
                seed=config.project.seed,
            )
        )
        self.prior_loader: EndlessLoader | None = None
        if config.data.prior_manifest is not None and config.data.prior_loss_weight > 0:
            if config.data.prior_cache_dir is None:
                raise ValueError("prior manifest configured without prior cache directory")
            prior_dataset = CachedConceptDataset(config.data.prior_cache_dir, ("train",))
            self.prior_loader = EndlessLoader(
                make_loader(
                    prior_dataset,
                    batch_size=config.train.micro_batch_size,
                    workers=config.data.num_workers,
                    shuffle=True,
                    seed=config.project.seed + 10_000,
                )
            )

        self.validation: ValidationRunner | None = None
        if config.validation.enabled:
            try:
                validation_dataset = CachedConceptDataset(config.data.cache_dir, ("validation",))
            except ValueError:
                validation_dataset = None
            if validation_dataset is not None:
                self.validation = ValidationRunner(
                    validation_dataset,
                    fixed_sigmas=config.validation.fixed_sigmas,
                    max_samples=config.validation.max_samples,
                    seed=config.project.seed + 20_000,
                )
                baseline_path = self.output_dir / "validation_baseline.pt"
                cache_signature = str(config.data.cache_dir.resolve())
                if baseline_path.is_file():
                    try:
                        self.validation.load_baseline(baseline_path, cache_signature=cache_signature)
                    except ValueError:
                        self.validation.capture_baseline(self.model)
                        self.validation.save_baseline(baseline_path, cache_signature=cache_signature)
                else:
                    self.validation.capture_baseline(self.model)
                    self.validation.save_baseline(baseline_path, cache_signature=cache_signature)

        resume_path = Path(resume).resolve() if resume else config.train.resume
        self.step = 0
        if resume_path is not None:
            self.step = load_training_checkpoint(
                resume_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
        self._event(
            "initialized",
            step=self.step,
            scope=config.train.scope.kind,
            trainable_parameters=self.scope_report.trainable_parameters,
            total_parameters=self.scope_report.total_parameters,
            matched=len(self.scope_report.matched_modules),
            prior_enabled=self.prior_loader is not None,
            quantization=config.model.quantization,
            quantize_extent=config.model.quantize_extent if config.model.quantization else None,
            quant_modules=(self.quant_report or {}).get("modules", 0),
            quant_by_component=(self.quant_report or {}).get("by_component", {}),
            validation_enabled=self.validation is not None,
        )

    def _event(self, event: str, **payload: Any) -> None:
        record = {"event": event, "time": time.time(), **payload}
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        if event in {"initialized", "log", "checkpoint", "validation", "finished"}:
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))

    def _autocast(self):
        if self.model_dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.model_dtype)

    def _build_step(self, batch: dict[str, Any]):
        """Shared σ/noise preparation so target and anchor passes see the
        SAME x_t despite running as separate B=1 graphs."""
        latents = batch["latents"].to(self.device, dtype=torch.float32, non_blocking=True)
        cond = batch["cond"].to(self.device, dtype=self.model_dtype, non_blocking=True)
        cond_no_trigger = batch["cond_no_trigger"].to(self.device, dtype=self.model_dtype, non_blocking=True)
        weights = batch["weights"].to(self.device, non_blocking=True)
        sigmas = sample_sigmas(
            latents.shape[0],
            device=self.device,
            method=self.config.train.timestep_sampling,
            sigmoid_scale=self.config.train.sigmoid_scale,
            shift=self.config.train.sigma_shift,
        )
        flow = make_flow_batch(latents, sigmas)
        if self.config.train.timestep_sampling == "weighted":
            weights = weights * bell_timestep_weights(sigmas).to(weights.dtype)
        return cond, cond_no_trigger, weights, sigmas, flow

    def _velocity_forward(self, noisy: torch.Tensor, sigmas: torch.Tensor, cond: torch.Tensor, boundary_hooks=None) -> torch.Tensor:
        with self._autocast():
            return self.model.forward_latent(
                noisy.to(self.model_dtype),
                sigmas,
                cond,
                checkpoint_fn=self._ckpt_fn,
                checkpoint_components=self._ckpt_comps,
                checkpoint_group_size=self._ckpt_group,
                transport=self._transport,
                boundary_hooks=boundary_hooks,
                checkpoint_context_fn=self._ckpt_ctx,
            )

    def _sequential_step(
        self,
        batch: dict[str, Any],
        *,
        anchor_weight: float,
        inv_accum: float,
        prior_batch: dict[str, Any] | None,
        prior_weight: float,
    ) -> dict[str, float]:
        """Forward+backward per loss component; only ONE autograd graph is
        alive at a time. ∇(Σ w_i L_i) == Σ w_i ∇L_i so accumulating into
        .grad per component equals the old joint backward."""
        cond, cond_no_trigger, weights, sigmas, flow = self._build_step(batch)
        noisy = flow.noisy_latents

        prediction = self._velocity_forward(noisy, sigmas, cond)
        target_loss = weighted_flow_mse(prediction, flow.target_velocity, weights)
        if getattr(self.config.train, "debug_sync_checks", False) and not torch.isfinite(target_loss):
            raise FloatingPointError(f"non-finite target loss at step {self.step}: {target_loss.item()}")
        (target_loss * inv_accum).backward()
        out = {"target_loss": float(target_loss.detach().item())}

        anchor_loss = 0.0
        if anchor_weight > 0:
            with torch.no_grad(), lora_disabled(self.model):
                base_pred = self._velocity_forward(noisy, sigmas, cond_no_trigger)
            anchored_pred = self._velocity_forward(noisy, sigmas, cond_no_trigger)
            a_loss = weighted_flow_mse(anchored_pred, base_pred.detach().to(anchored_pred.dtype), weights)
            if getattr(self.config.train, "debug_sync_checks", False) and not torch.isfinite(a_loss):
                raise FloatingPointError(f"non-finite anchor loss at step {self.step}: {a_loss.item()}")
            ((a_loss * anchor_weight) * inv_accum).backward()
            anchor_loss = float(a_loss.detach().item())
        out["anchor_loss"] = anchor_loss

        prior_loss = 0.0
        if prior_batch is not None and prior_weight > 0:
            p_cond, _, p_weights, p_sigmas, p_flow = self._build_step(prior_batch)
            p_prediction = self._velocity_forward(p_flow.noisy_latents, p_sigmas, p_cond)
            pl = weighted_flow_mse(p_prediction, p_flow.target_velocity, p_weights)
            if getattr(self.config.train, "debug_sync_checks", False) and not torch.isfinite(pl):
                raise FloatingPointError(f"non-finite prior loss at step {self.step}: {pl.item()}")
            ((pl * prior_weight) * inv_accum).backward()
            prior_loss = float(pl.detach().item())
        out["prior_loss"] = prior_loss
        return out

    def save(self, *, final: bool = False) -> dict[str, str]:
        stem = f"step-{self.step:06d}"
        checkpoint_path = self.output_dir / "checkpoints" / f"training-{stem}.pt"
        save_training_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            config_path=self.config.source,
            scope_kind=self.config.train.scope.kind,
        )
        artifacts = {"training": str(checkpoint_path)}
        metadata = {
            "anima_trainer_version": "0.1.0",
            "concept_mode": self.config.concept.mode,
            "trigger": self.config.concept.trigger or "",
            "scope": self.config.train.scope.kind,
            "step": str(self.step),
            "base_model": str(self.config.model.checkpoint),
        }
        if self.config.train.scope.kind in {"dit_lora", "selected_blocks_lora"}:
            adapter_path = self.output_dir / "artifacts" / f"adapter-{stem}.safetensors"
            export_lora(self.model, adapter_path, metadata=metadata)
            artifacts["adapter"] = str(adapter_path)
            if final:
                final_path = self.output_dir / "artifacts" / "adapter-final.safetensors"
                export_lora(self.model, final_path, metadata=metadata)
                artifacts["final"] = str(final_path)
        elif final:
            final_path = self.output_dir / "artifacts" / "model-final.safetensors"
            export_full_model(self.model, final_path, metadata=metadata)
            artifacts["final"] = str(final_path)
        self._event("checkpoint", step=self.step, final=final, artifacts=artifacts)
        return artifacts

    def run_validation(self) -> dict[str, Any] | None:
        if self.validation is None:
            return None
        metrics = self.validation.evaluate(self.model, step=self.step)
        self._event("validation", **metrics)
        return metrics

    def train(self) -> None:
        config = self.config
        self.model.train()
        last_log_time = time.time()
        loss_target_sum = 0.0
        loss_prior_sum = 0.0
        loss_anchor_sum = 0.0
        log_count = 0
        while self.step < config.train.steps:
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(config.train.gradient_accumulation):
                anchor_active = config.train.anchor_no_trigger_weight > 0 and (
                    config.train.anchor_every <= 1 or self.step % config.train.anchor_every == 0
                )
                prior_batch = self.prior_loader.next() if self.prior_loader is not None else None
                stats = self._sequential_step(
                    self.target_loader.next(),
                    anchor_weight=config.train.anchor_no_trigger_weight if anchor_active else 0.0,
                    inv_accum=1.0 / config.train.gradient_accumulation,
                    prior_batch=prior_batch,
                    prior_weight=config.data.prior_loss_weight if prior_batch is not None else 0.0,
                )
                loss_target_sum += stats["target_loss"]
                loss_prior_sum += stats["prior_loss"]
                loss_anchor_sum += stats["anchor_loss"]
                log_count += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, config.train.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            if self.step % config.train.log_every == 0:
                now = time.time()
                self._event(
                    "log",
                    step=self.step,
                    target_loss=loss_target_sum / max(1, log_count),
                    prior_loss=loss_prior_sum / max(1, log_count),
                    anchor_loss=loss_anchor_sum / max(1, log_count),
                    grad_norm=float(grad_norm),
                    learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    seconds_per_step=(now - last_log_time) / config.train.log_every,
                    peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                )
                loss_target_sum = 0.0
                loss_prior_sum = 0.0
                loss_anchor_sum = 0.0
                log_count = 0
                last_log_time = now

            if self.step % config.train.save_every == 0:
                self.save()
            if self.validation is not None and self.step % config.validation.every == 0:
                self.run_validation()

        self.save(final=True)
        if self.validation is not None and self.step % config.validation.every != 0:
            self.run_validation()
        self._event("finished", step=self.step)

