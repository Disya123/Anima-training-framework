from __future__ import annotations

import math

from dataclasses import dataclass

import torch


def shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    if shift <= 0:
        raise ValueError("sigma shift must be positive")
    if shift == 1.0:
        return sigmas
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def sample_sigmas(
    batch_size: int,
    *,
    device: torch.device | str,
    method: str = "logit_normal",
    sigmoid_scale: float = 1.0,
    shift: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if method in {"uniform", "balanced", "style_balanced", "weighted"}:
        # ai-toolkit flowmatch convention: uniform timestep grid 1000..1, sigma = t/1000.
        # No inference shift is applied during training; content_or_style=balanced and
        # timestep_type=weighted/linear all reduce to (weighted-)uniform coverage.
        sigmas = torch.rand(batch_size, device=device, generator=generator)
    elif method == "logit_normal":
        logits = torch.randn(batch_size, device=device, generator=generator) * sigmoid_scale
        sigmas = torch.sigmoid(logits)
    else:
        raise ValueError(f"unsupported timestep sampling: {method}")
    return shift_sigmas(sigmas, shift)


def bell_timestep_weights(sigmas: torch.Tensor) -> torch.Tensor:
    """Per-sample loss weights replicating ai-toolkit's bell-shaped
    mean-normalized timestep weighing (timestep_type=weighted)."""
    t = sigmas.clamp(0, 1) * 1000.0
    y = torch.exp(-2.0 * ((t - 500.0) / 1000.0) ** 2)
    w = y - math.exp(-0.5)
    return w.clamp_min(0.0)


@dataclass(frozen=True)
class FlowBatch:
    noisy_latents: torch.Tensor
    sigmas: torch.Tensor
    target_velocity: torch.Tensor
    noise: torch.Tensor


def make_flow_batch(latents: torch.Tensor, sigmas: torch.Tensor, noise: torch.Tensor | None = None) -> FlowBatch:
    if latents.ndim not in {4, 5}:
        raise ValueError(f"latents must be BCHW or BCTHW, got {tuple(latents.shape)}")
    if sigmas.ndim != 1 or sigmas.shape[0] != latents.shape[0]:
        raise ValueError("sigmas must be a vector with one value per sample")
    noise = torch.randn_like(latents) if noise is None else noise
    if noise.shape != latents.shape:
        raise ValueError("noise and latents must have identical shapes")
    expanded = sigmas.view(sigmas.shape[0], *([1] * (latents.ndim - 1)))
    noisy = (1.0 - expanded) * latents + expanded * noise
    return FlowBatch(noisy, sigmas, noise - latents, noise)


def weighted_flow_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    fast: bool = False,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    reduce_dims = tuple(range(1, prediction.ndim))
    if fast:
        # bf16 elementwise + fp32 accumulation inside the reduction kernel:
        # avoids materializing two fp32 copies of the full latent per call
        diff = prediction - target
        count = 1
        for dim in reduce_dims:
            count *= diff.shape[dim]
        per_sample = diff.square().sum(dim=reduce_dims, dtype=torch.float32) / count
    else:
        per_sample = (prediction.float() - target.float()).square().mean(dim=reduce_dims)
    if weights is None:
        return per_sample.mean()
    weights = weights.to(device=per_sample.device, dtype=per_sample.dtype).flatten()
    if weights.shape != per_sample.shape:
        raise ValueError("weights must contain one scalar per sample")
    if torch.any(weights <= 0):
        raise ValueError("all sample weights must be positive")
    return (per_sample * weights).sum() / weights.sum()

