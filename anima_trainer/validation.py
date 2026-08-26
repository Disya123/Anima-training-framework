from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from .data import CachedConceptDataset
from .objectives import make_flow_batch


def _stable_seed(run_seed: int, sample_id: str, sigma: float) -> int:
    digest = hashlib.sha256(f"{run_seed}:{sample_id}:{sigma:.8f}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _relative_l2(value: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    numerator = value.float().square().mean().sqrt()
    denominator = reference.float().square().mean().sqrt().clamp_min(eps)
    return float((numerator / denominator).item())


class ValidationRunner:
    """Deterministic held-out flow and no-trigger preservation diagnostics."""

    def __init__(
        self,
        dataset: CachedConceptDataset,
        *,
        fixed_sigmas: tuple[float, ...],
        max_samples: int,
        seed: int,
    ):
        self.dataset = dataset
        self.fixed_sigmas = fixed_sigmas
        self.indices = tuple(range(min(max_samples, len(dataset))))
        self.seed = seed
        self.baseline: dict[str, dict[str, torch.Tensor]] = {}

    @staticmethod
    def _model_device_dtype(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
        parameters = list(model.parameters())
        device_parameter = next((p for p in parameters if p.device.type != "cpu"), parameters[0])
        return device_parameter.device, device_parameter.dtype

    @torch.inference_mode()
    def _predictions(
        self,
        model,
        sample: dict[str, Any],
        sigma_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device, dtype = self._model_device_dtype(model)
        latent = sample["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(
            _stable_seed(self.seed, sample["id"], sigma_value)
        )
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32).to(device)
        sigma = torch.tensor([sigma_value], device=device, dtype=torch.float32)
        flow = make_flow_batch(latent, sigma, noise)
        inputs = flow.noisy_latents.to(dtype).repeat(2, 1, 1, 1, 1)
        sigmas = sigma.repeat(2)
        cond = torch.stack([sample["cond"], sample["cond_no_trigger"]]).to(device=device, dtype=dtype)
        output = model.forward_latent(inputs, sigmas, cond)
        triggered, no_trigger = output.float().chunk(2)
        return triggered.cpu(), no_trigger.cpu(), flow.target_velocity.cpu()

    @torch.inference_mode()
    def capture_baseline(self, model) -> None:
        was_training = model.training
        model.eval()
        self.baseline.clear()
        for index in self.indices:
            sample = self.dataset[index]
            for sigma in self.fixed_sigmas:
                triggered, no_trigger, _ = self._predictions(model, sample, sigma)
                self.baseline[f"{sample['id']}@{sigma:.8f}"] = {
                    "triggered": triggered.to(torch.bfloat16),
                    "no_trigger": no_trigger.to(torch.bfloat16),
                }
        model.train(was_training)

    def save_baseline(self, path: str | Path, *, cache_signature: str | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bucket_signature = None
        if self.dataset.entries:
            bucket_signature = tuple(tuple(e["bucket"]) for e in self.dataset.entries[: len(self.indices)])
        torch.save(
            {
                "fixed_sigmas": self.fixed_sigmas,
                "indices": self.indices,
                "seed": self.seed,
                "bucket_signature": bucket_signature,
                "cache_signature": cache_signature,
                "predictions": self.baseline,
            },
            path,
        )

    def load_baseline(self, path: str | Path, *, cache_signature: str | None = None) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(state["fixed_sigmas"]) != tuple(self.fixed_sigmas) or int(state["seed"]) != self.seed:
            raise ValueError("validation baseline settings do not match current config")
        bucket_signature = None
        if self.dataset.entries:
            bucket_signature = tuple(tuple(e["bucket"]) for e in self.dataset.entries[: len(self.indices)])
        if state.get("bucket_signature") is not None and bucket_signature is not None:
            if tuple(state["bucket_signature"]) != tuple(bucket_signature):
                raise ValueError("validation baseline buckets do not match current cache (stale baseline)")
        if state.get("cache_signature") is not None and cache_signature is not None:
            if state["cache_signature"] != cache_signature:
                raise ValueError("validation baseline cache does not match current cache")
        self.baseline = state["predictions"]

    @torch.inference_mode()
    def evaluate(self, model, *, step: int) -> dict[str, Any]:
        if not self.baseline:
            raise RuntimeError("capture_baseline must run before validation")
        was_training = model.training
        model.eval()
        flow_losses: list[float] = []
        target_drifts: list[float] = []
        preservation_drifts: list[float] = []
        trigger_responses: list[float] = []
        trigger_response_changes: list[float] = []
        per_sample: list[dict[str, Any]] = []
        for index in self.indices:
            sample = self.dataset[index]
            sample_values = {
                "id": sample["id"],
                "flow_loss": [],
                "target_drift": [],
                "preservation_drift": [],
                "trigger_response": [],
            }
            for sigma in self.fixed_sigmas:
                key = f"{sample['id']}@{sigma:.8f}"
                triggered, no_trigger, target = self._predictions(model, sample, sigma)
                base_triggered = self.baseline[key]["triggered"].float()
                base_no_trigger = self.baseline[key]["no_trigger"].float()
                flow_loss = float((triggered - target.float()).square().mean().item())
                target_drift = _relative_l2(triggered - base_triggered, base_triggered)
                preservation_drift = _relative_l2(no_trigger - base_no_trigger, base_no_trigger)
                trigger_response = _relative_l2(triggered - no_trigger, no_trigger)
                base_response = _relative_l2(base_triggered - base_no_trigger, base_no_trigger)
                response_change = trigger_response - base_response
                flow_losses.append(flow_loss)
                target_drifts.append(target_drift)
                preservation_drifts.append(preservation_drift)
                trigger_responses.append(trigger_response)
                trigger_response_changes.append(response_change)
                sample_values["flow_loss"].append(flow_loss)
                sample_values["target_drift"].append(target_drift)
                sample_values["preservation_drift"].append(preservation_drift)
                sample_values["trigger_response"].append(trigger_response)
            per_sample.append(
                {
                    "id": sample_values["id"],
                    **{
                        name: sum(values) / len(values)
                        for name, values in sample_values.items()
                        if name != "id"
                    },
                }
            )
        model.train(was_training)

        def mean(values: list[float]) -> float:
            return sum(values) / max(1, len(values))

        return {
            "step": int(step),
            "samples": len(self.indices),
            "sigmas": list(self.fixed_sigmas),
            "flow_loss": mean(flow_losses),
            "target_drift": mean(target_drifts),
            "preservation_drift_no_trigger": mean(preservation_drifts),
            "trigger_response": mean(trigger_responses),
            "trigger_response_change": mean(trigger_response_changes),
            "per_sample": per_sample,
        }

