"""SAC per-block experiment (fixed structure: 28 whole-block regions kept,
selective policy INSIDE each block). All variants are exact-gradient.

  block    : vanilla whole-block checkpoint
  sac_sdpa : MUST_SAVE SDPA inside each block (kills the most expensive recompute)
  sac_mm   : MUST_SAVE mm only (projections; not MLP-expansion monsters? they are mm too)
  sac_gelu : MUST_SAVE gelu/silu only (cheap saves, big recompute win)
"""
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint, CheckpointPolicy, create_selective_checkpoint_contexts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_trainer.config import load_config
from anima_trainer.loader import load_anima_dit
from anima_trainer.data import CachedConceptDataset
from anima_trainer.training import make_loader, seed_everything
from anima_trainer.scopes import apply_train_scope
from anima_trainer.lora import named_lora_modules
from anima_trainer.training import AnimaTrainer


def make_context_fn(kind):
    def policy_fn(ctx, func, *args, **kwargs):
        name = str(func)
        if "_scaled_dot_product" in name and kind == "sac_sdpa":
            return CheckpointPolicy.MUST_SAVE
        if kind == "sac_mm" and (".mm." in name or "addmm" in name):
            return CheckpointPolicy.MUST_SAVE
        if kind == "sac_gelu" and ("gelu" in name or "silu" in name):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    def context_fn():
        return create_selective_checkpoint_contexts(policy_fn)

    return context_fn


def main():
    cfg = load_config(ROOT / "configs" / "kor-lili-run.yaml")
    device = torch.device("cuda")
    model = load_anima_dit(cfg.model.checkpoint, device=device, dtype=torch.bfloat16)
    model.llm_adapter.to("cpu")
    apply_train_scope(model, cfg.train.scope)

    probe = object.__new__(AnimaTrainer)
    probe.device = device
    probe.model = model
    probe.model_dtype = torch.bfloat16
    probe.config = cfg
    probe.trainable = [p for p in model.parameters() if p.requires_grad]
    probe._ckpt_fn = checkpoint
    probe._ckpt_comps = None
    probe._ckpt_group = 1
    probe._transport = frozenset()
    probe.step = 0

    def _autocast():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    probe._autocast = _autocast

    ds = CachedConceptDataset(cfg.data.cache_dir, ("train",))
    loader = make_loader(ds, batch_size=1, workers=0, shuffle=False, seed=0)
    batches = []
    for b in loader:
        batches.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()})
        if len(batches) >= 3:
            break

    def run_once(kind):
        probe._ctx_fn = make_context_fn(kind) if kind != "block" else None
        for batch in batches:
            cond, _, w, sigmas, flow = probe._build_step(batch)
            pred = probe._velocity_forward(flow.noisy_latents, sigmas, cond)
            (pred - flow.target_velocity).square().mean().backward()
            for p in probe.trainable:
                p.grad = None

    results = {}
    grads_ref = None
    for kind in ("block", "sac_sdpa", "sac_gelu", "sac_mm"):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # patch forward to carry context_fn
            orig_fwd = AnimaTrainer._velocity_forward

            def fwd(self, noisy, sigmas, cond, _kind=kind, _orig=orig_fwd):
                self._ckpt_ctx = make_context_fn(_kind) if _kind != "block" else None
                with self._autocast():
                    return self.model.forward_latent(
                        noisy.to(self.model_dtype),
                        sigmas,
                        cond,
                        checkpoint_fn=self._ckpt_fn,
                        checkpoint_components=self._ckpt_comps,
                        checkpoint_group_size=self._ckpt_group,
                        transport=self._transport,
                        checkpoint_context_fn=self._ckpt_ctx,
                    )

            AnimaTrainer._velocity_forward = fwd
            seed_everything(999)
            run_once(kind)  # warmup
            seed_everything(999)
            t0 = time.time()
            n = 0
            for batch in batches:
                cond, _, w, sigmas, flow = probe._build_step(batch)
                pred = probe._velocity_forward(flow.noisy_latents, sigmas, cond)
                (pred - flow.target_velocity).square().mean().backward()
                n += 1
            torch.cuda.synchronize()
            secs = (time.time() - t0) / n
            peak = torch.cuda.max_memory_allocated() / 2**30
            grads = {}
            for name, m in named_lora_modules(model):
                grads[name] = torch.cat([m.lora_down.weight.grad.flatten(), m.lora_up.weight.grad.flatten()]).float().clone()
            if grads_ref is None:
                cos = 1.0
                grads_ref = grads
            else:
                cos = sum(float(torch.nn.functional.cosine_similarity(grads[nm], grads_ref[nm], dim=0)) for nm in grads) / len(grads)
            results[kind] = {"s_per_batch": round(secs, 2), "peak_gib": round(peak, 2), "grad_cos": round(cos, 5)}
            print(f"{kind:10} {secs:6.2f} s/batch  {peak:5.2f} GiB  grad_cos {cos:.5f}", flush=True)
        except torch.OutOfMemoryError as e:
            results[kind] = {"error": "OOM"}
            print(f"{kind:10} OOM", flush=True)
        finally:
            AnimaTrainer._velocity_forward = orig_fwd

    out = ROOT / "runs" / "sac_bench.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
