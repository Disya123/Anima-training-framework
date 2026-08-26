"""Gradient-fidelity oracle v2.

Measures, for a transport policy vs exact backward:
  - per-module parameter gradient cosine (grouped by block x section)
  - per-block-boundary adjoint cosine cos(g_x_i, g_x_i^exact)
  - time / peak VRAM

Horizon sweep: grow the local tail block by block and find the measured
fidelity horizon.

Usage:
  python verify_policy.py --policy mlp                    # single policy
  python verify_policy.py --horizon                       # tail sweep
  python verify_policy.py --sanity                        # exact vs exact
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_trainer.config import load_config
from anima_trainer.loader import load_anima_dit
from anima_trainer.lora import named_lora_modules
from anima_trainer.objectives import make_flow_batch
from anima_trainer.data import CachedConceptDataset
from anima_trainer.training import make_loader, seed_everything
from anima_trainer.scopes import apply_train_scope
from anima_trainer.training import AnimaTrainer


def section_of(name: str) -> str:
    for s in ("self_attn", "cross_attn", "mlp", "modulation"):
        if s in name:
            return s
    return "other"


def block_of(name: str):
    parts = name.split(".")
    return int(parts[1]) if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit() else -1


def grads_for(probe, batches, patterns, seed=1234, boundary=False):
    from anima_trainer.model import TransportPolicy
    probe._transport = TransportPolicy(patterns) if patterns else frozenset()
    for p in probe.trainable:
        p.grad = None
    seed_everything(seed)
    hooks: list = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for batch in batches:
        cond, _, weights, sigmas, flow = probe._build_step(batch)
        pred = probe._velocity_forward(flow.noisy_latents, sigmas, cond, boundary_hooks=hooks if boundary else None)
        (pred - flow.target_velocity).square().mean().backward()
    torch.cuda.synchronize()
    secs = (time.time() - t0) / max(len(batches), 1)
    peak = torch.cuda.max_memory_allocated() / 2**30
    snap = {}
    for name, m in named_lora_modules(probe.model):
        gd, gu = m.lora_down.weight.grad, m.lora_up.weight.grad
        if gd is None or gu is None:
            continue
        snap[name] = torch.cat([gd.detach().clone().flatten(), gu.detach().clone().flatten()])
    return snap, hooks, secs, peak


def compare(g_exact, g_pol):
    per = {}
    for name, a in g_exact.items():
        b = g_pol.get(name)
        if b is None:
            continue
        cos = float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0))
        key = (block_of(name), section_of(name))
        per.setdefault(key, []).append(cos)
    agg = {k: sum(v) / len(v) for k, v in per.items()}
    total = sum(float(torch.nn.functional.cosine_similarity(a.float(), g_pol[n].float(), dim=0)) for n, a in g_exact.items() if n in g_pol) / max(1, len(g_exact))
    return agg, total


def fmt_boundary(h_exact, h_pol):
    rows = []
    for i, (a, b) in enumerate(zip(h_exact, h_pol)):
        cos = float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))
        rows.append((i, cos))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "kor-lili-run.yaml")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--policy", type=str, default="")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--horizon", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda")
    model = load_anima_dit(cfg.model.checkpoint, device=device, dtype=torch.bfloat16)
    model.llm_adapter.to("cpu")
    apply_train_scope(model, cfg.train.scope)
    from torch.utils.checkpoint import checkpoint as _ckpt

    probe = object.__new__(AnimaTrainer)
    probe.device = device
    probe.model = model
    probe.model_dtype = torch.bfloat16
    probe.config = cfg
    probe.trainable = [p for p in model.parameters() if p.requires_grad]
    probe._ckpt_fn = _ckpt  # exact math either way; keeps the graph inside VRAM
    probe._ckpt_comps = None
    probe._ckpt_group = 1
    probe._ckpt_ctx = None
    probe.step = 0

    def _autocast():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    probe._autocast = _autocast

    ds = CachedConceptDataset(cfg.data.cache_dir, ("train",))
    loader = make_loader(ds, batch_size=1, workers=0, shuffle=False, seed=0)
    batches = []
    for b in loader:
        batches.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()})
        if len(batches) >= args.batches:
            break

    g_exact, h_exact, secs_e, peak_e = grads_for(probe, batches, [], boundary=True)

    if args.sanity:
        g_pol, h_pol, secs_p, peak_p = grads_for(probe, batches, [], boundary=True)
        agg, total = compare(g_exact, g_pol)
        rows = fmt_boundary(h_exact, h_pol)
        print(f"SANITY total cos {total:.5f}; worst boundary {min(c for _, c in rows):.5f}")
        return

    if args.horizon:
        sweep = [
            ["blocks.27.mlp"],
            ["blocks.26-27.mlp"],
            ["blocks.25-27.mlp"],
            ["blocks.24-27.mlp"],
            ["blocks.22-27.mlp"],
            ["blocks.23-27.self_attn"],
            ["blocks.23-27.cross_attn"],
            ["mlp"],
        ]
        results = {}
        print(f"{'policy':26} {'total_cos':>9} {'bnd_min':>8} {'@blk':>5} {'s/b':>6} {'GiB':>5}")
        for pol in sweep:
            g_pol, h_pol, secs_p, peak_p = grads_for(probe, batches, pol, boundary=True)
            agg, total = compare(g_exact, g_pol)
            rows = fmt_boundary(h_exact, h_pol)
            worst_i, worst_c = min(rows, key=lambda r: r[1])
            name = ",".join(pol)
            print(f"{name:26} {total:9.5f} {worst_c:8.5f} {worst_i:5d} {secs_p:6.2f} {peak_p:5.2f}")
            results[name] = {
                "total_cos": round(total, 5),
                "boundary_min": round(worst_c, 5),
                "boundary_argmin": worst_i,
                "s_per_batch": round(secs_p, 2),
                "peak_gib": round(peak_p, 2),
                "per_block_section_cos": {f"b{k[0]}.{k[1]}": round(v, 5) for k, v in sorted(agg.items()) if k[0] >= 20},
            }
        out = ROOT / "runs" / "policy_horizon.json"
        out.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print("saved", out)
        return

    patterns = [s.strip() for s in args.policy.split(",") if s.strip()]
    if not patterns:
        patterns = [k for k, v in cfg.train.gradient_transport.items() if v == "local"]
    g_pol, h_pol, secs_p, peak_p = grads_for(probe, batches, patterns, boundary=True)
    agg, total = compare(g_exact, g_pol)
    rows = fmt_boundary(h_exact, h_pol)
    print(f"policy: {patterns}")
    print(f"{'blk.sec':22} {'cos':>9}")
    for k, v in sorted(agg.items()):
        if k[0] >= 0:
            print(f"b{k[0]:02d}.{k[1]:12} {v:9.5f}")
    worst_i, worst_c = min(rows, key=lambda r: r[1])
    print(f"\ntotal {total:.5f}  adjoint worst b{worst_i} {worst_c:.5f}  time {secs_e:.2f}->{secs_p:.2f}  peak {peak_e:.2f}->{peak_p:.2f}")


if __name__ == "__main__":
    main()
