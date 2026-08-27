"""Per-step profiler for AnimaTrainer: wall-clock phases + CUDA op breakdown.

Answers "where does step time actually go" for float vs quantized configs:
  - steady-state seconds/optimizer-step (replicates the real train loop incl
    gradient accumulation, clipping and scheduler)
  - anchor-pass cost isolated by re-running steps with anchor_weight=0
  - torch.profiler CUDA-time shares grouped into GEMM / SDPA / elementwise /
    reduction / cast / optimizer / other buckets

Usage:
    python scripts/profile_step.py --config configs/red-lili-turbo-qsafe.yaml --steps 6
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anima_trainer.config import load_config  # noqa: E402
from anima_trainer.training import AnimaTrainer  # noqa: E402


def bucket_of(op_name: str) -> str:
    n = op_name.lower()
    if "_int_mm" in n or "mm" in n or "matmul" in n or "gemm" in n or "linear" in n and "layer" not in n:
        return "GEMM"
    if "scaled_dot_product" in n:
        return "SDPA"
    if "adam" in n:
        return "optimizer"
    if any(k in n for k in ("item", "_local_scalar", "nonzero", "synchronize")):
        return "sync"
    if any(k in n for k in ("_to_copy", "cast", "convert")):
        return "cast"
    if any(k in n for k in ("sum", "mean", "amax", "norm", "max", "softmax")):
        return "reduce"
    if any(k in n for k in ("elementwise", "vectorized", "cat", "copy", "fill", "index", "clamp", "round")):
        return "elementwise"
    return "other"


def run_steps(trainer: AnimaTrainer, count: int, *, anchor_weight: float | None = None) -> None:
    """Replicates the train loop body without logging/checkpoint side effects."""
    config = trainer.config
    trainer.model.train()
    for _ in range(count):
        trainer.optimizer.zero_grad(set_to_none=True)
        weight = (
            config.train.anchor_no_trigger_weight
            if anchor_weight is None
            else anchor_weight
        )
        for _ in range(config.train.gradient_accumulation):
            stats = trainer._sequential_step(
                trainer.target_loader.next(),
                anchor_weight=weight,
                inv_accum=1.0 / config.train.gradient_accumulation,
                prior_batch=None,
                prior_weight=0.0,
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(trainer.trainable, config.train.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite grad norm during profiling")
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.step += 1


def timed_steps(trainer: AnimaTrainer, count: int, label: str, **kw) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_steps(trainer, count, **kw)
    torch.cuda.synchronize()
    per_step = (time.perf_counter() - t0) / count
    print(f"  {label:<28} {per_step * 1000:8.1f} ms/step")
    return per_step


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=6, help="active profiled optimizer-steps")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = load_config(args.config)
    tag = cfg.project.name
    profile_dir = Path(r"E:\Temp\opencode\profiles") / f"{tag}-{int(time.time())}"
    profile_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(cfg, project=replace(cfg.project, output_dir=profile_dir))

    print(f"device: {torch.cuda.get_device_name(0)} | config: {args.config}")
    torch._dynamo.config.cache_size_limit = 256
    t_init = time.perf_counter()
    trainer = AnimaTrainer(cfg)
    print(f"init {time.perf_counter() - t_init:.0f}s | quant={cfg.model.quantization} "
          f"modules={trainer.quant_report['modules'] if trainer.quant_report else 0}")

    print("warmup ...")
    run_steps(trainer, args.warmup)

    results: dict[str, float] = {}
    print("\n== wall-clock phases ==")
    base = timed_steps(trainer, args.steps, "full step (as configured)")
    no_anchor = timed_steps(trainer, args.steps, "anchor disabled", anchor_weight=0.0)
    results["full_step_ms"] = base * 1000
    results["no_anchor_ms"] = no_anchor * 1000
    results["anchor_overhead_ms"] = max(0.0, (base - no_anchor) * 1000)

    print("\n== torch.profiler ==")
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    rows: list[tuple[int, str]] = []
    # no schedule(): profiler object tears itself down at the schedule end and
    # key_averages() then asserts; plain context collects every step instead
    with profile(activities=activities) as prof:
        run_steps(trainer, args.steps)
        for e in prof.key_averages():
            cuda_us = getattr(e, "self_device_time_total", 0) or getattr(e, "cuda_time_total", 0)
            if cuda_us > 0:
                rows.append((cuda_us, e.key, "cuda"))
    if not rows:
        # Windows/kineto: CUPTI device timings unavailable -> host-inclusive
        # op time is a faithful proxy for this synchronous GPU-bound loop
        for e in prof.key_averages():
            cpu_us = getattr(e, "cpu_time_total", 0)
            if cpu_us > 0 and getattr(e, "key", "").startswith(("aten::", "torchvision::", "bnb", "_ConvRot")):
                rows.append((cpu_us, e.key, "cpu-incl"))
    basis = rows[0][2] if rows else "none"
    rows = [(us, name) for us, name, _ in rows]
    total_cuda = sum(us for us, _ in rows)
    results["profiled_steps"] = float(args.steps)
    results["profile_basis"] = basis
    categories: dict[str, int] = {}
    for us, name in rows:
        categories[bucket_of(name)] = categories.get(bucket_of(name), 0) + us

    print(f"\ntotal {basis} op time: {total_cuda / 1e3:.0f} ms over {args.steps} steps "
          f"({total_cuda / 1e3 / args.steps:.0f} ms/step)")
    print("\ncategory shares:")
    for cat, us in sorted(categories.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<12} {us / 1e3:8.0f} ms  {us / total_cuda * 100:5.1f}%")
    print("\ntop-12 kernels:")
    for us, name in sorted(rows, reverse=True)[:12]:
        short = name[:96]
        print(f"  {us / 1e3:8.1f} ms  {us / total_cuda * 100:5.1f}%  {short}")

    gemm_share = categories.get("GEMM", 0) / total_cuda
    best_case = (
        1.0 / (gemm_share / 1.53 + (1 - gemm_share)) if categories.get("GEMM") else None
    )
    if best_case:
        print(f"\nheadroom model: GEMM share={gemm_share * 100:.0f}%; "
              f"if quant GEMMs hit ai-toolkit's 1.53x micro number -> "
              f"theoretical e2e ceiling {best_case:.2f}x")

    if results["anchor_overhead_ms"] > 0:
        anchor_s = results["anchor_overhead_ms"] / 1000
        rest = results["no_anchor_ms"] / 1000
        for every in (2, 3, 4):
            projected = rest + anchor_s / every
            print(f"anchor_every={every:<2} projection: {projected * 1000:6.0f} ms/step "
                  f"(~{base / projected:.2f}x vs current)")

    payload = {
        "config": args.config,
        "steps": args.steps,
        "results": results,
        "categories_ms": {k: v / 1e3 for k, v in categories.items()},
        "total_cuda_ms": total_cuda / 1e3,
        "headroom": best_case,
    }
    out_path = profile_dir / "profile.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # trace exported inside the profiler context
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
