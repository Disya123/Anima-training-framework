"""Throughput matrix bench: micro_batch_size x num_workers x loss reduction.

Runs short timed windows on the REAL trainer loop for each combo and prints
ms/optimizer-step. Requires a free GPU (run after the main training finishes).

Usage:
    python scripts/bench_throughput.py --config configs/red-lili-turbo-ae4.yaml --steps 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anima_trainer.config import load_config  # noqa: E402
from anima_trainer.training import AnimaTrainer  # noqa: E402

PROFILE_ROOT = Path(r"E:\Temp\opencode\profiles")


def run_steps(trainer: AnimaTrainer, count: int) -> None:
    config = trainer.config
    trainer.model.train()
    for _ in range(count):
        trainer.optimizer.zero_grad(set_to_none=True)
        for _ in range(config.train.gradient_accumulation):
            trainer._sequential_step(
                trainer.target_loader.next(),
                anchor_weight=config.train.anchor_no_trigger_weight,
                inv_accum=1.0 / config.train.gradient_accumulation,
                prior_batch=None,
                prior_weight=0.0,
            )
        torch.nn.utils.clip_grad_norm_(trainer.trainable, config.train.max_grad_norm)
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.step += 1


def timed(trainer: AnimaTrainer, count: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_steps(trainer, count)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / count * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--combos", default="mbs:1,2;workers:0,2;fastloss:0,1")
    args = ap.parse_args()

    dims: dict[str, list[str]] = {}
    for chunk in args.combos.split(";"):
        key, values = chunk.split(":")
        dims[key.strip()] = [v.strip() for v in values.split(",")]

    keys = list(dims)
    combos = list(itertools.product(*dims.values()))
    free_b, total_b = torch.cuda.mem_get_info()
    print(
        f"device: {torch.cuda.get_device_name(0)} | {len(combos)} combos x ({args.warmup}+{args.steps}) steps | "
        f"free VRAM before: {free_b / 2**30:.2f} / {total_b / 2**30:.2f} GiB"
    )
    if free_b / 2**30 < 4.0:
        print("WARNING: <4 GiB free — mbs=2 combos risk WDDM spillover (desktop apps count against this budget)")

    rows = []
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        cfg = load_config(args.config)
        cfg = replace(
            cfg,
            data=replace(cfg.data, num_workers=int(params.get("workers", 0))),
            train=replace(
                cfg.train,
                micro_batch_size=int(params.get("mbs", cfg.train.micro_batch_size)),
                fast_loss_reduction=bool(int(params.get("fastloss", 0))),
            ),
        )
        tag = "-".join(f"{k}{v}" for k, v in params.items())
        out_dir = PROFILE_ROOT / f"thr-{tag}-{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = replace(cfg, project=replace(cfg.project, output_dir=out_dir))

        t_init = time.perf_counter()
        trainer = AnimaTrainer(cfg)
        run_steps(trainer, args.warmup)
        per_step = timed(trainer, args.steps)
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append({"combo": params, "ms_per_step": per_step, "peak_vram_gib": peak})
        samples_per_step = int(params.get("mbs", cfg.train.micro_batch_size)) * cfg.train.gradient_accumulation
        rows[-1]["ms_per_sample"] = per_step / samples_per_step
        print(f"[{idx + 1}/{len(combos)}] {tag:<28} {per_step:7.0f} ms/step ({per_step / samples_per_step:6.0f} ms/sample) | peak {peak:.2f} GiB | init {time.perf_counter() - t_init:.0f}s")
        del trainer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if rows:
        best = min(rows, key=lambda r: r["ms_per_sample"])
        base = next((r for r in rows if r["combo"].get("mbs") == "1" and r["combo"].get("workers") == "0"), None)
        print("\nsummary (normalized per SAMPLE):")
        for r in rows:
            speed = f"{base['ms_per_sample'] / r['ms_per_sample']:.2f}x" if base else "-"
            print(f"  {json.dumps(r['combo']):<44} {r['ms_per_step']:7.0f} ms/step  {r['ms_per_sample']:6.0f} ms/sample  {speed}")
        print(f"best: {json.dumps(best['combo'])} at {best['ms_per_sample']:.0f} ms/sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
