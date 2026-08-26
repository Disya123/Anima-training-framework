"""Probe: precompute shared invariants across the anchor-pair forwards.

Invariant per (x_t, sigma) triple: x_embedder output, t_emb/adaln_lora, rope.
Invariant per (context): cross-attn K/V projections of the BASE weights
(no LoRA) — reused across target/anchor/base passes if projections share base.

Two experiments:
  A) forward_latent x3 (target, anchor, base-no-trigger) as the trainer does
  B) prepared-once + KV-cache reuse, same math
Measure time and verify bitwise identity.
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_trainer.config import load_config
from anima_trainer.loader import load_anima_dit

cfg = load_config(ROOT / "configs" / "kor-lili-run.yaml")
device = torch.device("cuda")
model = load_anima_dit(cfg.model.checkpoint, device=device, dtype=torch.bfloat16)
model.llm_adapter.to("cpu")
model.eval()

g = torch.Generator(device="cpu").manual_seed(7)
x = torch.randn(1, 16, 1, 64, 64, generator=g).to(device).bfloat16()
context = torch.randn(1, 128, 1024, generator=g).to(device).bfloat16()
sigma = torch.tensor([0.5], device=device, dtype=torch.float32)


def timed(fn, n=6):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


# --- A: three full forwards (current trainer structure, no anchor yet ---
with torch.inference_mode():
    t_a = timed(lambda: [model.forward_latent(x, sigma, context) for _ in range(3)])
print(f"A) 3x full forward_latent: {t_a:.0f} ms")

# --- B: prepared reuse ---
# what does prepare cost? (exact same sequence as forward_latent)
import torch.nn.functional as F_

with torch.inference_mode():
    def prep():
        x_B_C_1_H_W = x
        B, C, T, H, W = x_B_C_1_H_W.shape
        pad = model.patch_spatial - H % model.patch_spatial if H % model.patch_spatial else 0
        padw = model.patch_spatial - W % model.patch_spatial if W % model.patch_spatial else 0
        xb = F_.pad(x_B_C_1_H_W, (0, padw, 0, pad)) if (pad or padw) else x_B_C_1_H_W
        mask = torch.zeros(B, 1, T, xb.shape[-2], xb.shape[-1], dtype=xb.dtype, device=xb.device)
        xb = torch.cat([xb, mask], dim=1)
        hidden = model.x_embedder(xb)
        t_emb, adaln = model.timestep_embedding(sigma)
        rope = model.prepare_rope(hidden.shape[2], hidden.shape[3], device)
        return hidden, t_emb, adaln, rope

    t_prep = timed(prep)
    hidden, t_emb, adaln, rope = prep()
    print(f"   prepare (pad+mask+embedder+t_emb+rope): {t_prep:.0f} ms ({t_prep/t_a*100:.0f}% of A)")

    def rest3():
        for _ in range(3):
            hh = hidden
            for blk in model.blocks:
                hh = blk(hh, t_emb, context, rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln)
            out = model.final_layer(hh, t_emb, adaln_lora_B_T_3D=adaln)
            model.unpatchify(out)

    t_rest = timed(rest3)
    print(f"B) prepare 1x + 3x blocks-only: {t_prep + t_rest:.0f} ms")

# --- C: cost of cross-attn K/V projections per forward (what caching saves) ---
with torch.inference_mode():
    def kv_only():
        for blk in model.blocks:
            k = blk.cross_attn.k_proj(context)
            v = blk.cross_attn.v_proj(context)
            _ = (k, v)

    t_kv = timed(kv_only)
    print(f"   cross-attn k/v_proj GEMMs (28 blocks, per 'forward'): {t_kv:.0f} ms")

    import einops
    def kv_full_with_norms():
        for blk in model.blocks:
            ca = blk.cross_attn
            k = einops.rearrange(ca.k_proj(context), "b l (h d) -> b h l d", h=ca.n_heads, d=ca.head_dim)
            v = einops.rearrange(ca.v_proj(context), "b l (h d) -> b h l d", h=ca.n_heads, d=ca.head_dim)
            k = ca.k_norm(k)
            _ = (k, v)

    t_kvf = timed(kv_full_with_norms)
    print(f"   k/v_proj + reshape + k_norm (per 'forward'): {t_kvf:.0f} ms")
    print(f"   => over 2 extra anchor-pair forwards saves ~{t_kvf * 2:.0f} ms/step potential")
