"""Op-level audit of the Anima Block: where does block time actually go?

For one real-size block (D=2048, ctx=1024, seq=1024, bf16, CUDA) measures:
  - forward time per section (modulation x3, SA, CA, MLP)
  - checkpointed backward total vs sum of parts
  - MLP-recompute share (the W2a replay that exact VJP would eliminate)
  - kernel count per section (launch overhead exposure)
  - extra: fused-QKV potential for SA (3 GEMMs -> 1)

No changes to the trainer; measurement only. Output: audit table + JSON.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_trainer.model import Block
from anima_trainer.lora import inject_lora

device = "cuda"
D, CD, NH, HD = 2048, 1024, 16, 128
SEQ = 1024  # 32x32 latent -> 16x16 patches? real run uses 64x64 -> 32x32=1024 tokens
B, T = 1, 1

torch.manual_seed(0)
block = Block(x_dim=D, context_dim=CD, num_heads=NH).to(device, torch.bfloat16)
inject_lora(block, (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj",
    "cross_attn.q_proj", "cross_attn.v_proj",
    "mlp.layer1", "mlp.layer2",
    "adaln_modulation_self_attn.2", "adaln_modulation_cross_attn.2", "adaln_modulation_mlp.2",
), rank=8, alpha=8)

x = torch.randn(B, T, 32, 32, D, device=device, dtype=torch.bfloat16, requires_grad=True)
emb = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
ctx = torch.randn(B, SEQ, CD, device=device, dtype=torch.bfloat16)
adaln_lora = torch.randn(B, T, 3 * D, device=device, dtype=torch.bfloat16)
rope = torch.randn(1024, HD // 2, 2, 2, device=device, dtype=torch.bfloat16)

gy = torch.randn_like(x)


def timed(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


results = {}

# ---------- forward per section ----------
flat = lambda t: t  # shapes already right
shift_sa = scale_sa = gate_sa = torch.randn(B, T, 1, 1, D, device=device, dtype=torch.bfloat16)
shift_ca = scale_ca = gate_ca = torch.randn_like(shift_sa)
shift_mlp = scale_mlp = gate_mlp = torch.randn_like(shift_sa)
from einops import rearrange


def sa_fwd():
    xb = x
    normed = block.layer_norm_self_attn(xb) * (1 + scale_sa) + shift_sa
    normed = rearrange(normed.to(ctx.dtype), "b t h w d -> b (t h w) d")
    out = block.self_attn(normed, rope_emb=rope)
    out = rearrange(out, "b (t h w) d -> b t h w d", t=T, h=32, w=32)
    return torch.addcmul(xb, gate_sa, out.to(xb.dtype))


def ca_fwd():
    xb = x
    normed = block.layer_norm_cross_attn(xb) * (1 + scale_ca) + shift_ca
    normed = rearrange(normed.to(ctx.dtype), "b t h w d -> b (t h w) d")
    out = block.cross_attn(normed, context=ctx)
    out = rearrange(out, "b (t h w) d -> b t h w d", t=T, h=32, w=32)
    return torch.addcmul(xb, gate_ca, out.to(xb.dtype))


def mlp_fwd():
    xb = x
    normed = block.layer_norm_mlp(xb) * (1 + scale_mlp) + shift_mlp
    return block.mlp(normed.to(ctx.dtype))


def mod_fwd():
    (block.adaln_modulation_self_attn(emb) + adaln_lora).chunk(3, dim=-1)
    (block.adaln_modulation_cross_attn(emb) + adaln_lora).chunk(3, dim=-1)
    (block.adaln_modulation_mlp(emb) + adaln_lora).chunk(3, dim=-1)


results["fwd_sa"] = timed(sa_fwd)
results["fwd_ca"] = timed(ca_fwd)
results["fwd_mlp"] = timed(mlp_fwd)
results["fwd_modulation"] = timed(mod_fwd)

total_fwd = results["fwd_sa"] + results["fwd_ca"] + results["fwd_mlp"] + results["fwd_modulation"]
for k in ("fwd_sa", "fwd_ca", "fwd_mlp", "fwd_modulation"):
    results[k + "_pct"] = round(results[k] / total_fwd * 100, 1)

# ---------- checkpointed backward vs eager backward ----------
from torch.utils.checkpoint import checkpoint


def full_fwd_bwd(use_ckpt):
    if x.grad is not None:
        x.grad = None
    block.zero_grad(set_to_none=True)
    def fn(xb):
        return block(xb, emb, ctx, rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln_lora)
    y = checkpoint(fn, x, use_reentrant=False) if use_ckpt else fn(x)
    y.square().mean().backward()


results["bwd_ckpt_total"] = timed(lambda: full_fwd_bwd(True))
results["bwd_eager_total"] = timed(lambda: full_fwd_bwd(False))
results["recompute_overhead"] = round(results["bwd_ckpt_total"] - results["bwd_eager_total"], 2)

# ---------- MLP internals: share of W2 replay ----------
# exact-VJP saving = time of (W2 a) recompute in backward. Measure W2 fwd cost:
with torch.no_grad():
    normed_mlp = (block.layer_norm_mlp(x) * (1 + scale_mlp) + shift_mlp).to(ctx.dtype)
a_act = block.mlp.activation(block.mlp.layer1(normed_mlp))
results["mlp_w2_fwd"] = timed(lambda: block.mlp.layer2(a_act))
results["mlp_w1_fwd"] = timed(lambda: block.mlp.layer1(normed_mlp))
results["mlp_gelu_fwd"] = timed(lambda: block.mlp.activation(a_act))

# backward of MLP branch alone (eager) for scale:
def mlp_bwd():
    block.zero_grad(set_to_none=True)
    out = block.mlp(normed_mlp.detach().requires_grad_(True))
    out.square().mean().backward()

results["mlp_bwd_eager"] = timed(mlp_bwd)

# ---------- SA QKV fusion potential ----------
sa = block.self_attn
seq_flat = rearrange(x, "b t h w d -> b (t h w) d").to(ctx.dtype)


def qkv_separate():
    q = sa.q_proj(seq_flat)
    k = sa.k_proj(seq_flat)
    v = sa.v_proj(seq_flat)
    return q, k, v


w_q = sa.q_proj.base_layer.weight.detach()
w_k = sa.k_proj.base_layer.weight.detach()
w_v = sa.v_proj.base_layer.weight.detach()
w_qkv = torch.cat([w_q, w_k, w_v], dim=0)


def qkv_fused():
    return F.linear(seq_flat, w_qkv)


results["sa_qkv_separate"] = timed(qkv_separate)
results["sa_qkv_fused"] = timed(qkv_fused)
# verify
q1, k1, v1 = qkv_separate()
qkv = qkv_fused()
q2, k2, v2 = qkv.split([sa.q_proj.base_layer.out_features] * 3, dim=-1)
results["sa_qkv_fused_maxdiff"] = float(max((q1 - q2).abs().max(), (k1 - k2).abs().max(), (v1 - v2).abs().max()))

# ---------- kernel counts (launch overhead exposure) ----------
def count_kernels(fn, n=1):
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart() if hasattr(torch.cuda, "cudart") else None
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]):
        fn()
    # counted separately below

from torch.profiler import profile, ProfilerActivity


def kernel_count(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
    return sum(e.count for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA or "cuda" in str(e.device_type).lower())


try:
    results["kernels_sa"] = kernel_count(sa_fwd)
    results["kernels_ca"] = kernel_count(ca_fwd)
    results["kernels_mlp"] = kernel_count(mlp_fwd)
    results["kernels_modulation"] = kernel_count(mod_fwd)
except Exception as e:
    results["kernels_error"] = str(e)[:80]

print(f"{'metric':24} {'value':>10}")
for k, v in results.items():
    if isinstance(v, float):
        print(f"{k:24} {v:10.2f}")
    else:
        print(f"{k:24} {v!s:>10}")

out = ROOT / "runs" / "block_audit.json"
out.write_text(json.dumps(results, indent=1), encoding="utf-8")
print("saved", out)

