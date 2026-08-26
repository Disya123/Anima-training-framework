"""Fast probes: (1) which SDPA backend actually runs on this machine,
(2) SAC API availability in this torch build, (3) SAC gradient-equivalence
on a toy block, (4) SAC op-capture: what ops the policy_fn actually sees.
"""
import sys
import torch

print(f"torch {torch.__version__}, cuda {torch.version.cuda}")

# --- 1. SDPA backends ---
from torch.nn.attention import SDPBackend, sdpa_kernel

q = torch.randn(1, 16, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH", "CUDNN_ATTENTION"):
    try:
        backend = getattr(SDPBackend, name)
        with sdpa_kernel(backend):
            out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            out.sum().backward()
        print(f"SDPA {name}: OK")
    except Exception as e:
        print(f"SDPA {name}: FAIL {str(e)[:90]}")
q.grad = None
out_default = torch.nn.functional.scaled_dot_product_attention(q, q, q)
print("default backend ran OK")

# --- 2. SAC API ---
try:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
    print("SAC API: present", [p for p in dir(CheckpointPolicy) if not p.startswith("_")])
except Exception as e:
    print("SAC API: MISSING", e)
    sys.exit(0)

# --- 3. SAC gradient equivalence + op capture on a toy MLP+SDPA block ---
seen_ops: list[str] = []


def sac_context_fn():
    def policy_fn(ctx, func, *args, **kwargs):
        try:
            name = str(func)
        except Exception:
            name = "?"
        if len(seen_ops) < 40:
            seen_ops.append(name)
        if any(k in name for k in ("scaled_dot_product", "flash", "efficient", "cudnn_attention")):
            return CheckpointPolicy.MUST_SAVE
        if any(k in name for k in ("mm", "addmm", "bmm", "gelu")):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(128, 512, bias=False)
        self.w2 = torch.nn.Linear(512, 128, bias=False)
        self.wq = torch.nn.Linear(128, 128, bias=False)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.w1(x))
        m = self.w2(h)
        qkv = self.wq(x).view(1, 256, 16, 8).transpose(1, 2)
        a = torch.nn.functional.scaled_dot_product_attention(qkv, qkv, qkv)
        a = a.transpose(1, 2).reshape(1, 256, 128)
        return m + a


def run(use_sac):
    torch.manual_seed(0)
    blk = ToyBlock().cuda().bfloat16()
    x = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    if use_sac:
        y = torch.utils.checkpoint.checkpoint(blk, x, context_fn=sac_context_fn, use_reentrant=False)
    else:
        y = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
    y.square().mean().backward()
    return (x.grad.clone(), blk.w1.weight.grad.clone(), blk.w2.weight.grad.clone(), blk.wq.weight.grad.clone())


g_eager = run(use_sac=False)
g_sac = run(use_sac=True)
for i, name in enumerate(("x", "w1", "w2", "wq")):
    a, b = g_eager[i], g_sac[i]
    cos = float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))
    print(f"SAC grad {name}: cos={cos:.6f}")

# memory + speed on bigger tensor
import time

def timed(use_sac, D=2048, T=4096):
    class BigBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Linear(D, 4 * D, bias=False).bfloat16()
            self.w2 = torch.nn.Linear(4 * D, D, bias=False).bfloat16()
        def forward(self, x):
            return self.w2(torch.nn.functional.gelu(self.w1(x)))
    torch.manual_seed(0)
    blk = BigBlock().cuda()
    x = torch.randn(1, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    opt = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        if x.grad is not None:
            x.grad = None
        blk.w1.weight.grad = None
        blk.w2.weight.grad = None
        if use_sac:
            y = torch.utils.checkpoint.checkpoint(blk, x, context_fn=sac_context_fn, use_reentrant=False)
        else:
            y = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        y.square().mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        if x.grad is not None:
            x.grad = None
        blk.w1.weight.grad = None
        blk.w2.weight.grad = None
        if use_sac:
            y = torch.utils.checkpoint.checkpoint(blk, x, context_fn=sac_context_fn, use_reentrant=False)
        else:
            y = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        y.square().mean().backward()
    torch.cuda.synchronize()
    return (time.time() - t0) / 10, torch.cuda.max_memory_allocated() / 2**30

t_v, m_v = timed(False)
t_s, m_s = timed(True)
print(f"MLP-only big: vanilla {t_v*1000:.0f}ms {m_v:.2f}GiB | SAC(save mm+gelu) {t_s*1000:.0f}ms {m_s:.2f}GiB")
print("\nops seen (first 40):")
for op in seen_ops:
    print("  ", op)
