"""MLP exact-VJP probe: a torch.autograd.Function that computes the SAME
gradients as autograd but skips the W2·a forward replay that checkpoint
performs on recompute. Verified against plain autograd (LoRA-wrapped layers).

Math: y = W2(gelu(W1 x)) (+ LoRA branches per layer)
  given gy:
    ga = W2^T gy            (+ lora2: ga += up2(lor2_down... ) chain)
    gh = gelu'(h) ⊙ ga
    gx = W1^T gh            (+ lora1 chains)
  parameter grads: standard, computed from saved h / a as needed.

Saved tensors: x (already saved by the section closure anyway), h pre-activation.
NOT saved: a, y (recomputed-free backward).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_trainer.model import GPT2FeedForward
from anima_trainer.lora import inject_lora


class MLPExactFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, act, l1_down, l1_up, l1_scale, l2_down, l2_up, l2_scale):
        h = F.linear(x, w1)
        a = act(h)
        y = F.linear(a, w2)
        if l1_down is not None:
            xd = x.to(l1_down.dtype)
            a = a + (F.linear(F.linear(xd, l1_down), l1_up) * l1_scale).to(a.dtype)
        if l2_down is not None:
            ad = a.to(l2_down.dtype)
            y = y + (F.linear(F.linear(ad, l2_down), l2_up) * l2_scale).to(y.dtype)
        ctx.save_for_backward(x, h, a, w1, w2, l1_down, l1_up, l2_down, l2_up)
        ctx.l1_scale = l1_scale
        ctx.l2_scale = l2_scale
        ctx.act = act
        return y

    @staticmethod
    def backward(ctx, gy):
        x, h, a, w1, w2, l1_down, l1_up, l2_down, l2_up = ctx.saved_tensors
        act = ctx.act
        s1, s2 = ctx.l1_scale, ctx.l2_scale
        grads = [None] * 10

        with torch.enable_grad():
            xt = x.detach().requires_grad_(True)
            ht = h.detach().requires_grad_(True)
            at = a.detach().requires_grad_(True)
            # ga = W2^T gy  (+ LoRA2 input path)
            gyd = gy.to(w2.dtype)
            ga = gyd @ w2
            if l2_down is not None:
                gyf = gy.to(l2_up.dtype)
                ga = ga + ((gyf @ l2_up) @ l2_down * s2).to(ga.dtype)
            # gh = act'(h) ⊙ ga
            with torch.no_grad():
                if isinstance(act, torch.nn.GELU):
                    gh = torch.ops.aten.gelu_backward(ga, h, approximate="none")
                else:
                    hh = ht.detach().requires_grad_(True)
                    with torch.enable_grad():
                        aa = act(hh)
                    gh = torch.autograd.grad(aa, hh, ga)[0]
            gx = gh @ w1
            if l1_down is not None:
                ghf = gh.to(l1_up.dtype)
                gx = gx + (((ghf @ l1_up) @ l1_down) * s1).to(gx.dtype)

        # param grads for base weights
        g_w1 = None
        g_w2 = None
        flat_dims = tuple(range(x.dim() - 1))
        gy_f = gy.flatten(0, -2) if flat_dims else gy.unsqueeze(0)
        a_f = a.flatten(0, -2) if flat_dims else a.unsqueeze(0)
        x_f = x.flatten(0, -2) if flat_dims else x.unsqueeze(0)
        h_f = h.flatten(0, -2) if flat_dims else h.unsqueeze(0)
        g_w2 = gy_f.transpose(0, 1) @ a_f
        # gh in flat form for w1
        with torch.no_grad():
            ga_f = gy_f @ w2
            if l2_down is not None:
                gy_l = gy_f.to(l2_up.dtype)
                ga_f = ga_f + ((gy_l @ l2_up) @ l2_down * s2).to(ga_f.dtype)
            if isinstance(act, torch.nn.GELU):
                gh_f = torch.ops.aten.gelu_backward(ga_f, h_f, approximate="none")
            else:
                gh_f = gh.flatten(0, -2)
            g_w1 = gh_f.transpose(0, 1) @ x_f
        grads[1] = g_w1
        grads[2] = g_w2
        if l1_down is not None:
            xd = x_f.to(l1_down.dtype)
            ghd = gh_f.to(l1_up.dtype)
            z1 = xd @ l1_down.transpose(0, 1)            # [N, r]
            gh_l1 = ghd @ l1_up                          # [N, r]
            grads[4] = gh_l1.transpose(0, 1) @ xd * s1   # d/d down1 [r, in]
            grads[5] = ghd.transpose(0, 1) @ z1 * s1     # d/d up1 [out, r]
        if l2_down is not None:
            ad = a_f.to(l2_down.dtype)
            gy_l = gy_f.to(l2_up.dtype) @ l2_up          # [N, r]
            grads[7] = gy_l.transpose(0, 1) @ ad * s2    # d/d down2 [r, Dff]
            z2 = ad @ l2_down.transpose(0, 1)            # [N, r]
            grads[8] = gy_f.to(l2_up.dtype).transpose(0, 1) @ z2 * s2  # d/d up2 [out, r]
        grads[0] = gx
        return tuple(grads)


def run(kind, seed=0):
    torch.manual_seed(seed)
    D, Dff = 2048, 8192
    mlp = GPT2FeedForward(D, Dff).cuda().bfloat16()
    if kind == "lora":
        inject_lora(mlp, ("layer1", "layer2"), rank=8, alpha=8)
        # zero-init up makes dL/dDown exactly 0 in BOTH paths (cos 0/0); use
        # non-zero up for a meaningful comparison
        with torch.no_grad():
            mlp.layer1.lora_up.weight.normal_(0, 0.01)
            mlp.layer2.lora_up.weight.normal_(0, 0.01)
    x = torch.randn(1, 1024, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    params = [p for p in mlp.parameters() if p.requires_grad]

    def fwd():
        if kind == "lora":
            ld1 = mlp.layer1.lora_down.weight
            lu1 = mlp.layer1.lora_up.weight
            ld2 = mlp.layer2.lora_down.weight
            lu2 = mlp.layer2.lora_up.weight
            sc = mlp.layer1.scaling
            return MLPExactFn.apply(x, mlp.layer1.base_layer.weight, mlp.layer2.base_layer.weight,
                                    mlp.activation, ld1, lu1, sc, ld2, lu2, mlp.layer2.scaling)
        return MLPExactFn.apply(x, mlp.layer1.weight, mlp.layer2.weight, mlp.activation, None, None, None, None, None, None)

    y = fwd()
    gy = torch.randn_like(y)
    y.backward(gy)
    gx = x.grad.clone()
    gparams = [p.grad.clone() for p in params]
    x.grad = None
    for p in params:
        p.grad = None

    # reference: plain autograd module call
    y2 = mlp(x)
    y2.backward(gy)
    gx_ref = x.grad.clone()
    gref = [p.grad.clone() for p in params]
    return gx, gparams, gx_ref, gref


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def main():
    for kind in ("plain", "lora"):
        gx, gp, gxr, gref = run(kind)
        print(f"[{kind}] gx cos {cos(gx, gxr):.6f}  rel {float((gx-gxr).norm()/gxr.norm()):.6f}")
        for i, (a, b) in enumerate(zip(gp, gref)):
            print(f"   param{i}: cos {cos(a, b):.6f} rel {float((a-b).norm()/b.norm()):.6f}")


if __name__ == "__main__":
    main()



def bench():
    import time
    from torch.utils.checkpoint import checkpoint
    D, Dff = 2048, 8192
    torch.manual_seed(0)
    mlp = GPT2FeedForward(D, Dff).cuda().bfloat16()
    inject_lora(mlp, ("layer1", "layer2"), rank=8, alpha=8)
    with torch.no_grad():
        mlp.layer1.lora_up.weight.normal_(0, 0.01)
        mlp.layer2.lora_up.weight.normal_(0, 0.01)
    x = torch.randn(1, 1024, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def custom():
        if x.grad is not None:
            x.grad = None
        mlp.zero_grad(set_to_none=True)
        ld1, lu1 = mlp.layer1.lora_down.weight, mlp.layer1.lora_up.weight
        ld2, lu2 = mlp.layer2.lora_down.weight, mlp.layer2.lora_up.weight
        y = MLPExactFn.apply(x, mlp.layer1.base_layer.weight, mlp.layer2.base_layer.weight,
                             mlp.activation, ld1, lu1, mlp.layer1.scaling, ld2, lu2, mlp.layer2.scaling)
        y.square().mean().backward()

    def vanilla():
        if x.grad is not None:
            x.grad = None
        mlp.zero_grad(set_to_none=True)
        def fn(xx):
            return mlp(xx)
        y = checkpoint(fn, x, use_reentrant=False)
        y.square().mean().backward()

    for f, name in ((vanilla, "ckpt-vanilla"), (custom, "exact-vjp")):
        for _ in range(5):
            f()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            f()
        torch.cuda.synchronize()
        print(f"{name:14} {(time.perf_counter()-t0)/50*1000:.2f} ms")


bench()
