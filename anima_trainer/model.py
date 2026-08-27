import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from .rope3d import Timesteps


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias=False):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


def rope_apply(x_B_H_L_D, rope_L_F_I_J):
    """Apply 3D RoPE with comfy-kernel split-half pairing.

    The rotation matrix is (L, D/2, 2, 2) with blocks [cos, -sin; sin, cos].
    Production (comfy.quant_ops.ck.rms_rope_split_half and the pure-python
    fallback in ldm/ideogram4) pairs dim i with dim i + D/2 — NOT adjacent
    pairs. Pairing must match or trained deltas do not transfer.
    """
    q = x_B_H_L_D.float()
    half = q.shape[-1] // 2
    cos = rope_L_F_I_J[..., 0, 0].to(q.dtype)  # (L, D/2)
    sin = rope_L_F_I_J[..., 1, 0].to(q.dtype)  # (L, D/2)
    x1, x2 = q[..., :half], q[..., half:]
    out = torch.empty_like(q)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x1 * sin + x2 * cos
    return out


class CosmosAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, n_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self._w_qkv_cache: Optional[torch.Tensor] = None

    def _qkv_fused(self, x):
        """One GEMM for self-attention Q/K/V (same input), split into views.
        Concatenated base weight is cached (device/dtype-keyed); LoRA deltas
        are three small rank-r GEMMs added to their slices. Exact: the base
        GEMM order change only shifts bf16 accumulation rounding."""
        def _unwrap(mod):
            return mod.base_layer if hasattr(mod, "base_layer") else mod

        wrappers = (self.q_proj, self.k_proj, self.v_proj)
        bases = tuple(_unwrap(m) for m in wrappers)
        if any(hasattr(b, "cr_w_t") for b in bases):
            # quantized projections: weight-cat cache cannot represent packed
            # int8 storage; route through wrappers (LoRA deltas included there)
            return tuple(m(x) for m in wrappers)
        trainable_base = any(p.requires_grad for b in bases for p in b.parameters(recurse=True))
        lora_present = any(hasattr(m, "lora_down") for m in wrappers)
        if trainable_base or (lora_present and any(not m.enabled for m in wrappers)):
            # trainable bases invalidate the frozen-weight cache; disabled
            # wrappers must be bypassed so lora_disabled() is honored exactly
            return tuple(m(x) for m in wrappers)
        wq = bases[0].weight
        wk = bases[1].weight
        wv = bases[2].weight
        w_qkv = self._w_qkv_cache
        if w_qkv is None or w_qkv.device != wq.device or w_qkv.dtype != wq.dtype or w_qkv.is_inference():
            w_qkv = torch.cat([wq, wk, wv], dim=0)
            if w_qkv.is_inference():
                # created inside inference (anchor-base pass / recompute): SAC
                # cannot save inference tensors; clone into a normal tensor
                w_qkv = w_qkv.clone()
            self._w_qkv_cache = w_qkv
        out = F.linear(x, w_qkv)
        nq, nk = wq.shape[0], wk.shape[0]
        q, k, v = out.split([nq, nk, wv.shape[0]], dim=-1)
        parts = [q, k, v]
        for idx, proj in enumerate((self.q_proj, self.k_proj, self.v_proj)):
            if hasattr(proj, "lora_down") and proj.enabled:
                # mirror LoRALinear semantics incl. adapter dropout
                adapter_input = proj.dropout(x.to(proj.lora_down.weight.dtype))
                d = F.linear(F.linear(adapter_input, proj.lora_down.weight), proj.lora_up.weight) * proj.scaling
                parts[idx] = parts[idx] + d.to(parts[idx].dtype)
        return tuple(parts)

    def forward(self, x, context=None, rope_emb=None):
        context = x if context is None else context

        if self.is_selfattn:
            q, k, v = self._qkv_fused(x)
            q = rearrange(q, "b l (h d) -> b h l d", h=self.n_heads, d=self.head_dim)
            k = rearrange(k, "b l (h d) -> b h l d", h=self.n_heads, d=self.head_dim)
            v = rearrange(v, "b l (h d) -> b h l d", h=self.n_heads, d=self.head_dim)
        else:
            q = rearrange(self.q_proj(x), "b l (h d) -> b h l d", h=self.n_heads, d=self.head_dim)
            k = rearrange(self.k_proj(context), "b l (h d) -> b h l d", h=self.n_heads, d=self.head_dim)
            v = rearrange(self.v_proj(context), "b l (h d) -> b h l d", h=self.n_heads, d=self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.is_selfattn and rope_emb is not None:
            q = rope_apply(q, rope_emb)
            k = rope_apply(k, rope_emb)

        out = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.output_proj(out)


class LayerNormNoAffine(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(dim, elementwise_affine=False, eps=1e-6)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

    def forward(self, sample):
        emb = self.linear_2(self.activation(self.linear_1(sample)))
        if self.use_adaln_lora:
            return sample, emb
        return emb, None


class PatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size: int, temporal_patch_size: int, in_channels: int, out_channels: int):
        super().__init__()
        from einops.layers.torch import Rearrange

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                out_channels,
                bias=False,
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

    def forward(self, x):
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        return self.proj(x)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, spatial_patch_size: int, temporal_patch_size: int, out_channels: int, adaln_lora_dim: int = 256):
        super().__init__()
        self.layer_norm = LayerNormNoAffine(hidden_size)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, adaln_lora_dim, bias=False),
            nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
        )

    def forward(self, x_B_T_H_W_D, emb_B_T_D, adaln_lora_B_T_3D=None):
        shift_B_T_D, scale_B_T_D = (self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]).chunk(2, dim=-1)
        shift = rearrange(shift_B_T_D, "b t d -> b t 1 1 d")
        scale = rearrange(scale_B_T_D, "b t d -> b t 1 1 d")
        x = self.layer_norm(x_B_T_H_W_D) * (1 + scale) + shift
        return self.linear(x)


class TransportPolicy:
    """Pattern set -> per-block section set.

    Patterns: ``section`` (all blocks), ``blocks.N.section``,
    ``blocks.N-M.section`` where section is one of self_attn/cross_attn/mlp.
    """

    __slots__ = ("entries",)

    def __init__(self, patterns):
        self.entries: list[tuple[int, int, str]] = []
        for p in patterns:
            head, _, section = str(p).rpartition(".")
            if head.startswith("blocks."):
                rng = head[len("blocks."):]
                if "-" in rng:
                    lo, hi = (int(v) for v in rng.split("-", 1))
                else:
                    lo = hi = int(rng)
                self.entries.append((lo, hi, section))
            else:
                self.entries.append((0, 10**9, str(p)))

    def for_block(self, i: int) -> frozenset[str]:
        return frozenset(s for lo, hi, s in self.entries if lo <= i <= hi)


class _SectionCheckpoint:
    """Callable wrapper carrying the per-section checkpoint allowlist."""

    __slots__ = ("checkpoint", "components")

    def __init__(self, components: tuple[str, ...]):
        self.checkpoint = checkpoint
        self.components = set(components)

    def __call__(self, fn, *args):
        return self.checkpoint(fn, *args, use_reentrant=False)


class Block(nn.Module):
    def __init__(self, x_dim: int, context_dim: int, num_heads: int, mlp_ratio: float = 4.0, adaln_lora_dim: int = 256):
        super().__init__()
        head_dim = x_dim // num_heads
        self.layer_norm_self_attn = LayerNormNoAffine(x_dim)
        self.self_attn = CosmosAttention(x_dim, None, num_heads, head_dim)
        self.layer_norm_cross_attn = LayerNormNoAffine(x_dim)
        self.cross_attn = CosmosAttention(x_dim, context_dim, num_heads, head_dim)
        self.layer_norm_mlp = LayerNormNoAffine(x_dim)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        def adaln():
            return nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )

        self.adaln_modulation_self_attn = adaln()
        self.adaln_modulation_cross_attn = adaln()
        self.adaln_modulation_mlp = adaln()

    def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None, ckpt=None, transport=None, mods=None):
        """``ckpt`` selects gradient-checkpoint granularity: None keeps every
        section eager; a callable(fn, *args) memoizes individual sections.
        Which sections are wrapped is decided by ``ckpt.components``.

        ``transport`` is a set of section names running with LOCAL gradient
        transport: the branch input is detached, so autograd skips the branch
        Jacobian for upstream adapters while parameter gradients inside the
        branch (LoRA, modulation) stay exact. The residual identity path
        x + gate*branch still transports gradient exactly. Forward output is
        bit-identical to full transport."""
        flat = lambda t: rearrange(t, "b t d -> b t 1 1 d")
        if mods is not None:
            shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = mods
        else:
            shift_sa, scale_sa, gate_sa = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            shift_ca, scale_ca, gate_ca = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)

        B, T, H, W, D = x_B_T_H_W_D.shape
        local = transport or ()

        def _sa(x, scale_sa, shift_sa, gate_sa, rope_emb):
            xb = x.detach() if "self_attn" in local else x
            normed = self.layer_norm_self_attn(xb) * (1 + flat(scale_sa)) + flat(shift_sa)
            normed = rearrange(normed.to(crossattn_emb.dtype), "b t h w d -> b (t h w) d")
            attn_out = self.self_attn(normed, rope_emb=rope_emb)
            attn_out = rearrange(attn_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
            return torch.addcmul(x, flat(gate_sa), attn_out.to(x.dtype))

        def _ca(x, scale_ca, shift_ca, gate_ca, context):
            xb = x.detach() if "cross_attn" in local else x
            normed = self.layer_norm_cross_attn(xb) * (1 + flat(scale_ca)) + flat(shift_ca)
            normed = rearrange(normed.to(context.dtype), "b t h w d -> b (t h w) d")
            attn_out = self.cross_attn(normed, context=context)
            attn_out = rearrange(attn_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
            return torch.addcmul(x, flat(gate_ca), attn_out.to(x.dtype))

        def _mlp(x, scale_mlp, shift_mlp, gate_mlp):
            xb = x.detach() if "mlp" in local else x
            normed = self.layer_norm_mlp(xb) * (1 + flat(scale_mlp)) + flat(shift_mlp)
            mlp_out = self.mlp(normed.to(crossattn_emb.dtype))
            return torch.addcmul(x, flat(gate_mlp), mlp_out.to(x.dtype))

        sections = (
            (_sa, "self_attn", (scale_sa, shift_sa, gate_sa, rope_emb_L_1_1_D)),
            (_ca, "cross_attn", (scale_ca, shift_ca, gate_ca, crossattn_emb)),
            (_mlp, "mlp", (scale_mlp, shift_mlp, gate_mlp)),
        )
        enabled = getattr(ckpt, "components", None) if ckpt is not None else None
        for fn, name, args in sections:
            if ckpt is not None and (enabled is None or name in enabled):
                x_B_T_H_W_D = ckpt(fn, x_B_T_H_W_D, *args)
            else:
                x_B_T_H_W_D = fn(x_B_T_H_W_D, *args)
        return x_B_T_H_W_D


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class AdapterRotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class AdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        kv_shape = (*context.shape[:-1], self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class AdapterTransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=True):
        super().__init__()
        self.use_self_attn = use_self_attn
        head_dim = model_dim // num_heads

        if use_self_attn:
            self.norm_self_attn = RMSNorm(model_dim, eps=1e-6)
            self.self_attn = AdapterAttention(model_dim, model_dim, num_heads, head_dim)

        self.norm_cross_attn = RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = AdapterAttention(model_dim, source_dim, num_heads, head_dim)
        self.norm_mlp = RMSNorm(model_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim),
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None, position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            attn_out = self.self_attn(
                self.norm_self_attn(x),
                mask=target_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )
            x = x + attn_out

        attn_out = self.cross_attn(
            self.norm_cross_attn(x),
            mask=source_attention_mask,
            context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        x = x + attn_out
        return x + self.mlp(self.norm_mlp(x))


class LLMAdapter(nn.Module):
    def __init__(
        self,
        source_dim=1024,
        target_dim=1024,
        model_dim=1024,
        num_layers=6,
        num_heads=16,
        use_self_attn=True,
    ):
        super().__init__()
        self.embed = nn.Embedding(32128, target_dim)
        if model_dim != target_dim:
            self.in_proj = nn.Linear(target_dim, model_dim)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = AdapterRotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList(
            [AdapterTransformerBlock(source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = RMSNorm(target_dim, eps=1e-6)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)
        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        context = source_hidden_states
        x = self.embed(target_input_ids).to(context.dtype)
        x = self.in_proj(x)
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(
                x,
                context,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )
        return self.norm(self.out_proj(x))


class AnimaDIT(nn.Module):
    """Standalone port of comfy MiniTrainDIT + Anima extension.

    State-dict compatible with circlestone-labs/Anima single-file checkpoints
    (keys carry a leading "net." prefix which is stripped on load).
    """

    def __init__(
        self,
        max_img_h: int = 240,
        max_img_w: int = 240,
        max_frames: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        concat_padding_mask: bool = True,
        model_channels: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        min_fps: int = 1,
        max_fps: int = 30,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
    ):
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.concat_padding_mask = concat_padding_mask
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        from .rope3d import VideoRopePosition3DEmb

        self.pos_embedder = VideoRopePosition3DEmb(
            head_dim=model_channels // num_heads,
            len_h=max_img_h // patch_spatial,
            len_w=max_img_w // patch_spatial,
            len_t=max_frames // patch_temporal,
            h_extrapolation_ratio=rope_h_extrapolation_ratio,
            w_extrapolation_ratio=rope_w_extrapolation_ratio,
            t_extrapolation_ratio=rope_t_extrapolation_ratio,
        )

        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        eff_in = in_channels + (1 if concat_padding_mask else 0)
        self.x_embedder = PatchEmbed(patch_spatial, patch_temporal, eff_in, model_channels)

        self.blocks = nn.ModuleList(
            [
                Block(model_channels, crossattn_emb_channels, num_heads, mlp_ratio, adaln_lora_dim)
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = FinalLayer(model_channels, patch_spatial, patch_temporal, out_channels, adaln_lora_dim)
        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)
        self.llm_adapter = LLMAdapter()

    def prepare_rope(self, H, W, device):
        return self.pos_embedder.generate_embeddings((1, 1, H, W, self.model_channels), fps=None, device=device)

    def compute_modulations(self, emb_B_T_D, adaln_lora_B_T_3D):
        """Precompute per-block AdaLN (shift, scale, gate) triples OUTSIDE the
        checkpointed region. Inputs are tiny (t_emb); outputs are nine small
        tensors per block. Exact: same math, just hoisted; the modulation
        autograd graph stays alive so parameter gradients remain exact."""
        out = []
        for blk in self.blocks:
            mods = []
            for net in (blk.adaln_modulation_self_attn, blk.adaln_modulation_cross_attn, blk.adaln_modulation_mlp):
                mods.extend((net(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1))
            out.append(mods)
        return out

    def timestep_embedding(self, timesteps):
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        emb_in = self.t_embedder[0](timesteps).to(self.t_embedder[1].linear_1.weight.dtype)
        t_emb, adaln_lora = self.t_embedder[1](emb_in)
        return self.t_embedding_norm(t_emb), adaln_lora

    def unpatchify(self, x_B_T_H_W_M):
        return rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )

    def forward_latent(self, x_B_C_1_H_W, timesteps, context, checkpoint_fn=None, checkpoint_components=None, checkpoint_group_size=1, transport=None, boundary_hooks=None, checkpoint_context_fn=None):
        """Original latent-space forward; used for weight verification.

        ``transport`` is an iterable of gradient-transport patterns (see
        TransportPolicy); resolved per block. ``boundary_hooks`` (oracle only):
        when a list, one input-gradient hook per block boundary is registered
        and appended as (block_index, grad) during backward.

        Three mutually exclusive checkpointing modes (requires grad enabled):
        - whole-block: ``checkpoint_fn`` wraps each single block;
        - per-section: ``checkpoint_components`` subset of
          {"self_attn", "cross_attn", "mlp"} (empty set = off);
        - coarse groups: ``checkpoint_group_size`` > 1 wraps consecutive runs
          of that many blocks in ONE checkpoint call (fewer boundaries saved;
          costs more recompute). Captured by the group fn are modules and the
          long-lived locals t_emb/context/rope/adaln_lora only; the carried
          activation is always passed as an explicit argument."""
        if sum((checkpoint_fn is not None, checkpoint_components is not None)) > 1:
            raise ValueError("use only one of checkpoint_fn / checkpoint_components")
        if checkpoint_group_size < 1:
            raise ValueError("checkpoint_group_size must be >= 1")
        if checkpoint_fn is not None or checkpoint_components is not None:
            if checkpoint_group_size != 1:
                raise ValueError("checkpoint_group_size is incompatible with checkpoint_fn/components modes")
        orig_shape = list(x_B_C_1_H_W.shape)
        B, C, T, H, W = x_B_C_1_H_W.shape
        pad = self.patch_spatial - H % self.patch_spatial if H % self.patch_spatial else 0
        padw = self.patch_spatial - W % self.patch_spatial if W % self.patch_spatial else 0
        if pad or padw:
            x_B_C_1_H_W = F.pad(x_B_C_1_H_W, (0, padw, 0, pad))

        if self.concat_padding_mask:
            mask = torch.zeros(B, 1, T, x_B_C_1_H_W.shape[-2], x_B_C_1_H_W.shape[-1], dtype=x_B_C_1_H_W.dtype, device=x_B_C_1_H_W.device)
            x_B_C_1_H_W = torch.cat([x_B_C_1_H_W, mask], dim=1)

        x_B_T_H_W_D = self.x_embedder(x_B_C_1_H_W)
        t_emb, adaln_lora = self.timestep_embedding(timesteps)
        rope = self.prepare_rope(x_B_T_H_W_D.shape[2], x_B_T_H_W_D.shape[3], x_B_T_H_W_D.device)
        # hoist all AdaLN modulation nets out of the per-block path: tiny
        # inputs, exact gradients, and the checkpoint replay stops re-running
        # 84 small GEMMs per pass
        all_mods = self.compute_modulations(t_emb, adaln_lora)

        n_blocks = len(self.blocks)
        policy = transport if isinstance(transport, TransportPolicy) else (TransportPolicy(transport) if transport else None)
        if checkpoint_components is not None and checkpoint_fn is None and torch.is_grad_enabled() and len(checkpoint_components) > 0:
            section_ckpt = _SectionCheckpoint(tuple(checkpoint_components))
            for bi, blk in enumerate(self.blocks):
                t_i = policy.for_block(bi) if policy else None
                x_B_T_H_W_D = blk(x_B_T_H_W_D, t_emb, context, rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln_lora, ckpt=section_ckpt, transport=t_i, mods=all_mods[bi])
                if boundary_hooks is not None:
                    x_B_T_H_W_D.register_hook(lambda g: boundary_hooks.append(g.detach().clone()))
        elif checkpoint_group_size > 1 and torch.is_grad_enabled():
            for start in range(0, n_blocks, checkpoint_group_size):
                end = min(start + checkpoint_group_size, n_blocks)
                group = [self.blocks[i] for i in range(start, end)]  # modules only

                def run_group(xi, _group=tuple(group), _start=start, _mods=tuple(all_mods[start:end])):
                    for k, gb in enumerate(_group):
                        t_i = policy.for_block(_start + k) if policy else None
                        xi = gb(xi, t_emb, context, rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln_lora, transport=t_i, mods=_mods[k])
                    return xi

                x_B_T_H_W_D = checkpoint(run_group, x_B_T_H_W_D, use_reentrant=False)
                if boundary_hooks is not None:
                    x_B_T_H_W_D.register_hook(lambda g: boundary_hooks.append(g.detach().clone()))
        elif checkpoint_fn is not None and torch.is_grad_enabled():
            for bi, blk in enumerate(self.blocks):
                t_i = policy.for_block(bi) if policy else None
                kwargs_ckpt = {"use_reentrant": False}
                if checkpoint_context_fn is not None:
                    kwargs_ckpt["context_fn"] = checkpoint_context_fn
                x_B_T_H_W_D = checkpoint_fn(
                    lambda b, xi, _t=t_i, _m=all_mods[bi]: b(xi, t_emb, context, rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln_lora, transport=_t, mods=_m),
                    blk,
                    x_B_T_H_W_D,
                    **kwargs_ckpt,
                )
                if boundary_hooks is not None:
                    x_B_T_H_W_D.register_hook(lambda g: boundary_hooks.append(g.detach().clone()))
        else:
            for bi, blk in enumerate(self.blocks):
                t_i = policy.for_block(bi) if policy else None
                x_B_T_H_W_D = blk(x_B_T_H_W_D, t_emb, context, rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln_lora, transport=t_i, mods=all_mods[bi])
                if boundary_hooks is not None:
                    x_B_T_H_W_D.register_hook(lambda g: boundary_hooks.append(g.detach().clone()))

        out = self.final_layer(x_B_T_H_W_D, t_emb, adaln_lora_B_T_3D=adaln_lora)
        return self.unpatchify(out)[..., : orig_shape[-2], : orig_shape[-1]]


def build_anima_dit(**overrides) -> AnimaDIT:
    cfg = dict(
        max_img_h=240,
        max_img_w=240,
        max_frames=128,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=True,
        model_channels=2048,
        num_blocks=28,
        num_heads=16,
        crossattn_emb_channels=1024,
        # comfy model_detection for anima (in_channels==16, model_channels==2048):
        # NTK rope extrapolation ratios must match production or the positional
        # geometry diverges and trained deltas do not transfer.
        rope_h_extrapolation_ratio=4.0,
        rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0,
    )
    cfg.update(overrides)
    return AnimaDIT(**cfg)
