"""Standalone Qwen Image / Wan 2.1 VAE encoder.

Ported from comfy/ldm/wan/vae.py, which carries the original
Apache-2.0 Wan-Video/Wan2.1 implementation. Only the encode path is
provided (trainer v0.1 caches latents once); the streaming feat_cache
machinery is dropped because single-image encoding never populates it,
and the causal convolutions fall back to explicit zero padding which is
mathematically identical to the weight-truncation fast path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file
from torch import nn


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x):
        if self._padding > 0:
            shape = list(x.shape)
            shape[2] = self._padding
            pad = torch.zeros(shape, device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=2)
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        normalized = F.normalize(x, dim=(1 if self.channel_first else -1))
        result = normalized * self.scale * self.gamma.to(x)
        if self.bias is not None:
            result = result + self.bias.to(x)
        return result


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Causal-free single-head spatial self-attention."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        bt = q.shape[0]
        q = q.view(bt, c, -1).transpose(1, 2).unsqueeze(1)
        k = k.view(bt, c, -1).transpose(1, 2).unsqueeze(1)
        v = v.view(bt, c, -1).transpose(1, 2).unsqueeze(1)
        out = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        out = out.squeeze(1).transpose(1, 2).reshape(bt, c, h, w).to(identity.dtype)
        out = self.proj(out)
        out = rearrange(out, "(b t) c h w -> b c t h w", t=t)
        return out + identity


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        input_channels=3,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(True, True, False),
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_downsample = list(temperal_downsample)

        dims = [dim * u for u in [1] + self.dim_mult]
        scale = 1.0
        self.conv1 = CausalConv3d(input_channels, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in self.attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(self.dim_mult) - 1:
                mode = "downsample3d" if self.temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class WanVAE(nn.Module):
    """Encode-only Wan 2.1 VAE; decoder weights are accepted but unused."""

    def __init__(
        self,
        dim=128,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(False, True, True),
        image_channels=3,
        conv_out_channels=3,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        mult = list(dim_mult)
        self.temperal_downsample = list(temperal_downsample)
        self.temperal_upsample = self.temperal_downsample[::-1]
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            image_channels,
            mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)

    @torch.no_grad()
    def encode(self, x):
        assert x.ndim == 5 and x.shape[2] == 1, "image encoding expects BCTHW with T=1"
        out = self.encoder(x)
        mu, _log_var = self.conv1(out).chunk(2, dim=1)
        return mu


def wan_vae_from_state_dict(state: dict[str, torch.Tensor]) -> WanVAE:
    dim = state["decoder.head.0.gamma"].shape[0]
    image_channels = state["encoder.conv1.weight"].shape[1]
    conv_out_channels = state["decoder.head.2.weight"].shape[0]
    model = WanVAE(
        dim=dim,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(False, True, True),
        image_channels=image_channels,
        conv_out_channels=conv_out_channels,
        dropout=0.0,
    )
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    bad_unexpected = [key for key in unexpected if not key.startswith("decoder.")]
    assert not bad_unexpected, f"unexpected encoder keys: {bad_unexpected[:5]}"
    missing = list(incompatible.missing_keys)
    assert not missing, f"missing encoder keys: {missing[:5]}"
    return model


class QwenImageVAEEncoder:
    """Native Qwen Image VAE latents: pixel [-1,1] -> causal encode -> mu."""

    def __init__(self, vae_path: str, *, device: str | torch.device = "cuda", dtype=torch.bfloat16):
        state = {k: v for k, v in load_file(str(vae_path)).items()}
        self.model = wan_vae_from_state_dict(state)
        del state
        self.device = torch.device(device)
        self.dtype = dtype
        self.model.to(device=self.device, dtype=dtype).eval()

    @torch.inference_mode()
    def encode(self, image_bhwc: torch.Tensor) -> torch.Tensor:
        """image_bhwc: [B, H, W, C] floats in [0, 1]; returns mu [B, 16, 1, h, w] float32."""
        pixels = image_bhwc
        pixels = pixels[..., : pixels.shape[-3] // 8 * 8, : pixels.shape[-2] // 8 * 8]
        x = pixels.movedim(-1, 1).movedim(1, 0).unsqueeze(0)
        x = (x * 2.0 - 1.0).to(device=self.device, dtype=self.dtype)
        mu = self.model.encode(x)[0]
        mu = mu.permute(1, 0, 2, 3).unsqueeze(2).float().cpu()
        return mu
