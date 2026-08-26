import math

import torch
from einops import rearrange, repeat
from torch import nn


class VideoRopePosition3DEmb(nn.Module):
    """Ported from comfy/ldm/cosmos/position_embedding.py (NVIDIA Cosmos, Apache-2.0)."""

    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        device=None,
    ):
        super().__init__()
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.enable_fps_modulation = enable_fps_modulation

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2, device=device)[: (dim_h // 2)].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2, device=device)[: (dim_t // 2)].float() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def generate_embeddings(self, B_T_H_W_C, fps=None, device=None):
        h_ntk_factor = self.h_ntk_factor
        w_ntk_factor = self.w_ntk_factor
        t_ntk_factor = self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device=device))
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device=device))
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device=device))

        B, T, H, W, _ = B_T_H_W_C
        seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)
        half_emb_h = torch.outer(seq[:H].to(device=device), h_spatial_freqs)
        half_emb_w = torch.outer(seq[:W].to(device=device), w_spatial_freqs)

        if fps is None or self.enable_fps_modulation is False:
            half_emb_t = torch.outer(seq[:T].to(device=device), temporal_freqs)
        else:
            half_emb_t = torch.outer(seq[:T].to(device=device) / fps * self.base_fps, temporal_freqs)

        half_emb_h = torch.stack([torch.cos(half_emb_h), -torch.sin(half_emb_h), torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1)
        half_emb_w = torch.stack([torch.cos(half_emb_w), -torch.sin(half_emb_w), torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1)
        half_emb_t = torch.stack([torch.cos(half_emb_t), -torch.sin(half_emb_t), torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d x -> t h w d x", h=H, w=W),
                repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W),
                repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H),
            ],
            dim=-2,
        )

        return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()

    def forward(self, x_B_T_H_W_C, fps=None, device=None):
        return self.generate_embeddings(x_B_T_H_W_C.shape, fps=fps, device=device)


def time_snr_shift(alpha, t):
    """ComfyUI: sigma = alpha * t / (1 + (alpha - 1) * t); identity at alpha == 1."""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def flow_sigma_to_timestep(sigma, multiplier=1.0):
    return sigma * multiplier


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb, "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])
