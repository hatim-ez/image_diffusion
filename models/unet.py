from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import Backbone


def sinusoidal_embedding(dim: int) -> nn.Module:
    class SinusoidalPositionEmbedding(nn.Module):
        def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
            device = timesteps.device
            half_dim = dim // 2
            exponent = torch.arange(half_dim, device=device) / half_dim
            freqs = torch.exp(-math.log(10000) * exponent)
            args = timesteps[:, None].float() * freqs[None]
            embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if dim % 2 == 1:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding
    return SinusoidalPositionEmbedding()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)


class CrossAttention2D(nn.Module):
    def __init__(self, channels: int, context_dim: int, heads: int = 4, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.to_q = nn.Linear(channels, inner_dim)
        self.to_k = nn.Linear(context_dim, inner_dim)
        self.to_v = nn.Linear(context_dim, inner_dim)
        self.proj = nn.Linear(inner_dim, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.to_q(rearrange(x, "b c h w -> b (h w) c"))
        k = self.to_k(context)
        v = self.to_v(context)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        scale = q.shape[-1] ** -0.5
        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) * scale
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, context_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = CrossAttention2D(channels, context_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x)
        return x + self.attn(h, context, mask)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNetBackbone(Backbone):
    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 320,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        context_dim: int = 768,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        time_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            sinusoidal_embedding(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        downs = []
        channels = model_channels
        self.downsample_channels: List[int] = []
        for i, mult in enumerate(channel_mults):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks):
                downs.append(ResidualBlock(channels, out_channels, time_dim, dropout))
                channels = out_channels
                downs.append(AttentionBlock(channels, context_dim))
                self.downsample_channels.append(channels)
            if i != len(channel_mults) - 1:
                downs.append(Downsample(channels))
        self.downs = nn.ModuleList(downs)

        self.mid_block1 = ResidualBlock(channels, channels, time_dim, dropout)
        self.mid_attn = AttentionBlock(channels, context_dim)
        self.mid_block2 = ResidualBlock(channels, channels, time_dim, dropout)

        ups = []
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_channels = self.downsample_channels.pop()
                ups.append(ResidualBlock(channels + skip_channels, out_channels, time_dim, dropout))
                ups.append(AttentionBlock(out_channels, context_dim))
                channels = out_channels
            if i != 0:
                ups.append(Upsample(channels))
        self.ups = nn.ModuleList(ups)
        self.out_norm = nn.GroupNorm(32, channels)
        self.out_conv = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, conditioning: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_context = conditioning.get("text")
        mask = conditioning.get("mask")
        t_emb = self.time_embed(timesteps)
        h = self.in_conv(x)
        residuals = [h]
        idx = 0
        for module in self.downs:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            elif isinstance(module, AttentionBlock):
                h = module(h, text_context, mask)
            else:
                h = module(h)
            residuals.append(h)
            idx += 1

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, text_context, mask)
        h = self.mid_block2(h, t_emb)

        for module in self.ups:
            if isinstance(module, ResidualBlock):
                skip = residuals.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, t_emb)
            elif isinstance(module, AttentionBlock):
                h = module(h, text_context, mask)
            else:
                h = module(h)

        h = self.out_norm(h)
        h = torch.nn.functional.silu(h)
        return self.out_conv(h)
