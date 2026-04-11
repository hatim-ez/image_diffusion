from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .base import Backbone


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return rearrange(x, "b c h w -> b (h w) c")


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, context_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, kdim=context_dim, vdim=context_dim)
        self.cross_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h = self.cross_norm(x)
        key_padding = (mask == 0) if mask is not None else None
        cross_out, _ = self.cross_attn(h, context, context, key_padding_mask=key_padding)
        x = x + cross_out
        x = x + self.mlp(x)
        return x


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = torch.arange(half_dim, device=timesteps.device) / half_dim
    freqs = torch.exp(-math.log(10000) * exponent)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TransformerBackbone(Backbone):
    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        context_dim: int = 768,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(embed_dim, heads, mlp_ratio, context_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, (patch_size ** 2) * in_channels)
        self.patch_size = patch_size
        self.image_size = image_size

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def _run_block(
        self,
        block: DiTBlock,
        patches: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.gradient_checkpointing or not self.training:
            return block(patches, context, mask)
        if mask is None:
            return checkpoint(lambda p, c: block(p, c, None), patches, context, use_reentrant=False)
        return checkpoint(block, patches, context, mask, use_reentrant=False)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, conditioning: Dict[str, torch.Tensor]) -> torch.Tensor:
        b, _, h, w = x.shape
        patches = self.patch_embed(x)
        patches = patches + self.pos_embed
        time_emb = self.time_embed(timestep_embedding(timesteps, patches.shape[-1]))
        patches = patches + time_emb[:, None, :]
        context = conditioning.get("text")
        mask = conditioning.get("mask")
        for block in self.blocks:
            patches = self._run_block(block, patches, context, mask)
        patches = self.norm(patches)
        pixels = self.proj(patches)
        pixels = rearrange(pixels, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=self.patch_size, p2=self.patch_size, h=h // self.patch_size, w=w // self.patch_size, c=x.shape[1])
        return pixels
