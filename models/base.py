from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


class Backbone(nn.Module):
    """
    Abstract base class for diffusion backbones.
    """

    def enable_gradient_checkpointing(self) -> None:
        return None

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, conditioning: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class Conditioning:
    text_embeds: torch.Tensor
    pooled: torch.Tensor | None = None
    mask: torch.Tensor | None = None


class TextToImageModel(nn.Module):
    def __init__(self, text_encoder: nn.Module, backbone: Backbone, head: nn.Module) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.backbone = backbone
        self.head = head

    def encode_conditioning(self, tokens: torch.Tensor, mask: torch.Tensor) -> Conditioning:
        embeddings = self.text_encoder(tokens, attention_mask=mask)
        return Conditioning(text_embeds=embeddings["embeds"], pooled=embeddings.get("pooled"), mask=mask)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        conditioning = self.encode_conditioning(tokens, mask)
        hidden = self.backbone(x, timesteps, {"text": conditioning.text_embeds, "mask": mask})
        return self.head(hidden)
