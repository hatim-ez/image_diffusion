from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SentencePieceTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 77,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("Text encoder hidden size must be divisible by the number of heads.")
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if input_ids.size(1) > self.max_length:
            raise ValueError(f"Expected sequence length <= {self.max_length}, got {input_ids.size(1)}.")
        hidden = self.token_embedding(input_ids) + self.position_embedding[:, : input_ids.size(1)]
        key_padding_mask = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        hidden = self.norm(hidden)
        pooled = self._masked_mean(hidden, attention_mask)
        return {"embeds": hidden, "pooled": pooled}

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (hidden * weights).sum(dim=1) / denom
