from __future__ import annotations

import torch.nn as nn


class DiffusionHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        layers = [
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        ]
        super().__init__(*layers)
