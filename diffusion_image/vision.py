from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def save_image_grid(tensor: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2
    grid = make_grid(tensor, nrow=nrow)
    save_image(grid, path)
