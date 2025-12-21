"""
Simple exponential moving average helper for model parameters.
"""

from __future__ import annotations

from typing import Iterable

import torch


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow or not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow or not param.requires_grad:
                continue
            param.data.copy_(self.shadow[name].data)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]

    def apply_to(self, model: torch.nn.Module) -> "EMATemporaryScope":
        return EMATemporaryScope(model, self)


class EMATemporaryScope:
    def __init__(self, model: torch.nn.Module, ema: ExponentialMovingAverage) -> None:
        self.model = model
        self.ema = ema
        self.backup = None

    def __enter__(self):
        self.backup = {name: param.detach().clone() for name, param in self.model.named_parameters()}
        self.ema.copy_to(self.model)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, param in self.model.named_parameters():
            param.data.copy_(self.backup[name])
        self.backup = None
