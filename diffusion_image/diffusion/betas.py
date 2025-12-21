from __future__ import annotations

import math
import numpy as np
import torch


def make_beta_schedule(
    schedule: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == "cosine":
        timesteps_plus = timesteps + 1
        s = 0.008
        x = torch.linspace(0, timesteps, timesteps_plus)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 1e-8, 0.999)
    else:
        raise ValueError(f"Unknown beta schedule {schedule}")
    return betas
