from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from diffusion_image.config import DiffusionConfig
from .betas import make_beta_schedule


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    batch = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch, *((1,) * (len(x_shape) - 1)))


class DiffusionProcess(nn.Module):
    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        betas = make_beta_schedule(cfg.schedule, cfg.timesteps, cfg.beta_start, cfg.beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("posterior_variance", posterior_variance)

    @property
    def timesteps(self) -> int:
        return self.cfg.timesteps

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alphas = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas * x_start + sqrt_one_minus * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_recip_alphas = extract(1.0 / self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = extract(1.0 / self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas * x_t - sqrt_recipm1 * noise

    def p_mean_variance(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        alphas_cumprod = extract(self.alphas_cumprod, t, x_t.shape)
        model_mean = (1.0 / torch.sqrt(1.0 - betas_t)) * (x_t - betas_t / sqrt_one_minus * model_output)
        return {
            "mean": model_mean,
            "variance": extract(self.posterior_variance, t, x_t.shape),
        }
