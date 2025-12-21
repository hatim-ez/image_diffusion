from __future__ import annotations

from typing import Optional

import torch

from .process import DiffusionProcess, extract


def _guided_prediction(
    model: torch.nn.Module,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    guidance_scale: float,
    uncond_tokens: Optional[torch.Tensor],
    uncond_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if guidance_scale == 1.0 or uncond_tokens is None or uncond_mask is None:
        return model(x, timesteps, tokens, mask)
    x_in = torch.cat([x, x], dim=0)
    tokens_in = torch.cat([uncond_tokens, tokens], dim=0)
    mask_in = torch.cat([uncond_mask, mask], dim=0)
    noise_pred = model(x_in, timesteps.repeat(2), tokens_in, mask_in)
    noise_uncond, noise_text = noise_pred.chunk(2)
    return noise_uncond + guidance_scale * (noise_text - noise_uncond)


def ddpm_sample(
    model: torch.nn.Module,
    diffusion: DiffusionProcess,
    x_shape: torch.Size,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    guidance_scale: float = 1.0,
    uncond_tokens: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = tokens.device
    x = torch.randn(x_shape, device=device)
    for t in reversed(range(diffusion.timesteps)):
        timesteps = torch.full((x_shape[0],), t, device=device, dtype=torch.long)
        noise_pred = _guided_prediction(
            model, x, timesteps, tokens, mask, guidance_scale, uncond_tokens, uncond_mask
        )
        stats = diffusion.p_mean_variance(noise_pred, x, timesteps)
        if t > 0:
            noise = torch.randn_like(x)
            x = stats["mean"] + torch.sqrt(stats["variance"]) * noise
        else:
            x = stats["mean"]
    return x


def ddim_sample(
    model: torch.nn.Module,
    diffusion: DiffusionProcess,
    x_shape: torch.Size,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    num_steps: int = 20,
    guidance_scale: float = 1.0,
    uncond_tokens: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = tokens.device
    times = torch.linspace(0, diffusion.timesteps - 1, num_steps, dtype=torch.long, device=device)
    x = torch.randn(x_shape, device=device)
    for idx in reversed(range(num_steps)):
        t = times[idx]
        timesteps = torch.full((x_shape[0],), t, device=device, dtype=torch.long)
        noise_pred = _guided_prediction(
            model, x, timesteps, tokens, mask, guidance_scale, uncond_tokens, uncond_mask
        )
        alpha = extract(diffusion.alphas_cumprod, timesteps, x.shape)
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha)
        pred_x0 = (x - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        if idx == 0:
            x = pred_x0
            continue
        alpha_next = extract(diffusion.alphas_cumprod, torch.full_like(timesteps, times[idx - 1]), x.shape)
        sigma = 0.0
        dir_xt = torch.sqrt(1 - alpha_next) * noise_pred
        x = torch.sqrt(alpha_next) * pred_x0 + dir_xt + sigma * torch.randn_like(x)
    return x
