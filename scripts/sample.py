#!/usr/bin/env python3
"""
Sampling script for trained checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL

from diffusion_image import torch_compat  # noqa: F401
from diffusion_image.config import load_config
from diffusion_image.diffusion import DiffusionProcess, ddim_sample, ddpm_sample
from diffusion_image.vision import save_image_grid
from data import SentencePieceTokenizer
from models.builder import build_model


def load_prompts(path: Path | None, prompt: str | None) -> list[str]:
    prompts = []
    if path and path.exists():
        prompts.extend([line.strip() for line in path.read_text().splitlines() if line.strip()])
    if prompt:
        prompts.append(prompt)
    if not prompts:
        raise ValueError("No prompts provided")
    return prompts


def make_tokens(tokenizer: SentencePieceTokenizer, prompts: list[str], device: torch.device) -> torch.Tensor:
    tokens = [torch.tensor(tokenizer.encode(p), dtype=torch.long) for p in prompts]
    return torch.stack(tokens).to(device)


def sample_latents(
    model: torch.nn.Module,
    diffusion: DiffusionProcess,
    cfg,
    x_shape: tuple[int, int, int, int],
    tokens: torch.Tensor,
    mask: torch.Tensor,
    guidance_scale: float,
    uncond_tokens: torch.Tensor,
    uncond_mask: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    sampler = cfg.diffusion.sampler.lower()
    if sampler == "ddim":
        return ddim_sample(
            model,
            diffusion,
            x_shape,
            tokens,
            mask,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            uncond_tokens=uncond_tokens,
            uncond_mask=uncond_mask,
        )
    if sampler == "ddpm":
        return ddpm_sample(
            model,
            diffusion,
            x_shape,
            tokens,
            mask,
            guidance_scale=guidance_scale,
            uncond_tokens=uncond_tokens,
            uncond_mask=uncond_mask,
        )
    raise ValueError(f"Unsupported sampler: {cfg.diffusion.sampler}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from diffusion model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("samples"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SentencePieceTokenizer(cfg.dataset.tokenizer_path, cfg.model.text_context_tokens)
    model = build_model(cfg.model, latent_mode=cfg.dataset.latent_mode).to(device)
    diffusion = DiffusionProcess(cfg.diffusion).to(device)
    vae = None
    if cfg.dataset.latent_mode and cfg.model.vae_model:
        vae = AutoencoderKL.from_pretrained(cfg.model.vae_model, subfolder="vae")
        vae.to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompts = load_prompts(args.prompts, args.prompt)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for idx in range(0, len(prompts), args.batch_size):
        chunk = prompts[idx : idx + args.batch_size]
        tokens = make_tokens(tokenizer, chunk, device)
        mask = (tokens != tokenizer.pad_id).long()
        uncond = torch.full_like(tokens, tokenizer.pad_id)
        uncond[:, 0] = tokenizer.null_prompt_id
        uncond_mask = (uncond != tokenizer.pad_id).long()
        with torch.no_grad():
            samples = sample_latents(
                model,
                diffusion,
                cfg,
                (
                    tokens.size(0),
                    cfg.model.latent_dim if cfg.dataset.latent_mode else 3,
                    cfg.dataset.image_size,
                    cfg.dataset.image_size,
                ),
                tokens,
                mask,
                args.guidance_scale,
                uncond,
                uncond_mask,
                args.num_steps,
            )
            if cfg.dataset.latent_mode and vae is not None:
                samples = vae.decode(samples / vae.config.scaling_factor).sample
        save_image_grid(samples.cpu(), args.outdir / f"batch_{idx//args.batch_size:04d}.png", nrow=min(4, samples.size(0)))


if __name__ == "__main__":
    main()
