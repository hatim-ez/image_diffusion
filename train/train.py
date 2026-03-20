#!/usr/bin/env python3
"""
Training script for text-to-image diffusion.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import open_clip
from torchvision.transforms import functional as TF
from diffusers import AutoencoderKL

from diffusion_image import torch_compat  # noqa: F401
from diffusion_image.config import load_config
from diffusion_image.logging_utils import ExperimentLogger
from diffusion_image.ema import ExponentialMovingAverage
from diffusion_image.checkpointing import load_checkpoint, save_checkpoint
from diffusion_image.distributed import init_distributed, is_main_process
from diffusion_image.diffusion import DiffusionProcess, ddim_sample
from diffusion_image.vision import save_image_grid
from data import SentencePieceTokenizer, create_webdataset_dataloader
from models.builder import build_model


def create_optimizer(model: torch.nn.Module, cfg) -> AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)


def create_scheduler(optimizer: AdamW, cfg) -> LambdaLR:
    warmup_steps = int(cfg.warmup_ratio * cfg.total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(1, warmup_steps), 1e-3)
        progress = (step - warmup_steps) / max(1, cfg.total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    return LambdaLR(optimizer, lr_lambda)


def prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def load_validation_prompts(path: Path, max_prompts: int) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()][:max_prompts]


def make_null_tokens(tokenizer: SentencePieceTokenizer, batch_size: int, device: torch.device) -> torch.Tensor:
    tokens = torch.full((batch_size, tokenizer.max_length), tokenizer.pad_id, dtype=torch.long, device=device)
    tokens[:, 0] = tokenizer.null_prompt_id
    return tokens


def compute_clip_stats(images: torch.Tensor, prompts: List[str], device: torch.device) -> Dict[str, float]:
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.to(device)
    model.eval()
    with torch.no_grad():
        image_list = []
        for img in images:
            img = ((img + 1) / 2).clamp(0, 1)
            pil = TF.to_pil_image(img.cpu())
            image_list.append(preprocess(pil))
        image_input = torch.stack(image_list).to(device)
        text_tokens = open_clip.tokenize(prompts).to(device)
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sims = (image_features * text_features).sum(dim=-1)
        return {"clip_mean": sims.mean().item(), "clip_std": sims.std().item()}


def evaluate(
    model: torch.nn.Module,
    diffusion: DiffusionProcess,
    tokenizer: SentencePieceTokenizer,
    prompts_file: Path,
    device: torch.device,
    dataset_cfg,
    model_cfg,
    training_cfg,
    diffusion_cfg,
    step: int,
    output_dir: Path,
    ema: ExponentialMovingAverage,
    vae: AutoencoderKL | None = None,
) -> Dict[str, float]:
    prompts = load_validation_prompts(prompts_file, training_cfg.eval_batch_size)
    if not prompts:
        return {}
    tokens = torch.stack([torch.tensor(tokenizer.encode(p), dtype=torch.long) for p in prompts]).to(device)
    mask = (tokens != tokenizer.pad_id).long()
    uncond_tokens = make_null_tokens(tokenizer, tokens.size(0), device)
    uncond_mask = (uncond_tokens != tokenizer.pad_id).long()

    with ema.apply_to(model):
        samples = ddim_sample(
            model,
            diffusion,
            (
                tokens.size(0),
                model_cfg.latent_dim if dataset_cfg.latent_mode else 3,
                dataset_cfg.image_size,
                dataset_cfg.image_size,
            ),
            tokens,
            mask,
            num_steps=diffusion_cfg.ddim_steps,
            guidance_scale=training_cfg.guidance_scale,
            uncond_tokens=uncond_tokens,
            uncond_mask=uncond_mask,
        )
    if dataset_cfg.latent_mode:
        if vae is None:
            raise RuntimeError("Latent mode requires a VAE for decoding.")
        with torch.no_grad():
            latents = samples / vae.config.scaling_factor
            samples = vae.decode(latents).sample
    grid_path = output_dir / f"samples_step_{step:07d}.png"
    save_image_grid(samples.detach().cpu(), grid_path, nrow=min(4, samples.size(0)))
    clip_stats = compute_clip_stats(samples.detach().cpu(), prompts, device)
    return clip_stats


def train(config_path: Path) -> None:
    cfg = load_config(config_path)
    ctx = init_distributed(cfg.training.seed)
    device = ctx.device
    tokenizer = SentencePieceTokenizer(cfg.dataset.tokenizer_path, max_length=cfg.model.text_context_tokens)
    dataloader = create_webdataset_dataloader(cfg.dataset, tokenizer)
    model = build_model(
        cfg.model,
        latent_mode=cfg.dataset.latent_mode,
        vocab_size=tokenizer.vocab_size,
    ).to(device)
    diffusion = DiffusionProcess(cfg.diffusion).to(device)
    vae = None
    if cfg.dataset.latent_mode and cfg.model.vae_model:
        vae = AutoencoderKL.from_pretrained(cfg.model.vae_model, subfolder="vae")
        vae.to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
    optimizer = create_optimizer(model, cfg.optimizer)
    scheduler = create_scheduler(optimizer, cfg.scheduler)
    ema = ExponentialMovingAverage(model, decay=cfg.training.ema_decay)
    scaler = GradScaler(enabled=cfg.model.use_fp16 and device.type == "cuda")
    logger = None
    if is_main_process():
        log_dir = Path(cfg.training.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = ExperimentLogger(log_dir / "train.jsonl", log_dir)

    start_step = 0
    if cfg.training.resume_from:
        ckpt = load_checkpoint(cfg.training.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt.get("scheduler", {}))
        ema.load_state_dict(ckpt["ema"])
        start_step = ckpt["step"]

    model.train()
    step = start_step
    iterator = iter(dataloader)
    progress = tqdm(total=cfg.training.max_steps, initial=start_step, disable=not is_main_process())

    while step < cfg.training.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        batch = prepare_batch(batch, device)
        images = batch["images"]
        tokens = batch["tokens"]
        mask = batch["mask"]
        noise = torch.randn_like(images)
        timesteps = diffusion.sample_timesteps(images.size(0), device)
        noisy = diffusion.q_sample(images, timesteps, noise)
        with autocast(enabled=cfg.model.use_fp16 and device.type == "cuda"):
            preds = model(noisy, timesteps, tokens, mask)
            loss = F.mse_loss(preds, noise)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        ema.update(model)
        step += 1
        if logger and step % 10 == 0:
            logger.log_metrics({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step)
        progress.update(1)

        if logger and step % cfg.training.eval_interval == 0 and is_main_process():
            eval_metrics = evaluate(
                model=model,
                diffusion=diffusion,
                tokenizer=tokenizer,
                prompts_file=Path(cfg.training.validation_prompts),
                device=device,
                dataset_cfg=cfg.dataset,
                model_cfg=cfg.model,
                training_cfg=cfg.training,
                diffusion_cfg=cfg.diffusion,
                step=step,
                output_dir=Path(cfg.training.log_dir),
                ema=ema,
                vae=vae,
            )
            if eval_metrics:
                logger.log_metrics(eval_metrics, step)

        if step % cfg.training.checkpoint_interval == 0 and is_main_process():
            ckpt_path = Path(cfg.training.output_dir) / f"step_{step:07d}.pt"
            save_checkpoint(
                ckpt_path,
                step,
                model,
                optimizer,
                scheduler,
                ema.state_dict(),
                tokenizer_hash=tokenizer.hash(),
                config=cfg.raw,
            )

    if logger:
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(args.config)
