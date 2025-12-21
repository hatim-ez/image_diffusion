#!/usr/bin/env python3
"""
Encode WebDataset images into Stable Diffusion VAE latents.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import torch
import webdataset as wds
from diffusers import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm


def load_vae(name: str, device: torch.device) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(name, subfolder="vae")
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def encode_dataset(shards: str, output_dir: Path, vae_name: str, image_size: int, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(vae_name, device)
    transform = transforms.Compose(
        [
            transforms.Resize(image_size + 16),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = (
        wds.WebDataset(shards)
        .decode("pil")
        .to_tuple("jpg;png;jpeg", "txt")
        .map_tuple(transform, lambda x: x)
    )
    loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = wds.ShardWriter(str(output_dir / "%06d.tar"))
    idx = 0
    with torch.no_grad():
        for images, captions in tqdm(loader, desc="Encoding latents"):
            images = images.to(device)
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            for latent, caption in zip(latents, captions):
                buffer = io.BytesIO()
                torch.save(latent.cpu().half(), buffer)
                sample = {
                    "__key__": f"latent-{idx:09d}",
                    "pt": buffer.getvalue(),
                    "txt": caption,
                }
                writer.write(sample)
                idx += 1
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode latent representations")
    parser.add_argument("--shards", type=str, required=True, help="Input image WebDataset pattern")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vae", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    encode_dataset(args.shards, args.output_dir, args.vae, args.image_size, args.batch_size)


if __name__ == "__main__":
    main()
