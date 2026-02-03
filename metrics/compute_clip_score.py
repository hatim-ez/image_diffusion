#!/usr/bin/env python3
"""
Compute CLIP similarity between generated images and prompts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import open_clip
from PIL import Image


def load_images(path: Path) -> list[Image.Image]:
    images = []
    for file in sorted(path.glob("*.png")):
        try:
            images.append(Image.open(file).convert("RGB"))
        except Exception:
            continue
    if not images:
        raise ValueError(f"No PNG images found in {path}")
    return images


def read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP score for samples")
    parser.add_argument("--images", type=Path, required=True, help="Directory with PNG outputs")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    prompts = read_prompts(args.prompts)
    images = load_images(args.images)
    device = torch.device(args.device)

    model, preprocess = open_clip.create_model_from_pretrained("ViT-B-32-quickgelu", "laion400m_e32")
    model.to(device).eval()

    with torch.no_grad():
        image_tensors = torch.stack([preprocess(img) for img in images]).to(device)
        text_tokens = open_clip.tokenize(prompts[: len(images)]).to(device)
        image_features = model.encode_image(image_tensors)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        sims = (image_features * text_features).sum(dim=-1)
        print(f"CLIP mean: {sims.mean().item():.4f} ± {sims.std().item():.4f}")


if __name__ == "__main__":
    main()
