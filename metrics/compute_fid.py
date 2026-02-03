#!/usr/bin/env python3
"""
Compute FID/KID between real and generated image folders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def load_folder(path: Path) -> list[Image.Image]:
    images = []
    for file in sorted(path.glob("*.png")):
        try:
            images.append(Image.open(file).convert("RGB"))
        except Exception:
            continue
    if not images:
        raise ValueError(f"No PNG files in {path}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID/KID")
    parser.add_argument("--real", type=Path, required=True, help="Directory of real/reference PNGs")
    parser.add_argument("--fake", type=Path, required=True, help="Directory of generated PNGs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    preprocess = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
        ]
    )
    device = torch.device(args.device)
    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=50, normalize=True).to(device)

    real_images = load_folder(args.real)
    fake_images = load_folder(args.fake)

    for img in real_images:
        tensor = preprocess(img).unsqueeze(0).to(device)
        fid.update(tensor, real=True)
        kid.update(tensor, real=True)

    for img in fake_images:
        tensor = preprocess(img).unsqueeze(0).to(device)
        fid.update(tensor, real=False)
        kid.update(tensor, real=False)

    print(f"FID: {fid.compute().item():.4f}")
    mean, std = kid.compute()
    print(f"KID mean: {mean.item():.4f} ± {std.item():.4f}")


if __name__ == "__main__":
    main()
