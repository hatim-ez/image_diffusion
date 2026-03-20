#!/usr/bin/env python3
"""
Compute CLIP similarity between generated images and prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import open_clip
from PIL import Image


def _load_metadata(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _discover_metadata(images_path: Path, metadata_path: Path | None) -> Path | None:
    if metadata_path is not None:
        return metadata_path
    candidate = images_path / "metadata.jsonl"
    if candidate.exists():
        return candidate
    return None


def load_image_records(images_path: Path, prompts_path: Path | None, metadata_path: Path | None) -> list[tuple[Path, str]]:
    discovered_metadata = _discover_metadata(images_path, metadata_path)
    if discovered_metadata is not None:
        base_dir = discovered_metadata.parent
        records = _load_metadata(discovered_metadata)
        return [(base_dir / record["image"], record["prompt"]) for record in records]

    if prompts_path is None:
        raise ValueError("Provide --prompts or point --images to a directory containing metadata.jsonl.")

    image_paths = sorted(images_path.glob("*.png"))
    prompts = read_prompts(prompts_path)
    if len(image_paths) != len(prompts):
        raise ValueError(
            f"Image/prompt count mismatch: found {len(image_paths)} PNGs and {len(prompts)} prompts."
        )
    return list(zip(image_paths, prompts))


def read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP score for samples")
    parser.add_argument("--images", type=Path, required=True, help="Directory with PNG outputs")
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_image_records(args.images, args.prompts, args.metadata)
    device = torch.device(args.device)

    model, preprocess = open_clip.create_model_from_pretrained("ViT-B-32-quickgelu", "laion400m_e32")
    model.to(device).eval()

    with torch.no_grad():
        images = [Image.open(path).convert("RGB") for path, _ in records]
        prompts = [prompt for _, prompt in records]
        image_tensors = torch.stack([preprocess(img) for img in images]).to(device)
        text_tokens = open_clip.tokenize(prompts).to(device)
        image_features = model.encode_image(image_tensors)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        sims = (image_features * text_features).sum(dim=-1)
        print(f"CLIP mean: {sims.mean().item():.4f} ± {sims.std().item():.4f}")


if __name__ == "__main__":
    main()
