"""
WebDataset pipeline yielding (image, text_tokens, mask) batches.
"""

from __future__ import annotations

import io
import random
from typing import Callable, Dict, Iterable

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import webdataset as wds

from diffusion_image.config import DatasetConfig
from .text_cleaning import clean_caption
from .tokenizer import SentencePieceTokenizer


def build_image_transform(size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize(size + 16),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _null_tokens(tokenizer: SentencePieceTokenizer) -> torch.Tensor:
    tokens = [tokenizer.null_prompt_id] + [tokenizer.pad_id] * (tokenizer.max_length - 1)
    return torch.tensor(tokens, dtype=torch.long)


def create_webdataset(
    cfg: DatasetConfig,
    tokenizer: SentencePieceTokenizer,
    distributed: bool = False,
) -> wds.WebDataset:
    handler = wds.handlers.warn_and_continue
    transform = build_image_transform(cfg.image_size)

    def preprocess_pixels(sample):
        image, caption = sample
        caption = clean_caption(caption)
        if not caption:
            raise wds.handlers.SkipSample("empty caption after cleaning")
        tokens = torch.tensor(tokenizer.encode(caption), dtype=torch.long)
        mask = (tokens != tokenizer.pad_id).long()
        if random.random() < cfg.caption_dropout:
            tokens = _null_tokens(tokenizer)
            mask = (tokens != tokenizer.pad_id).long()
        return {
            "image": transform(image),
            "tokens": tokens,
            "mask": mask,
            "raw_caption": caption,
        }

    def preprocess_latents(sample):
        latent_bytes, caption = sample
        caption = clean_caption(caption)
        if not caption:
            raise wds.handlers.SkipSample("empty caption after cleaning")
        tensor = torch.load(io.BytesIO(latent_bytes))
        if tensor.dim() != 3:
            raise wds.handlers.SkipSample("invalid latent tensor shape")
        tokens = torch.tensor(tokenizer.encode(caption), dtype=torch.long)
        mask = (tokens != tokenizer.pad_id).long()
        if random.random() < cfg.caption_dropout:
            tokens = _null_tokens(tokenizer)
            mask = (tokens != tokenizer.pad_id).long()
        return {
            "image": tensor.float(),
            "tokens": tokens,
            "mask": mask,
            "raw_caption": caption,
        }

    nodesplitter = wds.split_by_node if distributed else wds.shardlists.single_node_only
    dataset = wds.WebDataset(
        url=cfg.shards_path,
        handler=handler,
        resampled=True,
        nodesplitter=nodesplitter,
    ).shuffle(cfg.shuffle_buffer)
    if cfg.latent_mode:
        dataset = dataset.to_tuple("pt", "txt").map(preprocess_latents, handler=handler)
    else:
        dataset = dataset.decode("pil").to_tuple("jpg;png;jpeg", "txt").map(preprocess_pixels, handler=handler)
    return dataset


def collate_fn(batch: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    tokens = torch.stack([item["tokens"] for item in batch], dim=0)
    mask = torch.stack([item["mask"] for item in batch], dim=0)
    return {"images": images, "tokens": tokens, "mask": mask}


def create_webdataset_dataloader(
    cfg: DatasetConfig,
    tokenizer: SentencePieceTokenizer,
    distributed: bool = False,
) -> DataLoader:
    dataset = create_webdataset(cfg, tokenizer, distributed=distributed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        collate_fn=collate_fn,
    )
