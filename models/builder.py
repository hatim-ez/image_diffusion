from __future__ import annotations

from diffusion_image.config import ModelConfig
from .base import TextToImageModel
from .diffusion_head import DiffusionHead
from .text_encoder import SentencePieceTextEncoder
from .transformer import TransformerBackbone
from .unet import UNetBackbone


def _context_dim(cfg: ModelConfig) -> int:
    if cfg.architecture == "transformer":
        return cfg.transformer.get("context_dim", 768)
    return cfg.unet.get("context_dim", 768)


def build_model(cfg: ModelConfig, latent_mode: bool = True, vocab_size: int | None = None) -> TextToImageModel:
    in_channels = cfg.latent_dim if latent_mode else 3
    context_dim = _context_dim(cfg)
    if vocab_size is None:
        raise ValueError("A tokenizer vocab size is required to build the text encoder.")
    text_encoder = SentencePieceTextEncoder(
        vocab_size=vocab_size,
        max_length=cfg.text_context_tokens,
        hidden_size=context_dim,
        num_layers=cfg.text_encoder.get("depth", 6),
        num_heads=cfg.text_encoder.get("heads", 8),
        mlp_ratio=cfg.text_encoder.get("mlp_ratio", 4.0),
        dropout=cfg.text_encoder.get("dropout", 0.0),
    )
    if cfg.architecture == "transformer":
        backbone = TransformerBackbone(
            in_channels=in_channels,
            embed_dim=cfg.transformer.get("embed_dim", 768),
            depth=cfg.transformer.get("depth", 12),
            heads=cfg.transformer.get("heads", 12),
            mlp_ratio=cfg.transformer.get("mlp_ratio", 4.0),
            patch_size=cfg.transformer.get("patch_size", 2),
            context_dim=context_dim,
            image_size=cfg.transformer.get("image_size", 32),
        )
        head = DiffusionHead(in_channels=in_channels, out_channels=in_channels)
    else:
        backbone = UNetBackbone(
            in_channels=in_channels,
            model_channels=cfg.unet.get("model_channels", 320),
            channel_mults=tuple(cfg.unet.get("channel_mults", (1, 2, 4, 4))),
            num_res_blocks=cfg.unet.get("num_res_blocks", 2),
            context_dim=context_dim,
            dropout=cfg.unet.get("dropout", 0.0),
        )
        head = DiffusionHead(in_channels=in_channels, out_channels=in_channels)
    return TextToImageModel(text_encoder=text_encoder, backbone=backbone, head=head)
