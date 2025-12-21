"""
Configuration dataclasses and helpers to load YAML experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_ratio: float = 0.1
    total_steps: int = 100000


@dataclass
class DatasetConfig:
    shards_path: str
    tokenizer_path: str
    image_size: int = 256
    num_workers: int = 4
    batch_size: int = 8
    shuffle_buffer: int = 2048
    pin_memory: bool = True
    persistent_workers: bool = True
    latent_mode: bool = False
    caption_dropout: float = 0.1


@dataclass
class ModelConfig:
    architecture: str = "unet"
    latent_dim: int = 4
    text_context_tokens: int = 77
    use_fp16: bool = True
    gradient_checkpointing: bool = False
    clip_model: str = "openai/clip-vit-large-patch14"
    unet: Dict[str, Any] = field(default_factory=dict)
    transformer: Dict[str, Any] = field(default_factory=dict)
    vae_model: Optional[str] = None


@dataclass
class DiffusionConfig:
    schedule: str = "cosine"
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    sampler: str = "ddim"
    ddim_steps: int = 20


@dataclass
class TrainingConfig:
    max_steps: int = 100000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    eval_interval: int = 1000
    checkpoint_interval: int = 2000
    resume_from: Optional[str] = None
    output_dir: str = "checkpoints"
    log_dir: str = "logs"
    validation_prompts: str = "configs/validation_prompts.txt"
    mixed_precision: str = "fp16"
    seed: int = 42
    guidance_scale: float = 7.5
    eval_batch_size: int = 4


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig

    raw: Dict[str, Any] = field(default_factory=dict)


def _merge_dict(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = defaults.copy()
    merged.update(override)
    return merged


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r") as handle:
        raw = yaml.safe_load(handle)

    dataset = DatasetConfig(**raw.get("dataset", {}))
    model = ModelConfig(**raw.get("model", {}))
    diffusion = DiffusionConfig(**raw.get("diffusion", {}))
    optimizer = OptimizerConfig(**raw.get("optimizer", {}))
    scheduler = SchedulerConfig(**raw.get("scheduler", {}))
    training = TrainingConfig(**raw.get("training", {}))

    return ExperimentConfig(
        dataset=dataset,
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        scheduler=scheduler,
        training=training,
        raw=raw,
    )
