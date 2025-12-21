"""
Utility helpers for distributed or single-device training.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")


def init_distributed(seed: int = 42) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if distributed and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
    else:
        device = torch.device("cpu")

    seed_all(seed + rank)

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
