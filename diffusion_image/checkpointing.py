"""
Checkpoint helpers storing model, optimizer, EMA, and configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(
    path: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema_state: Dict[str, Any],
    tokenizer_hash: str,
    config: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": getattr(scheduler, "state_dict", lambda: {})(),
        "ema": ema_state,
        "tokenizer_hash": tokenizer_hash,
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
