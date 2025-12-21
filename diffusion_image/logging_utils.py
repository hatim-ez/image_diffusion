"""
Utilities for experiment logging via JSONL and TensorBoard.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class JSONLLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", buffering=1)

    def log(self, record: Dict[str, Any]) -> None:
        record = {"timestamp": time.time(), **record}
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class TensorboardLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        for key, value in values.items():
            self.writer.add_scalar(f"{tag}/{key}", value, step)

    def log_images(self, tag: str, images, step: int) -> None:
        self.writer.add_images(tag, images, step)

    def close(self) -> None:
        self.writer.close()


class ExperimentLogger:
    def __init__(self, jsonl_path: str | Path, tb_log_dir: Optional[str | Path] = None) -> None:
        self.jsonl_logger = JSONLLogger(jsonl_path)
        self.tb_logger = TensorboardLogger(tb_log_dir) if tb_log_dir else None

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        payload = {"step": step, **metrics}
        self.jsonl_logger.log(payload)
        if self.tb_logger:
            self.tb_logger.log_scalars("metrics", metrics, step)

    def log_images(self, tag: str, images, step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_images(tag, images, step)

    def close(self) -> None:
        self.jsonl_logger.close()
        if self.tb_logger:
            self.tb_logger.close()
