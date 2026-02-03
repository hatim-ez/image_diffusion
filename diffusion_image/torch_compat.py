"""
Compatibility shims for torch features that may be missing on older builds.
"""

from __future__ import annotations

import torch

try:
    import torch.distributed as _dist
except Exception:  # pragma: no cover
    _dist = None


class _DummyXPU:
    @staticmethod
    def empty_cache() -> None:
        return None

    @staticmethod
    def synchronize() -> None:
        return None

    @staticmethod
    def device_count() -> int:
        return 0

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def manual_seed(_: int) -> None:
        return None

    @staticmethod
    def manual_seed_all(_: int) -> None:
        return None


if not hasattr(torch, "xpu"):
    torch.xpu = _DummyXPU()  # type: ignore[attr-defined]

if _dist is not None and not hasattr(_dist, "device_mesh"):
    class _DeviceMeshStub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch.distributed.device_mesh.DeviceMesh not available on this build")

    class _DeviceMeshModule:
        DeviceMesh = _DeviceMeshStub

    _dist.device_mesh = _DeviceMeshModule()  # type: ignore[attr-defined]
