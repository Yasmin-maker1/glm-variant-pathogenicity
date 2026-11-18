"""Small shared helpers: seeding, logging, config loading and resource tracking."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOG = logging.getLogger("glmvar")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for data-only steps
        pass


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a YAML config. A config may name a parent with ``inherit: other.yaml``."""
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_update(load_config(path.parent / parent), cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)


class ResourceTracker:
    """Records wall-clock GPU-hours and peak GPU memory for the cost analysis."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.peak_mem_gb = 0.0

    @contextmanager
    def track(self):
        try:
            import torch

            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            yield self
        finally:
            self.seconds += time.perf_counter() - start
            if cuda:
                peak = torch.cuda.max_memory_allocated() / 1024**3
                self.peak_mem_gb = max(self.peak_mem_gb, peak)

    @property
    def gpu_hours(self) -> float:
        return self.seconds / 3600.0

    def as_dict(self) -> dict:
        return {"gpu_hours": self.gpu_hours, "peak_gpu_mem_gb": self.peak_mem_gb, "seconds": self.seconds}
