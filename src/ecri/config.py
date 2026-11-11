"""Typed run configuration, loaded from configs/*.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from . import paths


@dataclass(frozen=True)
class SplitConfig:
    # "school" holds whole schools out (the paper's School-split, the honest
    # generalisation test); "student" splits rows and leaks school context.
    strategy: str = "school"
    test_size: float = 0.25
    n_folds: int = 5


@dataclass(frozen=True)
class Config:
    target: str = "RMwidRS_target"
    seed: int = 20260101
    # Drop a feature column when this fraction of the labelled rows is missing.
    max_missing_rate: float = 0.95
    split: SplitConfig = field(default_factory=SplitConfig)
    # Rows to synthesise when the real CSVs are absent.
    synthetic_rows: int = 6803

    def __post_init__(self) -> None:
        from .schema import TARGETS

        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {sorted(TARGETS)}, got {self.target!r}")
        if self.split.strategy not in {"school", "student"}:
            raise ValueError(f"unknown split strategy {self.split.strategy!r}")
        if not 0.0 < self.split.test_size < 1.0:
            raise ValueError("split.test_size must be in (0, 1)")


def _coerce(raw: dict[str, Any]) -> Config:
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    split = SplitConfig(**raw.pop("split", {}) or {})
    return Config(split=split, **raw)


def load_config(path: str | Path | None = None) -> Config:
    """Read a YAML config. With no path, use configs/default.yaml if it exists."""
    if path is None:
        path = paths.CONFIGS / "default.yaml"
        if not Path(path).is_file():
            return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return _coerce(raw)
