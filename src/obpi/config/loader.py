"""YAML configuration loader with dot-notation access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Nested dict accessible via dot notation (``cfg.movement.v_threshold``)."""

    def __init__(self, mapping: dict[str, Any] | None = None) -> None:
        super().__init__()
        if mapping:
            for key, value in mapping.items():
                if isinstance(value, dict):
                    self[key] = Config(value)
                else:
                    self[key] = value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path | None = None) -> Config:
    """Load YAML config from disk.

    Args:
        path: Path to YAML file. Defaults to ``config/default.yaml``.

    Returns:
        ``Config`` object with dot-notation access.
    """
    if path is None:
        # repo root = two levels above src/obpi/config/loader.py
        path = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    return Config(data)
