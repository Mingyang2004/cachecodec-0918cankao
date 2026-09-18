"""Portable external path helpers for CacheCodec experiments."""

from __future__ import annotations

import os
from typing import Any


def expand_external_paths(value: Any) -> Any:
    """Expand environment variables in nested JSON/YAML configuration values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_external_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_external_paths(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_external_paths(item) for key, item in value.items()}
    return value


def external_root(kind: str) -> str:
    """Return a configurable root for models, data, or run artifacts."""
    env_name = {
        "model": "CACHECODEC_MODEL_ROOT",
        "data": "CACHECODEC_DATA_ROOT",
        "run": "CACHECODEC_RUN_ROOT",
    }.get(kind)
    if env_name is None:
        raise ValueError(f"Unknown CacheCodec external root kind: {kind!r}")
    return os.environ.get(env_name, os.path.join("local", f"{kind}s"))

