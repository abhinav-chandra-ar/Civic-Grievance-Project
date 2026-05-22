"""Shared helpers for integration client hooks."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LOCAL_STUB_PROVIDER = "local_stub"


def metadata_object(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a plain metadata object for integration result payloads."""
    return dict(value or {})


def confidence(value: float) -> float:
    """Clamp confidence values to the expected 0..1 range."""
    return max(0.0, min(1.0, value))
