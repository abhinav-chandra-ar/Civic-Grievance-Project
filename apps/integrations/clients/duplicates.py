"""Duplicate detection client hook."""
from __future__ import annotations

from .base import LOCAL_STUB_PROVIDER, confidence


def detect_possible_duplicates(
    *,
    raw_text: str,
    category_code: str = "",
    landmark_code: str | None = None,
    content_hashes: list[str] | None = None,
) -> dict[str, object]:
    """Return duplicate candidate hints only."""
    return {
        "possible_duplicate_tracking_code": None,
        "confidence": confidence(0.0),
        "candidates": [],
        "provider": LOCAL_STUB_PROVIDER,
        "metadata": {
            "text_length": len(raw_text.strip()),
            "category_code": category_code,
            "landmark_code": landmark_code,
            "content_hash_count": len(content_hashes or []),
        },
    }
