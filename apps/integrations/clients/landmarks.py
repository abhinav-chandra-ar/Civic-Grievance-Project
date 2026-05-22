"""Landmark resolution client hook."""
from __future__ import annotations

from .base import LOCAL_STUB_PROVIDER, confidence


def resolve_landmark_mention(*, mention: str, location_text: str = "") -> dict[str, object]:
    """Return provider-style landmark search hints only."""
    normalized_mention = mention.strip()
    return {
        "landmark_code": None,
        "confidence": confidence(0.0),
        "candidates": [],
        "provider": LOCAL_STUB_PROVIDER,
        "metadata": {
            "mention": normalized_mention,
            "location_text": location_text.strip(),
        },
    }
