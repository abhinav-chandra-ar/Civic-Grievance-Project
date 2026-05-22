"""Text/NLP classification client hook."""
from __future__ import annotations

from .base import LOCAL_STUB_PROVIDER, confidence

CATEGORY_KEYWORDS = {
    "road_damage": ("road", "pothole", "street", "damage"),
    "waste": ("waste", "garbage", "trash", "dump"),
    "water_supply": ("water", "pipe", "leak", "supply"),
    "street_light": ("light", "streetlight", "lamp"),
}


def classify_grievance_text(*, raw_text: str, language_hint: str | None = None) -> dict[str, object]:
    """Return deterministic text classification hints without mutating domain state."""
    text = raw_text.strip()
    lowered = text.lower()
    category_code = ""
    for candidate, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            category_code = candidate
            break

    return {
        "normalized_summary": text[:240],
        "category_code": category_code,
        "department_code": "",
        "priority": "medium",
        "confidence": confidence(0.6 if category_code else 0.0),
        "language": language_hint or "unknown",
        "provider": LOCAL_STUB_PROVIDER,
        "metadata": {"text_length": len(text)},
    }
