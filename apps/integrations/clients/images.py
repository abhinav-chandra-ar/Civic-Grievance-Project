"""Image validation client hook."""
from __future__ import annotations

from .base import LOCAL_STUB_PROVIDER


def validate_grievance_image(
    *, storage_reference: str, content_type: str, content_hash: str | None = None
) -> dict[str, object]:
    """Return image validation hooks without reading or mutating stored files."""
    normalized_content_type = content_type.strip().lower()
    return {
        "is_valid": normalized_content_type.startswith("image/") and bool(storage_reference.strip()),
        "moderation_status": "pending",
        "issue_classification": {},
        "text_consistency": {},
        "provider": LOCAL_STUB_PROVIDER,
        "metadata": {
            "storage_reference": storage_reference.strip(),
            "content_type": normalized_content_type,
            "content_hash": content_hash,
        },
    }
