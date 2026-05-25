"""Image validation client hook.

Two paths
---------
real_bytes path
    When ``image_bytes`` is supplied the function delegates to
    ``apps.ml.image_analyzer.analyze_image()`` which runs Pillow heuristics
    and CLIP zero-shot classification.  This is the path used by the
    attachment upload pipeline (see ``apps/attachments/receivers.py``).

metadata-only path (legacy / external-storage)
    When only ``storage_reference`` / ``content_type`` / ``content_hash``
    are provided (no bytes), the function falls back to a lightweight
    header-only validation that checks the MIME type string.  This preserves
    backward compatibility with external-storage registrations.
"""
from __future__ import annotations

import logging

from .base import LOCAL_STUB_PROVIDER

_logger = logging.getLogger(__name__)


def validate_grievance_image(
    *,
    storage_reference: str,
    content_type: str,
    content_hash: str | None = None,
    image_bytes: bytes | None = None,
    text_category: str = "",
    raw_text: str = "",
) -> dict[str, object]:
    """Return an image-validation payload, optionally running real ML analysis.

    Parameters
    ----------
    storage_reference
        File path (relative to MEDIA_ROOT) or external URL.  Used only when
        ``image_bytes`` is absent.
    content_type
        Image MIME type declared by the client.
    content_hash
        Optional SHA-256 hex digest for integrity verification.
    image_bytes
        Raw binary image data.  When provided, Pillow + CLIP analysis is run
        on the actual pixels; the metadata-only fallback is bypassed entirely.
    text_category
        Grievance category code used for the CLIP consistency check
        (e.g. ``"electrical_hazard"``).
    raw_text
        Raw complaint text forwarded to CLIP for text-image cosine similarity.

    Returns
    -------
    dict matching the shape expected by ``analyze_attachment_image()``.
    """
    normalized_content_type = content_type.strip().lower()

    # ── Real binary analysis ─────────────────────────────────────────────────
    if image_bytes:
        try:
            from apps.ml.image_analyzer import analyze_image  # noqa: PLC0415

            result = analyze_image(image_bytes, text_category=text_category, text=raw_text)
            return {
                "is_valid":               result.get("is_valid", False),
                "moderation_status":      "cleared",
                "issue_classification":   {
                    "vision_class":      result.get("vision_class"),
                    "vision_confidence": result.get("vision_confidence"),
                    "vision_all_scores": result.get("vision_all_scores"),
                    "vision_provider":   result.get("vision_provider", "unavailable"),
                },
                "text_consistency":       {
                    "consistency_verdict": result.get("consistency_verdict", "uncertain"),
                    "is_consistent":       result.get("is_consistent", False),
                    "consistency_score":   result.get("consistency_score", 0.0),
                    "conflict_reason":     result.get("conflict_reason", ""),
                },
                "provider": result.get("vision_provider", "clip_vit_b32"),
                "metadata": {
                    "storage_reference": storage_reference,
                    "content_type":      normalized_content_type,
                    "content_hash":      content_hash,
                    "quality_score":     result.get("quality_score", 0.0),
                    "quality_flags":     result.get("quality_flags", []),
                    "usable":            result.get("usable", False),
                    "fraud_flags":       result.get("fraud_flags", []),
                },
            }
        except Exception as exc:
            _logger.warning(
                "validate_grievance_image: real analysis failed (%s); "
                "falling back to metadata-only validation",
                exc,
            )

    # ── Metadata-only fallback (external storage / analysis unavailable) ─────
    return {
        "is_valid":             normalized_content_type.startswith("image/") and bool(storage_reference.strip()),
        "moderation_status":    "pending",
        "issue_classification": {},
        "text_consistency":     {},
        "provider":             LOCAL_STUB_PROVIDER,
        "metadata": {
            "storage_reference": storage_reference.strip(),
            "content_type":      normalized_content_type,
            "content_hash":      content_hash,
        },
    }
