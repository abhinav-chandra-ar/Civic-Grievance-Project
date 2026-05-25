"""Signal receivers for the attachments app.

``analyze_new_attachment`` fires synchronously when ``attachment_registered``
is sent.  It reads the binary bytes from the saved FileField, runs the full
Pillow + CLIP image-analysis pipeline, and writes results to the three
analysis metadata fields on the Attachment row.

A compact ``vision_analysis`` summary is also merged into the parent
Grievance's ``status_metadata`` so officers and admins can see the AI verdict
without joining the attachments table.

Safety contract
---------------
* Never raises — all failures are caught and logged as WARNING.
* If CLIP is unavailable, Pillow heuristics alone populate the metadata.
* If Pillow is also unavailable, a safe "unavailable" payload is stored.
* Grievance enrichment is never blocked by image analysis failure.
"""
from __future__ import annotations

import logging

from django.dispatch import receiver

from .models import Attachment
from .signals import attachment_registered

_logger = logging.getLogger(__name__)


@receiver(attachment_registered, sender=Attachment)
def analyze_new_attachment(
    sender: type[Attachment],
    attachment: Attachment,
    **kwargs: object,
) -> None:
    """Read image bytes → Pillow + CLIP analysis → save verdict to DB.

    Triggered by ``register_attachment_with_file()`` after the FileField has
    been written to MEDIA_ROOT.  Silently skips if ``image_file`` is absent
    (metadata-only registrations via external storage).
    """
    # ── Guard: only process direct-upload attachments ───────────────────────
    if not attachment.image_file:
        return

    # ── 1. Read bytes from disk ──────────────────────────────────────────────
    try:
        attachment.image_file.seek(0)
        image_bytes: bytes = attachment.image_file.read()
        attachment.image_file.close()
    except Exception as exc:
        _logger.warning(
            "analyze_new_attachment: cannot read image_file for %s: %s",
            attachment.attachment_code,
            exc,
        )
        return

    if not image_bytes:
        _logger.warning(
            "analyze_new_attachment: empty image_file for %s",
            attachment.attachment_code,
        )
        return

    # ── 2. Gather complaint context for consistency check ───────────────────
    try:
        grievance = attachment.grievance
        category_code: str = str(getattr(grievance, "category_code", "") or "")
        raw_text: str = str(getattr(grievance, "raw_text", "") or "")
    except Exception:
        category_code = ""
        raw_text = ""

    # ── 3. Run Pillow + CLIP analysis ────────────────────────────────────────
    try:
        from apps.ml.image_analyzer import analyze_image  # noqa: PLC0415

        result = analyze_image(image_bytes, text_category=category_code, text=raw_text)
        _logger.debug(
            "analyze_new_attachment: %s vision_class=%s verdict=%s provider=%s",
            attachment.attachment_code,
            result.get("vision_class"),
            result.get("consistency_verdict"),
            result.get("vision_provider"),
        )
    except Exception as exc:
        _logger.warning(
            "analyze_new_attachment: image_analyzer failed for %s: %s",
            attachment.attachment_code,
            exc,
        )
        result = {
            "is_valid": False,
            "quality_score": 0.0,
            "quality_flags": ["analysis_failed"],
            "usable": False,
            "is_irrelevant": False,
            "irrelevant_reason": "",
            "is_consistent": False,
            "consistency_score": 0.0,
            "conflict_reason": str(exc),
            "provider": "unavailable",
            "vision_class": None,
            "vision_confidence": None,
            "vision_all_scores": None,
            "consistency_verdict": "uncertain",
            "fraud_flags": [],
            "vision_provider": "unavailable",
        }

    # ── 4. Map analysis result to the three Attachment metadata fields ───────
    image_validation_metadata = {
        "is_valid":          result.get("is_valid", False),
        "quality_score":     result.get("quality_score", 0.0),
        "quality_flags":     result.get("quality_flags", []),
        "usable":            result.get("usable", False),
        "is_irrelevant":     result.get("is_irrelevant", False),
        "irrelevant_reason": result.get("irrelevant_reason", ""),
        "provider":          result.get("provider", "image_rule_v1"),
    }

    image_issue_classification_metadata = {
        "vision_class":      result.get("vision_class"),
        "vision_confidence": result.get("vision_confidence"),
        "vision_all_scores": result.get("vision_all_scores"),
        "is_civic_image":    result.get("vision_class") not in (
            None, "irrelevant_image", "indoor_irrelevant", "screenshot", "poor_quality"
        ),
        "vision_provider":   result.get("vision_provider", "unavailable"),
        "fraud_flags":       result.get("fraud_flags", []),
    }

    image_text_consistency_metadata = {
        "consistency_verdict": result.get("consistency_verdict", "uncertain"),
        "is_consistent":       result.get("is_consistent", False),
        "consistency_score":   result.get("consistency_score", 0.0),
        "conflict_reason":     result.get("conflict_reason", ""),
    }

    # ── 5. Persist the three metadata blocks ─────────────────────────────────
    try:
        from .services import update_attachment_metadata  # noqa: PLC0415

        update_attachment_metadata(
            attachment=attachment,
            values={
                "image_validation_metadata":           image_validation_metadata,
                "image_issue_classification_metadata": image_issue_classification_metadata,
                "image_text_consistency_metadata":     image_text_consistency_metadata,
            },
        )
    except Exception as exc:
        _logger.warning(
            "analyze_new_attachment: metadata update failed for %s: %s",
            attachment.attachment_code,
            exc,
        )
        return

    # ── 6. Write compact vision_analysis summary into Grievance.status_metadata
    try:
        from apps.grievances.services import update_grievance_enrichment  # noqa: PLC0415

        existing_sm: dict = dict(getattr(grievance, "status_metadata", {}) or {})
        existing_sm["vision_analysis"] = {
            "vision_class":        result.get("vision_class"),
            "confidence":          result.get("vision_confidence"),
            "consistency_verdict": result.get("consistency_verdict", "uncertain"),
            "provider":            result.get("vision_provider", "unavailable"),
            "attachment_code":     attachment.attachment_code,
        }
        # Reload the grievance to avoid overwriting concurrent enrichment changes
        grievance.refresh_from_db()
        existing_sm_fresh: dict = dict(getattr(grievance, "status_metadata", {}) or {})
        existing_sm_fresh["vision_analysis"] = existing_sm["vision_analysis"]
        update_grievance_enrichment(
            grievance=grievance,
            values={"status_metadata": existing_sm_fresh},
        )
        _logger.info(
            "analyze_new_attachment: vision_analysis written to grievance pk=%s "
            "(attachment=%s, verdict=%s)",
            grievance.pk,
            attachment.attachment_code,
            result.get("consistency_verdict"),
        )
    except Exception as exc:
        _logger.warning(
            "analyze_new_attachment: grievance status_metadata update failed "
            "for %s: %s",
            attachment.attachment_code,
            exc,
        )
