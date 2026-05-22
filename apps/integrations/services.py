"""Orchestration services for external intelligence hooks."""
from __future__ import annotations

from collections.abc import Sequence

from .clients.duplicates import detect_possible_duplicates
from .clients.images import validate_grievance_image
from .clients.landmarks import resolve_landmark_mention
from .clients.nlp import classify_grievance_text
from .selectors import local_landmark_candidates_for_mention
from .signals import integration_call_completed


def analyze_grievance_submission(
    *,
    raw_text: str,
    landmark_mention: str = "",
    citizen_location_text: str = "",
    content_hashes: Sequence[str] | None = None,
    language_hint: str | None = None,
) -> dict[str, object]:
    """Return a grievance enrichment payload without mutating domain models."""
    nlp_result = classify_grievance_text(raw_text=raw_text, language_hint=language_hint)
    landmark_result = resolve_landmark_mention(
        mention=landmark_mention,
        location_text=citizen_location_text,
    )
    local_candidates = local_landmark_candidates_for_mention(mention=landmark_mention)
    landmark_metadata = dict(landmark_result["metadata"])
    landmark_metadata["local_candidates"] = local_candidates
    duplicate_result = detect_possible_duplicates(
        raw_text=raw_text,
        category_code=str(nlp_result["category_code"]),
        landmark_code=landmark_result["landmark_code"],
        content_hashes=list(content_hashes or []),
    )

    payload = {
        "normalized_summary": nlp_result["normalized_summary"],
        "category_code": nlp_result["category_code"],
        "priority": nlp_result["priority"],
        "landmark_resolution_metadata": {
            "provider_result": landmark_result,
            "local_candidates": local_candidates,
        },
        "duplicate_detection_metadata": duplicate_result,
        "provider_metadata": {
            "nlp": {
                "provider": nlp_result["provider"],
                "confidence": nlp_result["confidence"],
                "language": nlp_result["language"],
                "metadata": nlp_result["metadata"],
            },
            "landmark": {
                "provider": landmark_result["provider"],
                "confidence": landmark_result["confidence"],
                "metadata": landmark_metadata,
            },
            "duplicate": {
                "provider": duplicate_result["provider"],
                "confidence": duplicate_result["confidence"],
                "metadata": duplicate_result["metadata"],
            },
        },
    }
    integration_call_completed.send(sender=analyze_grievance_submission, payload=payload)
    return payload


def analyze_attachment_image(
    *, storage_reference: str, content_type: str, content_hash: str | None = None
) -> dict[str, object]:
    """Return attachment-ready image validation metadata without domain writes."""
    result = validate_grievance_image(
        storage_reference=storage_reference,
        content_type=content_type,
        content_hash=content_hash,
    )
    payload = {
        "image_validation_metadata": {
            "is_valid": result["is_valid"],
            "provider": result["provider"],
            "metadata": result["metadata"],
        },
        "image_issue_classification_metadata": result["issue_classification"],
        "image_text_consistency_metadata": result["text_consistency"],
        "moderation_metadata": {
            "status": result["moderation_status"],
            "provider": result["provider"],
        },
    }
    integration_call_completed.send(sender=analyze_attachment_image, payload=payload)
    return payload
