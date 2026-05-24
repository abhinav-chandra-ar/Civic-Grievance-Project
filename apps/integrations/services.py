"""Orchestration services for external intelligence hooks."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .clients.duplicates import detect_possible_duplicates
from .clients.images import validate_grievance_image
from .clients.landmarks import resolve_landmark_mention
from .clients.nlp import classify_grievance_text
from .selectors import (
    local_landmark_candidates_for_mention,
    recent_grievance_summaries_for_duplicate_context,
)
from .signals import integration_call_completed

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FALLBACK_PROVIDER = "local_ml_v1"


def _nlp_fallback(raw_text: str) -> dict[str, object]:
    """Return a safe minimum-viable NLP payload when the AI pipeline fails.

    The payload is shaped identically to what ``classify_grievance_text()``
    returns so the rest of ``analyze_grievance_submission()`` can continue
    without branching.  The decision action is ``review_required`` with
    ``"ai_pipeline_error"`` as the sole review reason, ensuring the
    submission is not silently auto-routed.
    """
    text = raw_text.strip()
    return {
        "normalized_summary": text[:240],
        "category_code":      "",
        "department_code":    "",
        "priority":           "medium",
        "confidence":         0.0,
        "language":           "unknown",
        "provider":           _FALLBACK_PROVIDER,
        "metadata": {
            "text_length":            len(text),
            "ward_hint":              None,
            "landmark_hints":         [],
            "spam_check":             {"is_spam": False, "spam_score": 0.0, "spam_reason": ""},
            "duplicate_check":        {"is_duplicate": False, "similarity_score": 0.0, "matching_text": None},
            "needs_human_review":     True,
            "review_reasons":         ["ai_pipeline_error"],
            "image_analysis":         None,
            "consistency_check":      None,
            "evidence_quality":       None,
            "evidence_review_reason": "",
            "decision": {
                "automation_action":  "review_required",
                "routing_confidence": 0.0,
                "needs_review":       True,
                "review_reasons":     ["ai_pipeline_error"],
                "duplicate_risk":     {"risk_level": "low", "risk_score": 0.0, "is_confirmed": False},
                "escalation":         {"should_escalate": False, "escalation_reason": ""},
                "decision_metadata":  {},
            },
        },
    }


def _build_explainability_reason(nlp_result: dict[str, Any], ai_decision: dict[str, Any]) -> str:
    """Return a single human-readable sentence explaining the AI routing decision.

    Used to populate ``ai_explainability`` in the enrichment payload and
    ``status_reason`` on the grievance record.
    """
    action = ai_decision.get("automation_action", "")
    confidence = float(nlp_result.get("confidence", 0.0))

    if action == "reject":
        return "Submission rejected by AI: likely spam or invalid content."

    if action == "escalate":
        reason = (ai_decision.get("escalation") or {}).get("escalation_reason", "")
        if reason:
            return f"Escalated by AI: {reason}."
        return "Escalated by AI due to high priority or life-safety concern."

    if action == "review_required":
        reasons: list[str] = ai_decision.get("review_reasons") or []
        reason_str = ", ".join(reasons[:3]) if reasons else "low confidence"
        return f"Human review required: {reason_str}."

    if action == "auto_route":
        return f"Auto-routed by AI with {confidence:.0%} confidence."

    return ""


# ---------------------------------------------------------------------------
# Public orchestration services
# ---------------------------------------------------------------------------


def analyze_grievance_submission(
    *,
    raw_text: str,
    landmark_mention: str = "",
    citizen_location_text: str = "",
    content_hashes: Sequence[str] | None = None,
    language_hint: str | None = None,
    image_input: object = None,
    ward_code: str | None = None,
) -> dict[str, object]:
    """Return a grievance enrichment payload without mutating domain models.

    The AI analysis block (recent-summary lookup + NLP classification) is
    wrapped in a try/except so that any ML-pipeline failure degrades
    gracefully to a safe fallback payload.  The complaint is **never**
    silently auto-routed when AI fails — the fallback decision is always
    ``review_required`` with reason ``ai_pipeline_error``.

    Parameters
    ----------
    raw_text
        The citizen's original complaint text.
    landmark_mention
        Optional extracted landmark mention from the submission form.
    citizen_location_text
        Free-text location description from the citizen.
    content_hashes
        Optional list of content hashes for attachment-level duplicate detection.
    language_hint
        Optional caller-supplied language override forwarded to the NLP engine.
    image_input
        Optional image evidence (file path, bytes, or PIL Image) for Phase B
        image intelligence.
    ward_code
        TVMC ward code (e.g. ``"tvm_034"``).  When provided, the duplicate
        context query is scoped to the same ward, improving relevance.
    """
    # ── AI block — wrapped for failure safety ───────────────────────────────
    try:
        recent_summaries = recent_grievance_summaries_for_duplicate_context(
            ward_code=ward_code
        )
        nlp_result = classify_grievance_text(
            raw_text=raw_text,
            language_hint=language_hint,
            image_input=image_input,
            recent_texts=recent_summaries,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "AI pipeline failed for grievance submission, using fallback payload: %s",
            exc,
        )
        nlp_result = _nlp_fallback(raw_text)

    # ── Non-AI enrichment (landmark resolution, local candidates, stub dup) ─
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

    # ── Surface Phase C decision at the top level ────────────────────────────
    ai_decision: dict[str, Any] = nlp_result["metadata"].get("decision") or {}  # type: ignore[assignment]
    ai_explainability = _build_explainability_reason(nlp_result, ai_decision)

    payload: dict[str, object] = {
        "normalized_summary": nlp_result["normalized_summary"],
        "category_code":      nlp_result["category_code"],
        "priority":           nlp_result["priority"],
        "landmark_resolution_metadata": {
            "provider_result":  landmark_result,
            "local_candidates": local_candidates,
        },
        "duplicate_detection_metadata": duplicate_result,
        # Phase D — AI decision surfaced at the top level
        "ai_decision":      ai_decision,
        "ai_explainability": ai_explainability,
        # Phase E — routing context: ML hints surfaced for FK resolution.
        # DB lookups (Ward, Department) are deferred to enrich_grievance_with_ai()
        # so this function remains free of DB side-effects.
        "routing_context": {
            "ward_hint":       nlp_result["metadata"]["ward_hint"],
            "landmark_hints":  nlp_result["metadata"]["landmark_hints"],
            "category_code":   str(nlp_result["category_code"]),
            "department_code": str(nlp_result["department_code"]),
        },
        # Backward-compatible provider metadata (unchanged shape)
        "provider_metadata": {
            "nlp": {
                "provider":   nlp_result["provider"],
                "confidence": nlp_result["confidence"],
                "language":   nlp_result["language"],
                "metadata":   nlp_result["metadata"],
            },
            "landmark": {
                "provider":   landmark_result["provider"],
                "confidence": landmark_result["confidence"],
                "metadata":   landmark_metadata,
            },
            "duplicate": {
                "provider":   duplicate_result["provider"],
                "confidence": duplicate_result["confidence"],
                "metadata":   duplicate_result["metadata"],
            },
        },
    }
    integration_call_completed.send(sender=analyze_grievance_submission, payload=payload)
    return payload


def enrich_grievance_with_ai(
    *,
    grievance: Any,
    image_input: object = None,
) -> bool:
    """Apply AI enrichment to an existing grievance record.

    Calls ``analyze_grievance_submission()`` with the grievance's own text and
    maps the AI results to ``update_grievance_enrichment()``.  The Phase C
    routing decision is stored in ``status_metadata`` so downstream workflow
    transitions can act on it without re-running the AI.

    This function **never raises**.  Any failure — whether in the AI pipeline
    or in the DB write — is caught and logged; the grievance is left in its
    pre-enrichment state and ``False`` is returned.

    Parameters
    ----------
    grievance
        A :class:`apps.grievances.models.Grievance` instance.
    image_input
        Optional image evidence forwarded to Phase B analysis.

    Returns
    -------
    bool
        ``True`` if enrichment was applied successfully, ``False`` otherwise.
    """
    try:
        payload = analyze_grievance_submission(
            raw_text=grievance.raw_text,
            landmark_mention=getattr(grievance, "landmark_mention", ""),
            citizen_location_text=getattr(grievance, "citizen_location_text", ""),
            image_input=image_input,
            ward_code=None,  # ward resolved from landmarks after enrichment
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "AI analysis failed for grievance pk=%s: %s",
            getattr(grievance, "pk", "?"),
            exc,
        )
        return False

    try:
        ai_decision: dict[str, Any] = payload.get("ai_decision") or {}

        enrichment_values: dict[str, Any] = {
            "normalized_summary":           payload.get("normalized_summary", ""),
            "category_code":                payload.get("category_code", ""),
            "priority":                     payload.get("priority", "medium"),
            "landmark_resolution_metadata": payload.get("landmark_resolution_metadata", {}),
            "duplicate_detection_metadata": payload.get("duplicate_detection_metadata", {}),
            "status_reason":                payload.get("ai_explainability", ""),
            "status_metadata": {
                "ai_enrichment":      True,
                "automation_action":  ai_decision.get("automation_action", ""),
                "routing_confidence": ai_decision.get("routing_confidence", 0.0),
                "needs_review":       ai_decision.get("needs_review", False),
                "review_reasons":     ai_decision.get("review_reasons", []),
                "duplicate_risk":     ai_decision.get("duplicate_risk", {}),
                "escalation":         ai_decision.get("escalation", {}),
                "provider": (
                    (payload.get("provider_metadata") or {})
                    .get("nlp", {})
                    .get("provider", "")
                ),
            },
        }

        # Phase E — KSMART-style ward/department/queue resolution.
        # Wrapped in its own try/except so a routing failure (e.g. ward
        # not yet seeded in DB) is logged as a warning but never blocks
        # the base enrichment from completing.
        try:
            from apps.integrations.routing import build_phase_e_routing  # noqa: PLC0415

            routing_context = payload.get("routing_context") or {}
            phase_e = build_phase_e_routing(
                routing_context=routing_context,
                ai_decision=ai_decision,
            )
            enrichment_values["ward"]       = phase_e["ward_instance"]
            enrichment_values["department"] = phase_e["department_instance"]
            enrichment_values["status_metadata"]["phase_e_routing"] = (
                phase_e["routing_metadata"]
            )
        except Exception as _phase_e_exc:  # noqa: BLE001
            _logger.warning(
                "Phase E routing failed for grievance pk=%s: %s",
                getattr(grievance, "pk", "?"),
                _phase_e_exc,
            )

        from apps.grievances.services import update_grievance_enrichment  # noqa: PLC0415

        update_grievance_enrichment(grievance=grievance, values=enrichment_values)
        return True

    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "Enrichment write failed for grievance pk=%s: %s",
            getattr(grievance, "pk", "?"),
            exc,
        )
        return False


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
