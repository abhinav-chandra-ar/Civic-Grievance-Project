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
    exclude_grievance_pk: int | None = None,
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
    exclude_grievance_pk
        PK of the grievance being enriched.  Passed to the duplicate-context
        selector so the grievance cannot match its own previously stored
        ``normalized_summary`` and produce a false self-duplicate on
        re-enrichment.
    """
    # ── AI block — wrapped for failure safety ───────────────────────────────
    try:
        recent_summaries = recent_grievance_summaries_for_duplicate_context(
            ward_code=ward_code,
            exclude_pk=exclude_grievance_pk,
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
            # Exclude this grievance's own PK from the duplicate-context pool
            # so its previously stored normalized_summary never self-matches
            # and triggers a false confirmed-duplicate on re-enrichment.
            exclude_grievance_pk=getattr(grievance, "pk", None),
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

        # Capture the priority that was on the grievance when the SLA was
        # originally created (i.e. before enrichment mutates it).  Used
        # below to decide whether the SLA deadline needs recomputing.
        _priority_before_enrichment: str = str(getattr(grievance, "priority", "medium"))

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

        from apps.grievances.services import (  # noqa: PLC0415
            DEFAULT_SLA_DEADLINE_DELTAS,
            update_grievance_enrichment,
        )

        update_grievance_enrichment(grievance=grievance, values=enrichment_values)

        # ── BUG 3 fix: recalculate SLA deadlines when AI changes priority ────
        # update_grievance_enrichment() has already written the new priority
        # to the Grievance row.  If it differs from what the SLA was created
        # with, recompute both deadlines so the SLA deadline matches reality.
        # Only active (non-breached) SLAs are updated; a breached SLA must
        # not have its deadline silently moved.
        _enriched_priority: str = str(enrichment_values.get("priority", "medium"))
        if _enriched_priority != _priority_before_enrichment:
            try:
                from apps.slas.services import update_sla_deadlines  # noqa: PLC0415

                _sla = grievance.sla  # OneToOne reverse — raises if absent
                if not _sla.is_breached:
                    _delta = DEFAULT_SLA_DEADLINE_DELTAS.get(
                        _enriched_priority,
                        DEFAULT_SLA_DEADLINE_DELTAS["medium"],
                    )
                    _new_due_at = grievance.submitted_at + _delta
                    update_sla_deadlines(
                        sla=_sla,
                        response_due_at=_new_due_at,
                        resolution_due_at=_new_due_at,
                        policy_snapshot_metadata={
                            "source":             "ai_enrichment_priority_update",
                            "original_priority":  _priority_before_enrichment,
                            "updated_priority":   _enriched_priority,
                        },
                    )
                    _logger.debug(
                        "SLA deadlines recomputed for grievance pk=%s "
                        "(priority %s → %s, new deadline %s)",
                        getattr(grievance, "pk", "?"),
                        _priority_before_enrichment,
                        _enriched_priority,
                        _new_due_at,
                    )
            except Exception as _sla_exc:  # noqa: BLE001
                _logger.warning(
                    "SLA priority recompute failed for grievance pk=%s: %s",
                    getattr(grievance, "pk", "?"),
                    _sla_exc,
                )

        # ── Post-enrichment lifecycle dispatch ─────────────────────────────
        # Move the grievance out of "submitted" into the correct actor queue.
        #
        # Lifecycle state rules (escalation is metadata, NOT a state):
        #   reject         → REJECTED          (high-confidence spam)
        #   dup confirmed  → DUPLICATE_FLAGGED  (confirmed duplicate)
        #   dept resolved  → ASSIGNED           (routable — auto or escalate)
        #   else           → TRIAGED            (uncertain; ward officer queue)
        #
        # If the AI also flagged escalation (should_escalate=True), record it
        # as a separate ESCALATION WorkflowEvent AFTER the lifecycle transition.
        # Escalation does not change status — it is an urgency alert that
        # municipal_admin can monitor via the escalation event stream.
        #
        # Wrapped in its own try/except so a dispatch failure NEVER rolls
        # back enrichment fields or prevents returning True.
        try:
            from django.contrib.auth import get_user_model as _get_user_model  # noqa: PLC0415

            from apps.grievances.models import GrievanceStatus  # noqa: PLC0415
            from apps.workflows.models import WorkflowTransitionType  # noqa: PLC0415
            from apps.workflows.services import (  # noqa: PLC0415
                escalate_grievance_from_system,
                transition_grievance,
            )

            _User = _get_user_model()
            _system_user, _ = _User.objects.get_or_create(
                username="__system__",
                defaults={
                    "role": "system_operator",
                    "is_active": True,
                    "is_staff": False,
                },
            )

            _automation_action: str = str(ai_decision.get("automation_action", ""))
            _escalation_dict: dict = dict(ai_decision.get("escalation") or {})
            _should_escalate: bool = bool(_escalation_dict.get("should_escalate", False))
            _dup_risk: dict = dict(ai_decision.get("duplicate_risk") or {})
            _dup_confirmed: bool = bool(_dup_risk.get("is_confirmed", False))
            # Department resolved by Phase E (None if Phase E failed or unresolved).
            _dept_instance = enrichment_values.get("department")
            # The enrichment metadata dict — threaded into every transition so
            # change_grievance_status() preserves AI fields instead of overwriting
            # status_metadata with an empty dict.
            _enrichment_status_metadata: dict = dict(
                enrichment_values.get("status_metadata") or {}
            )

            # ── Step A: lifecycle status transition ──────────────────────────
            if _automation_action == "reject":
                # High-confidence spam — reject immediately.
                transition_grievance(
                    grievance=grievance,
                    actor=_system_user,
                    new_status=GrievanceStatus.REJECTED,
                    transition_type=WorkflowTransitionType.STATUS_CHANGE,
                    transition_reason="AI classified as spam or invalid content.",
                    remarks="Post-enrichment dispatch: reject.",
                    status_metadata=_enrichment_status_metadata,
                )

            elif _dup_confirmed:
                # Confirmed duplicate — hold for deduplication review.
                transition_grievance(
                    grievance=grievance,
                    actor=_system_user,
                    new_status=GrievanceStatus.DUPLICATE_FLAGGED,
                    transition_type=WorkflowTransitionType.STATUS_CHANGE,
                    transition_reason="AI detected a confirmed duplicate submission.",
                    remarks="Post-enrichment dispatch: duplicate_flagged.",
                    status_metadata=_enrichment_status_metadata,
                )

            elif _dept_instance is not None:
                # Department resolved (auto_route OR escalate with known dept)
                # → enter the department officer queue.
                transition_grievance(
                    grievance=grievance,
                    actor=_system_user,
                    new_status=GrievanceStatus.ASSIGNED,
                    transition_type=WorkflowTransitionType.ASSIGNMENT,
                    transition_reason=(
                        f"AI routed to {_dept_instance.code} "
                        f"(confidence {ai_decision.get('routing_confidence', 0.0):.0%})."
                    ),
                    remarks="Post-enrichment dispatch: dept resolved → assigned.",
                    status_metadata=_enrichment_status_metadata,
                )

            else:
                # review_required, no department, low confidence
                # → enter the ward-officer triage queue.
                transition_grievance(
                    grievance=grievance,
                    actor=_system_user,
                    new_status=GrievanceStatus.TRIAGED,
                    transition_type=WorkflowTransitionType.STATUS_CHANGE,
                    transition_reason="AI enrichment complete; awaiting human triage.",
                    remarks="Post-enrichment dispatch: uncertain → triaged.",
                    status_metadata=_enrichment_status_metadata,
                )

            # ── Step B: escalation alert (separate from lifecycle) ───────────
            # Records an ESCALATION WorkflowEvent with previous_status ==
            # new_status (the status just set above).  This does NOT change
            # the grievance status — it only fires the municipal-admin alert.
            if _should_escalate:
                _esc_reason: str = (
                    _escalation_dict.get("escalation_reason", "")
                    or "AI decision engine flagged for escalation."
                )
                escalate_grievance_from_system(
                    grievance=grievance,
                    transition_reason=_esc_reason,
                    escalation_metadata={
                        "source": "ai_enrichment",
                        "automation_action": _automation_action,
                        "escalation_reason": _esc_reason,
                        "routing_confidence": ai_decision.get("routing_confidence", 0.0),
                        "priority": enrichment_values.get("priority", ""),
                        "lifecycle_status": grievance.status,
                    },
                )

            _logger.debug(
                "Post-enrichment dispatch: pk=%s → status=%s escalation=%s",
                getattr(grievance, "pk", "?"),
                grievance.status,
                _should_escalate,
            )
        except Exception as _dispatch_exc:  # noqa: BLE001
            _logger.warning(
                "Post-enrichment status dispatch failed for grievance pk=%s: %s",
                getattr(grievance, "pk", "?"),
                _dispatch_exc,
            )

        return True

    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "Enrichment write failed for grievance pk=%s: %s",
            getattr(grievance, "pk", "?"),
            exc,
        )
        return False


def analyze_attachment_image(
    *,
    storage_reference: str,
    content_type: str,
    content_hash: str | None = None,
    image_bytes: bytes | None = None,
    text_category: str = "",
    raw_text: str = "",
) -> dict[str, object]:
    """Return attachment-ready image validation metadata without domain writes.

    When ``image_bytes`` is supplied, the real Pillow + CLIP pipeline runs.
    Otherwise falls back to metadata-only validation (content_type header check).
    """
    result = validate_grievance_image(
        storage_reference=storage_reference,
        content_type=content_type,
        content_hash=content_hash,
        image_bytes=image_bytes,
        text_category=text_category,
        raw_text=raw_text,
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
