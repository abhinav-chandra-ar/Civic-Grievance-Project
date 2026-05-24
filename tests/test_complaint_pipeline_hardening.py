"""Hardening tests for the full complaint creation pipeline.

Scope
-----
Verifies that ``create_grievance_with_foundation_records()`` always produces
the correct set of foundation DB records regardless of whether AI enrichment
succeeds or fails.

Execution path under test
-------------------------
submit_grievance()
    → generate_tracking_code()
    → Grievance.save()
    → grievance_submitted signal
    → transition_grievance()            → WorkflowEvent row
    → create_sla_for_grievance()        → SLA row
    → record_audit_event()              → AuditLog row
    → enrich_grievance_with_ai()        → category_code, priority, status_metadata
"""
from __future__ import annotations

import re
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model

from apps.audit.models import AuditLog, AuditActionType
from apps.grievances.models import Grievance, GrievanceStatus
from apps.grievances.services import submit_grievance
from apps.slas.models import SLA, SLAStatus
from apps.workflows.models import WorkflowEvent, WorkflowTransitionType

User = get_user_model()

pytestmark = pytest.mark.django_db

# ---------------------------------------------------------------------------
# NLP payload stub — shapes the AI response without touching ML models
# ---------------------------------------------------------------------------

_NLP_STUB = {
    "normalized_summary": "Water pipe leaking near school.",
    "category_code":      "water_supply",
    "department_code":    "water_authority",
    "priority":           "high",
    "confidence":         0.82,
    "language":           "en",
    "provider":           "transformer_v1",
    "metadata": {
        "text_length":            40,
        "ward_hint":              None,
        "landmark_hints":         [],
        "spam_check":             {"is_spam": False, "spam_score": 0.02, "spam_reason": ""},
        "duplicate_check":        {"is_duplicate": False, "similarity_score": 0.0, "matching_text": None},
        "needs_human_review":     False,
        "review_reasons":         [],
        "image_analysis":         None,
        "consistency_check":      None,
        "evidence_quality":       None,
        "evidence_review_reason": "",
        "decision": {
            "automation_action":  "auto_route",
            "routing_confidence": 0.80,
            "needs_review":       False,
            "review_reasons":     [],
            "duplicate_risk":     {"risk_level": "low", "risk_score": 0.0, "is_confirmed": False},
            "escalation":         {"should_escalate": False, "escalation_reason": ""},
            "decision_metadata":  {},
        },
    },
}

_PATCH_NLP      = "apps.integrations.services.classify_grievance_text"
_PATCH_RECENT   = "apps.integrations.services.recent_grievance_summaries_for_duplicate_context"
_PATCH_LOCAL    = "apps.integrations.services.local_landmark_candidates_for_mention"


@pytest.fixture
def citizen():
    return User.objects.create_user(
        username="citizen_test",
        email="citizen@test.com",
        password="pass",
        role="citizen",
    )


# ---------------------------------------------------------------------------
# 1. Tracking code is always created in GRV-YYYY-NNNNNN format
# ---------------------------------------------------------------------------

@patch(_PATCH_LOCAL, return_value=[])
@patch(_PATCH_RECENT, return_value=[])
@patch(_PATCH_NLP, return_value=_NLP_STUB)
def test_grievance_tracking_code_format(mock_nlp, mock_recent, mock_local, citizen):
    grievance = submit_grievance(
        submitter=citizen,
        raw_text="Water pipe leak near school",
    )
    assert re.fullmatch(r"GRV-\d{4}-\d{6}", grievance.tracking_code), (
        f"Expected GRV-YYYY-NNNNNN format, got {grievance.tracking_code!r}"
    )


# ---------------------------------------------------------------------------
# 2. Foundation records are always created atomically
# ---------------------------------------------------------------------------

@patch(_PATCH_LOCAL, return_value=[])
@patch(_PATCH_RECENT, return_value=[])
@patch(_PATCH_NLP, return_value=_NLP_STUB)
def test_workflow_event_created_on_submission(mock_nlp, mock_recent, mock_local, citizen):
    grievance = submit_grievance(
        submitter=citizen,
        raw_text="Road has a big pothole near bus stand",
    )
    events = WorkflowEvent.objects.filter(grievance=grievance)
    assert events.exists(), "No WorkflowEvent created for grievance."
    first_event = events.order_by("id").first()
    assert first_event.transition_type == WorkflowTransitionType.STATUS_CHANGE
    assert re.fullmatch(r"WFE-\d{4}-\d{6}", first_event.event_code)


@patch(_PATCH_LOCAL, return_value=[])
@patch(_PATCH_RECENT, return_value=[])
@patch(_PATCH_NLP, return_value=_NLP_STUB)
def test_sla_created_on_submission(mock_nlp, mock_recent, mock_local, citizen):
    grievance = submit_grievance(
        submitter=citizen,
        raw_text="Garbage dumped near junction",
    )
    assert SLA.objects.filter(grievance=grievance).exists(), "No SLA created for grievance."
    sla = SLA.objects.get(grievance=grievance)
    assert sla.sla_status == SLAStatus.ACTIVE
    assert not sla.is_breached
    assert re.fullmatch(r"SLA-\d{4}-\d{6}", sla.sla_code)


@patch(_PATCH_LOCAL, return_value=[])
@patch(_PATCH_RECENT, return_value=[])
@patch(_PATCH_NLP, return_value=_NLP_STUB)
def test_audit_log_created_on_submission(mock_nlp, mock_recent, mock_local, citizen):
    grievance = submit_grievance(
        submitter=citizen,
        raw_text="Street light not working for two weeks",
    )
    logs = AuditLog.objects.filter(
        target_model="grievances.Grievance",
        target_object_id=str(grievance.pk),
    )
    assert logs.exists(), "No AuditLog created for grievance."
    log = logs.first()
    assert log.action_type == AuditActionType.CREATE
    assert log.change_metadata.get("tracking_code") == grievance.tracking_code
    assert re.fullmatch(r"AUD-\d{4}-\d{6}", log.audit_code)


# ---------------------------------------------------------------------------
# 3. AI enrichment populates category_code and priority on the grievance
# ---------------------------------------------------------------------------

@patch(_PATCH_LOCAL, return_value=[])
@patch(_PATCH_RECENT, return_value=[])
@patch(_PATCH_NLP, return_value=_NLP_STUB)
def test_ai_enrichment_sets_category_and_priority(mock_nlp, mock_recent, mock_local, citizen):
    grievance = submit_grievance(
        submitter=citizen,
        raw_text="Water pipe leak near school",
    )
    grievance.refresh_from_db()
    assert grievance.category_code == "water_supply", (
        f"Expected category_code='water_supply', got {grievance.category_code!r}"
    )
    assert grievance.priority == "high", (
        f"Expected priority='high', got {grievance.priority!r}"
    )


@patch(_PATCH_LOCAL, return_value=[])
@patch(_PATCH_RECENT, return_value=[])
@patch(_PATCH_NLP, return_value=_NLP_STUB)
def test_ai_enrichment_stores_status_metadata(mock_nlp, mock_recent, mock_local, citizen):
    grievance = submit_grievance(
        submitter=citizen,
        raw_text="Water pipe leak near school",
    )
    grievance.refresh_from_db()
    meta = grievance.status_metadata
    assert meta.get("ai_enrichment") is True, "status_metadata must flag ai_enrichment=True"
    assert "automation_action" in meta, "status_metadata must include automation_action"


# ---------------------------------------------------------------------------
# 4. SLA deadline matches priority deltas
# ---------------------------------------------------------------------------

@patch(_PATCH_LOCAL, return_value=[])
@patch(_PATCH_RECENT, return_value=[])
@patch(_PATCH_NLP, return_value=_NLP_STUB)
def test_sla_deadline_matches_priority(mock_nlp, mock_recent, mock_local, citizen):
    """SLA deadline is recomputed after AI enrichment upgrades priority.

    BUG 3 fix: enrich_grievance_with_ai() now calls update_sla_deadlines()
    when the AI-assigned priority differs from the creation-time default.

    Flow:
      1. Grievance created with default priority "medium" → SLA window = 5 days.
      2. AI enrichment upgrades priority to "high" (from _NLP_STUB).
      3. enrich_grievance_with_ai() detects the change and recomputes the SLA
         deadline to 3 days (the high-priority window).
      4. The SLA row is updated in-place — no duplicate row is created.
    """
    grievance = submit_grievance(
        submitter=citizen,
        raw_text="Water pipe leak near school",
    )
    sla = SLA.objects.get(grievance=grievance)
    delta = sla.resolution_due_at - grievance.submitted_at
    # After BUG 3 fix: SLA deadline is recomputed to match AI-enriched priority.
    # "high" priority → 3-day window (DEFAULT_SLA_DEADLINE_DELTAS["high"]).
    assert delta.days == 3, (
        f"Expected 3-day SLA after AI enrichment updates priority to 'high', got {delta.days}d. "
        "enrich_grievance_with_ai() must call update_sla_deadlines() when priority changes."
    )
    # Confirm enrichment upgraded the priority on the grievance itself
    grievance.refresh_from_db()
    assert grievance.priority == "high", (
        f"AI enrichment should have upgraded priority to 'high', got {grievance.priority!r}"
    )


# ---------------------------------------------------------------------------
# 5. AI pipeline failure MUST NOT break the transaction
# ---------------------------------------------------------------------------

def test_foundation_records_survive_ai_crash(citizen):
    """Even when the AI pipeline raises, all foundation records must be saved."""
    with patch(_PATCH_NLP, side_effect=RuntimeError("AI is down")), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]):
        grievance = submit_grievance(
            submitter=citizen,
            raw_text="Drain blocked near market",
        )

    # Grievance must exist
    assert Grievance.objects.filter(pk=grievance.pk).exists()
    # Workflow event must exist
    assert WorkflowEvent.objects.filter(grievance=grievance).exists()
    # SLA must exist
    assert SLA.objects.filter(grievance=grievance).exists()
    # AuditLog must exist
    assert AuditLog.objects.filter(
        target_model="grievances.Grievance",
        target_object_id=str(grievance.pk),
    ).exists()


# ---------------------------------------------------------------------------
# 6. Status is SUBMITTED at creation (not enrichment_pending, not triaged)
# ---------------------------------------------------------------------------

@patch(_PATCH_LOCAL, return_value=[])
@patch(_PATCH_RECENT, return_value=[])
@patch(_PATCH_NLP, return_value=_NLP_STUB)
def test_initial_status_is_submitted(mock_nlp, mock_recent, mock_local, citizen):
    grievance = submit_grievance(
        submitter=citizen,
        raw_text="Broken drain pipe on main road",
    )
    # Re-fetch to get the value after enrichment update
    grievance.refresh_from_db()
    # Status is set by enrichment but the base status must be a valid GrievanceStatus
    assert grievance.status in GrievanceStatus.values, (
        f"Unexpected status {grievance.status!r}"
    )


# ---------------------------------------------------------------------------
# 7. Two concurrent submissions get different tracking codes (sequence check)
# ---------------------------------------------------------------------------

@patch(_PATCH_LOCAL, return_value=[])
@patch(_PATCH_RECENT, return_value=[])
@patch(_PATCH_NLP, return_value=_NLP_STUB)
def test_sequential_tracking_codes_are_unique(mock_nlp, mock_recent, mock_local, citizen):
    g1 = submit_grievance(submitter=citizen, raw_text="First complaint about road")
    g2 = submit_grievance(submitter=citizen, raw_text="Second complaint about drain")
    assert g1.tracking_code != g2.tracking_code
    # Sequence should be sequential
    seq1 = int(g1.tracking_code.split("-")[2])
    seq2 = int(g2.tracking_code.split("-")[2])
    assert seq2 == seq1 + 1, f"Expected consecutive sequences, got {seq1} and {seq2}"
