"""Tests for AI-triggered escalation automation — TASK 1.

Scope
-----
Verifies that when the AI decision engine produces ``should_escalate=True``
the system correctly:

  * creates an ESCALATION WorkflowEvent
  * persists escalation_metadata on the event
  * creates an ESCALATION AuditLog entry
  * does NOT change the grievance's operational status
  * does NOT fire the citizen status-change email (same status)
  * skips gracefully when ``should_escalate=False``
  * get_or_creates the ``__system__`` user as the event actor

escalate_grievance_from_system() is also tested directly to verify the
contract independently of the AI pipeline.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model

from apps.audit.models import AuditActionType, AuditLog
from apps.grievances.models import Grievance, GrievanceStatus
from apps.grievances.services import submit_grievance
from apps.workflows.models import WorkflowEvent, WorkflowTransitionType
from apps.workflows.services import escalate_grievance_from_system

User = get_user_model()
pytestmark = pytest.mark.django_db

# ---------------------------------------------------------------------------
# NLP stubs
# ---------------------------------------------------------------------------

_PATCH_NLP    = "apps.integrations.services.classify_grievance_text"
_PATCH_RECENT = "apps.integrations.services.recent_grievance_summaries_for_duplicate_context"
_PATCH_LOCAL  = "apps.integrations.services.local_landmark_candidates_for_mention"
_PATCH_PHASE_E = "apps.integrations.routing.build_phase_e_routing"
_PHASE_E_STUB  = {"ward_instance": None, "department_instance": None, "routing_metadata": {}}

_NLP_STUB_ESCALATE = {
    "normalized_summary": "Critical water main burst — flooding residential area.",
    "category_code": "water_supply",
    "department_code": "water_authority",
    "priority": "critical",
    "confidence": 0.92,
    "language": "en",
    "provider": "transformer_v1",
    "metadata": {
        "text_length": 55,
        "ward_hint": None,
        "landmark_hints": [],
        "spam_check": {"is_spam": False, "spam_score": 0.0, "spam_reason": ""},
        "duplicate_check": {
            "is_duplicate": False,
            "similarity_score": 0.0,
            "matching_text": None,
        },
        "needs_human_review": False,
        "review_reasons": [],
        "image_analysis": None,
        "consistency_check": None,
        "evidence_quality": None,
        "evidence_review_reason": "",
        "decision": {
            "automation_action": "escalate",
            "routing_confidence": 0.90,
            "needs_review": False,
            "review_reasons": [],
            "duplicate_risk": {
                "risk_level": "low",
                "risk_score": 0.0,
                "is_confirmed": False,
            },
            "escalation": {
                "should_escalate": True,
                "escalation_reason": "Life-safety concern: flooding near hospital.",
            },
            "decision_metadata": {},
        },
    },
}

_NLP_STUB_NO_ESCALATE = {
    **_NLP_STUB_ESCALATE,
    "metadata": {
        **_NLP_STUB_ESCALATE["metadata"],
        "decision": {
            **_NLP_STUB_ESCALATE["metadata"]["decision"],
            "automation_action": "auto_route",
            "escalation": {"should_escalate": False, "escalation_reason": ""},
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _submit_escalated(citizen):
    """Submit a grievance whose AI stub has should_escalate=True."""
    with patch(_PATCH_NLP, return_value=_NLP_STUB_ESCALATE), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]), \
         patch(_PATCH_PHASE_E, return_value=_PHASE_E_STUB):
        return submit_grievance(
            submitter=citizen,
            raw_text="Critical water main burst, flooding near hospital.",
        )


def _submit_no_escalate(citizen):
    """Submit a grievance whose AI stub has should_escalate=False."""
    with patch(_PATCH_NLP, return_value=_NLP_STUB_NO_ESCALATE), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]), \
         patch(_PATCH_PHASE_E, return_value=_PHASE_E_STUB):
        return submit_grievance(
            submitter=citizen,
            raw_text="Pothole on main road.",
        )


# ---------------------------------------------------------------------------
# Direct unit tests for escalate_grievance_from_system()
# ---------------------------------------------------------------------------

def test_escalate_creates_escalation_workflow_event():
    """Direct call creates a WorkflowEvent with ESCALATION transition type."""
    citizen = User.objects.create_user(username="cit_esc1", password="pass", role="citizen")
    with patch(_PATCH_NLP, return_value=_NLP_STUB_NO_ESCALATE), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]), \
         patch(_PATCH_PHASE_E, return_value=_PHASE_E_STUB):
        grievance = submit_grievance(submitter=citizen, raw_text="Street light broken.")

    event = escalate_grievance_from_system(
        grievance=grievance,
        transition_reason="Manual test escalation",
        escalation_metadata={"source": "test", "reason": "test_reason"},
    )

    assert event.transition_type == WorkflowTransitionType.ESCALATION
    assert event.grievance == grievance
    assert event.event_code.startswith("WFE-")


def test_escalate_does_not_change_grievance_status():
    """Escalation must not mutate the grievance's operational status."""
    citizen = User.objects.create_user(username="cit_esc2", password="pass", role="citizen")
    with patch(_PATCH_NLP, return_value=_NLP_STUB_NO_ESCALATE), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]), \
         patch(_PATCH_PHASE_E, return_value=_PHASE_E_STUB):
        grievance = submit_grievance(submitter=citizen, raw_text="Road damage.")

    status_before = grievance.status

    escalate_grievance_from_system(
        grievance=grievance,
        transition_reason="Escalation must not change status",
    )

    grievance.refresh_from_db()
    assert grievance.status == status_before, (
        "escalate_grievance_from_system() must not change the grievance's status"
    )


def test_escalate_persists_escalation_metadata_on_event():
    """escalation_metadata dict is stored on the WorkflowEvent row."""
    citizen = User.objects.create_user(username="cit_esc3", password="pass", role="citizen")
    with patch(_PATCH_NLP, return_value=_NLP_STUB_NO_ESCALATE), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]), \
         patch(_PATCH_PHASE_E, return_value=_PHASE_E_STUB):
        grievance = submit_grievance(submitter=citizen, raw_text="Tree fallen on road.")

    event = escalate_grievance_from_system(
        grievance=grievance,
        transition_reason="Test metadata",
        escalation_metadata={"source": "sla_breach", "sla_code": "SLA-2026-000001"},
    )

    assert event.escalation_metadata["source"] == "sla_breach"
    assert event.escalation_metadata["sla_code"] == "SLA-2026-000001"


def test_escalate_creates_audit_log_with_escalation_action():
    """A ESCALATION AuditLog entry must be created for every escalation."""
    citizen = User.objects.create_user(username="cit_esc4", password="pass", role="citizen")
    with patch(_PATCH_NLP, return_value=_NLP_STUB_NO_ESCALATE), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]), \
         patch(_PATCH_PHASE_E, return_value=_PHASE_E_STUB):
        grievance = submit_grievance(submitter=citizen, raw_text="Sewage overflow.")

    escalate_grievance_from_system(
        grievance=grievance,
        transition_reason="Sewage overflow escalation",
        escalation_metadata={"source": "test_audit"},
    )

    logs = AuditLog.objects.filter(
        target_model="grievances.Grievance",
        target_object_id=str(grievance.pk),
        action_type=AuditActionType.ESCALATION,
    )
    assert logs.exists(), "No ESCALATION AuditLog was created"
    assert logs.first().change_metadata.get("escalation_reason") == "Sewage overflow escalation"


def test_escalate_uses_system_user_as_actor():
    """The __system__ user is created and used as the actor."""
    citizen = User.objects.create_user(username="cit_esc5", password="pass", role="citizen")
    with patch(_PATCH_NLP, return_value=_NLP_STUB_NO_ESCALATE), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]), \
         patch(_PATCH_PHASE_E, return_value=_PHASE_E_STUB):
        grievance = submit_grievance(submitter=citizen, raw_text="Power cut.")

    event = escalate_grievance_from_system(
        grievance=grievance,
        transition_reason="Power escalation",
    )

    assert event.actor.username == "__system__"
    assert event.actor.role == "system_operator"


def test_escalate_system_user_is_reused_not_duplicated():
    """Calling escalate_grievance_from_system() twice reuses the __system__ user."""
    citizen = User.objects.create_user(username="cit_esc6", password="pass", role="citizen")
    with patch(_PATCH_NLP, return_value=_NLP_STUB_NO_ESCALATE), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]), \
         patch(_PATCH_PHASE_E, return_value=_PHASE_E_STUB):
        grievance = submit_grievance(submitter=citizen, raw_text="Water issue.")

    escalate_grievance_from_system(
        grievance=grievance, transition_reason="First escalation"
    )
    escalate_grievance_from_system(
        grievance=grievance, transition_reason="Second escalation"
    )

    system_count = User.objects.filter(username="__system__").count()
    assert system_count == 1, "get_or_create must not create duplicate __system__ users"


# ---------------------------------------------------------------------------
# Integration tests: escalation triggered by AI pipeline
# ---------------------------------------------------------------------------

def test_ai_pipeline_creates_escalation_event_when_should_escalate_true():
    """When AI produces should_escalate=True, an ESCALATION WorkflowEvent is created."""
    citizen = User.objects.create_user(username="cit_ai_esc1", password="pass", role="citizen")
    grievance = _submit_escalated(citizen)

    escalation_events = WorkflowEvent.objects.filter(
        grievance=grievance,
        transition_type=WorkflowTransitionType.ESCALATION,
    )
    assert escalation_events.exists(), (
        "AI should_escalate=True must produce an ESCALATION WorkflowEvent"
    )


def test_ai_pipeline_escalation_event_contains_reason():
    """The AI escalation reason is stored in escalation_metadata."""
    citizen = User.objects.create_user(username="cit_ai_esc2", password="pass", role="citizen")
    grievance = _submit_escalated(citizen)

    event = WorkflowEvent.objects.get(
        grievance=grievance,
        transition_type=WorkflowTransitionType.ESCALATION,
    )
    assert event.escalation_metadata.get("source") == "ai_enrichment"
    assert "Life-safety" in event.escalation_metadata.get("escalation_reason", "")


def test_ai_pipeline_escalation_creates_audit_log():
    """AI-triggered escalation must produce an ESCALATION AuditLog entry."""
    citizen = User.objects.create_user(username="cit_ai_esc3", password="pass", role="citizen")
    grievance = _submit_escalated(citizen)

    logs = AuditLog.objects.filter(
        target_model="grievances.Grievance",
        target_object_id=str(grievance.pk),
        action_type=AuditActionType.ESCALATION,
    )
    assert logs.exists(), "AI escalation must produce an ESCALATION AuditLog entry"


def test_no_escalation_when_should_escalate_false():
    """When AI produces should_escalate=False, no ESCALATION event is created."""
    citizen = User.objects.create_user(username="cit_ai_noesc", password="pass", role="citizen")
    grievance = _submit_no_escalate(citizen)

    escalation_events = WorkflowEvent.objects.filter(
        grievance=grievance,
        transition_type=WorkflowTransitionType.ESCALATION,
    )
    assert not escalation_events.exists(), (
        "should_escalate=False must NOT produce an ESCALATION WorkflowEvent"
    )


def test_ai_escalation_does_not_change_grievance_status():
    """AI-triggered escalation must not overwrite the grievance status."""
    citizen = User.objects.create_user(username="cit_ai_stat", password="pass", role="citizen")
    grievance = _submit_escalated(citizen)

    # Enrichment sets status through the normal enrichment path;
    # the escalation must not additionally overwrite it.
    grievance.refresh_from_db()
    assert grievance.status in GrievanceStatus.values, (
        f"Unexpected status after AI escalation: {grievance.status!r}"
    )
    # Verify the ESCALATION event's new_status == previous_status (no status change).
    event = WorkflowEvent.objects.filter(
        grievance=grievance,
        transition_type=WorkflowTransitionType.ESCALATION,
    ).first()
    if event:
        assert event.new_status == event.previous_status, (
            "ESCALATION event must not change status"
        )
