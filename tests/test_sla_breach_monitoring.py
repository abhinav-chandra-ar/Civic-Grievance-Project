"""Tests for the check_sla_breaches management command — TASK 2.

Scope
-----
Verifies that the command correctly:

  * marks overdue (active, non-breached) SLAs as breached
  * fires the sla_breached signal (which wires to the breach alert email)
  * is idempotent — re-running does not double-process an already-breached SLA
  * leaves future SLAs untouched
  * does not write when --dry-run is passed
  * records an ESCALATION WorkflowEvent per grievance when --escalate is passed
  * skips escalation for grievances already in terminal state
"""
from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.utils import timezone

from apps.grievances.models import Grievance, GrievanceStatus
from apps.grievances.services import submit_grievance
from apps.slas.models import SLA, SLABreachType, SLAStatus
from apps.slas.services import create_sla_for_grievance
from apps.workflows.models import WorkflowEvent, WorkflowTransitionType

User = get_user_model()
pytestmark = pytest.mark.django_db

# ---------------------------------------------------------------------------
# NLP / integration stubs — bypass AI to create clean test fixtures
# ---------------------------------------------------------------------------

_PATCH_NLP    = "apps.integrations.services.classify_grievance_text"
_PATCH_RECENT = "apps.integrations.services.recent_grievance_summaries_for_duplicate_context"
_PATCH_LOCAL  = "apps.integrations.services.local_landmark_candidates_for_mention"
_PATCH_PHASE_E = "apps.integrations.routing.build_phase_e_routing"
_PHASE_E_STUB  = {"ward_instance": None, "department_instance": None, "routing_metadata": {}}

_NLP_STUB = {
    "normalized_summary": "Garbage not collected.",
    "category_code": "waste_management",
    "department_code": "sanitation",
    "priority": "medium",
    "confidence": 0.75,
    "language": "en",
    "provider": "transformer_v1",
    "metadata": {
        "text_length": 24,
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
            "automation_action": "auto_route",
            "routing_confidence": 0.75,
            "needs_review": False,
            "review_reasons": [],
            "duplicate_risk": {
                "risk_level": "low",
                "risk_score": 0.0,
                "is_confirmed": False,
            },
            "escalation": {"should_escalate": False, "escalation_reason": ""},
            "decision_metadata": {},
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_citizen(username: str):
    return User.objects.create_user(username=username, password="pass", role="citizen")


def _make_grievance(citizen):
    """Submit a grievance bypassing the AI pipeline."""
    with patch(_PATCH_NLP, return_value=_NLP_STUB), \
         patch(_PATCH_RECENT, return_value=[]), \
         patch(_PATCH_LOCAL, return_value=[]), \
         patch(_PATCH_PHASE_E, return_value=_PHASE_E_STUB):
        return submit_grievance(submitter=citizen, raw_text="Garbage not collected.")


def _make_overdue_sla(grievance, *, minutes_overdue: int = 10) -> SLA:
    """Create a second (extra) SLA row that is already overdue.

    NOTE: submit_grievance() already creates one SLA row.  Tests that need
    a separately controllable SLA (to avoid touching the auto-created one)
    create a fresh Grievance via submit_grievance and then rely on the SLA
    that was created as part of the submission flow.
    We instead use separate citizens per test to keep isolation clean.
    """
    now = timezone.now()
    return create_sla_for_grievance(
        grievance=grievance,
        response_due_at=now - timezone.timedelta(minutes=minutes_overdue),
        resolution_due_at=now - timezone.timedelta(minutes=minutes_overdue),
        policy_snapshot_metadata={"source": "test_overdue"},
    )


def _run_command(*args, **kwargs):
    """Call the management command, returning (stdout, stderr)."""
    out = StringIO()
    err = StringIO()
    call_command("check_sla_breaches", *args, stdout=out, stderr=err, **kwargs)
    return out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_command_marks_overdue_sla_as_breached():
    """An active SLA whose deadlines have passed is marked BREACHED."""
    citizen = _make_citizen("sla_breach_1")
    grievance = _make_grievance(citizen)

    # Get the auto-created SLA and push its deadlines into the past.
    sla = SLA.objects.get(grievance=grievance)
    past = timezone.now() - timezone.timedelta(hours=1)
    sla.response_due_at = past
    sla.resolution_due_at = past
    sla.save(update_fields=["response_due_at", "resolution_due_at"])

    assert not sla.is_breached

    _run_command()

    sla.refresh_from_db()
    assert sla.is_breached, "Command must mark overdue SLA as breached"
    assert sla.sla_status == SLAStatus.BREACHED
    assert sla.breach_type != SLABreachType.NONE
    assert sla.breached_at is not None


def test_command_is_idempotent():
    """Running the command twice does not double-process a breached SLA."""
    citizen = _make_citizen("sla_breach_idem")
    grievance = _make_grievance(citizen)

    sla = SLA.objects.get(grievance=grievance)
    past = timezone.now() - timezone.timedelta(hours=1)
    sla.response_due_at = past
    sla.resolution_due_at = past
    sla.save(update_fields=["response_due_at", "resolution_due_at"])

    _run_command()  # first run
    sla.refresh_from_db()
    first_breached_at = sla.breached_at

    _run_command()  # second run — must be a no-op
    sla.refresh_from_db()
    assert sla.breached_at == first_breached_at, (
        "Second run must not update breached_at (idempotent)"
    )
    # Confirm the breach count is still 1 (only one breach per SLA)
    assert SLA.objects.filter(grievance=grievance, is_breached=True).count() == 1


def test_command_dry_run_does_not_write():
    """--dry-run reports candidates without writing to the database."""
    citizen = _make_citizen("sla_dryrun")
    grievance = _make_grievance(citizen)

    sla = SLA.objects.get(grievance=grievance)
    past = timezone.now() - timezone.timedelta(hours=1)
    sla.response_due_at = past
    sla.resolution_due_at = past
    sla.save(update_fields=["response_due_at", "resolution_due_at"])

    stdout, _ = _run_command("--dry-run")

    sla.refresh_from_db()
    assert not sla.is_breached, "--dry-run must not write a breach to the database"
    assert "[DRY-RUN]" in stdout, "--dry-run output must include the DRY-RUN label"


def test_command_does_not_touch_future_slas():
    """SLAs whose deadlines are in the future are not breached."""
    citizen = _make_citizen("sla_future")
    grievance = _make_grievance(citizen)

    sla = SLA.objects.get(grievance=grievance)
    # Ensure both deadlines are well in the future.
    future = timezone.now() + timezone.timedelta(days=10)
    sla.response_due_at = future
    sla.resolution_due_at = future
    sla.save(update_fields=["response_due_at", "resolution_due_at"])

    _run_command()

    sla.refresh_from_db()
    assert not sla.is_breached, "Future SLAs must not be touched"
    assert sla.sla_status == SLAStatus.ACTIVE


def test_command_sends_breach_signal(monkeypatch):
    """sla_breached signal is fired once per newly breached SLA."""
    from apps.slas.signals import sla_breached

    fired_slas: list = []

    def _capture(sender, *, sla, **kwargs):
        fired_slas.append(sla.sla_code)

    sla_breached.connect(_capture)
    try:
        citizen = _make_citizen("sla_signal")
        grievance = _make_grievance(citizen)

        sla = SLA.objects.get(grievance=grievance)
        past = timezone.now() - timezone.timedelta(hours=1)
        sla.response_due_at = past
        sla.resolution_due_at = past
        sla.save(update_fields=["response_due_at", "resolution_due_at"])

        _run_command()

        assert sla.sla_code in fired_slas, "sla_breached signal must fire for the breached SLA"
    finally:
        sla_breached.disconnect(_capture)


def test_command_escalate_creates_workflow_event():
    """--escalate records an ESCALATION WorkflowEvent for each breached grievance."""
    citizen = _make_citizen("sla_esc_cmd")
    grievance = _make_grievance(citizen)

    sla = SLA.objects.get(grievance=grievance)
    past = timezone.now() - timezone.timedelta(hours=1)
    sla.response_due_at = past
    sla.resolution_due_at = past
    sla.save(update_fields=["response_due_at", "resolution_due_at"])

    _run_command("--escalate")

    escalation_events = WorkflowEvent.objects.filter(
        grievance=grievance,
        transition_type=WorkflowTransitionType.ESCALATION,
    )
    assert escalation_events.exists(), (
        "--escalate must create an ESCALATION WorkflowEvent for the breached grievance"
    )
    event = escalation_events.first()
    assert event.escalation_metadata.get("source") == "sla_breach_monitoring"
    assert event.escalation_metadata.get("sla_code") == sla.sla_code


def test_command_escalate_skips_terminal_grievances():
    """--escalate does not create a workflow event for resolved/closed grievances."""
    from apps.grievances.services import change_grievance_status

    citizen = _make_citizen("sla_term_skip")
    grievance = _make_grievance(citizen)

    # Move grievance to a terminal state.
    change_grievance_status(
        grievance=grievance,
        status=GrievanceStatus.RESOLVED,
        reason="Resolved before SLA breach.",
    )

    sla = SLA.objects.get(grievance=grievance)
    past = timezone.now() - timezone.timedelta(hours=1)
    sla.response_due_at = past
    sla.resolution_due_at = past
    sla.save(update_fields=["response_due_at", "resolution_due_at"])

    _run_command("--escalate")

    # SLA should be breached...
    sla.refresh_from_db()
    assert sla.is_breached, "SLA must still be marked breached even for terminal grievances"

    # ...but no escalation event for a resolved grievance
    escalation_events = WorkflowEvent.objects.filter(
        grievance=grievance,
        transition_type=WorkflowTransitionType.ESCALATION,
    )
    assert not escalation_events.exists(), (
        "--escalate must skip RESOLVED grievances"
    )


def test_command_no_escalate_without_flag():
    """Without --escalate, no ESCALATION events are created even for breached SLAs."""
    citizen = _make_citizen("sla_no_esc_flag")
    grievance = _make_grievance(citizen)

    sla = SLA.objects.get(grievance=grievance)
    past = timezone.now() - timezone.timedelta(hours=1)
    sla.response_due_at = past
    sla.resolution_due_at = past
    sla.save(update_fields=["response_due_at", "resolution_due_at"])

    _run_command()  # no --escalate flag

    escalation_events = WorkflowEvent.objects.filter(
        grievance=grievance,
        transition_type=WorkflowTransitionType.ESCALATION,
    )
    assert not escalation_events.exists(), (
        "Without --escalate, no ESCALATION events must be created"
    )


def test_command_outputs_success_message():
    """The command prints a summary line on successful completion."""
    citizen = _make_citizen("sla_output")
    grievance = _make_grievance(citizen)

    sla = SLA.objects.get(grievance=grievance)
    past = timezone.now() - timezone.timedelta(hours=1)
    sla.response_due_at = past
    sla.resolution_due_at = past
    sla.save(update_fields=["response_due_at", "resolution_due_at"])

    stdout, _ = _run_command()

    assert "breached" in stdout.lower(), "Command output must mention breach count"
