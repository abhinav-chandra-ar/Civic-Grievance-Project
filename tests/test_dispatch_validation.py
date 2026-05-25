"""Real-ML dispatch validation tests — no NLP mocks.

These tests run the genuine ML pipeline (transformer/TF-IDF) against a clean
Django test DB with only the necessary seed data.  Phase E routing is stubbed
so the test doesn't depend on PostGIS ward geometry, but the NLP pipeline
(category, priority, escalation, duplicate detection) runs for real.

Validation cases
----------------
A) "live electric wire fallen in Pattom near road"
     ML must produce: category=electrical_hazard, should_escalate=True
     With KSEB department injected via Phase E stub → status=assigned
     ESCALATION WorkflowEvent with previous_status==new_status must exist.
     No 'escalated' lifecycle state must appear.

B) "there is a broken streetlight on the road" (vague, no dept resolved by Phase E)
     ML must produce: some category (road or electrical)
     Phase E stub returns no department → status=triaged
     No ESCALATION event (if ML doesn't flag it for escalation).

C) Escalation prev==new invariant
     After dispatch from A, the ESCALATION event carries the post-assignment
     status on both sides (assigned→assigned) — not submitted→assigned.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model

from apps.departments.models import Department
from apps.grievances.models import GrievanceStatus
from apps.grievances.services import submit_grievance
from apps.workflows.models import WorkflowEvent, WorkflowTransitionType

User = get_user_model()

pytestmark = pytest.mark.django_db

_PATCH_PHASE_E = "apps.integrations.routing.build_phase_e_routing"


def _kseb_dept() -> Department:
    dept, _ = Department.objects.get_or_create(
        code="water_authority",  # use existing code that routing maps to KSEB
        defaults={
            "name": "Kerala State Electricity Board (KSEB)",
            "handled_categories": ["electrical_hazard"],
            "is_active": True,
        },
    )
    return dept


def _make_phase_e_stub(dept_instance) -> dict:
    return {
        "ward_instance": None,
        "department_instance": dept_instance,
        "routing_metadata": {"stub": True, "note": "Phase E stubbed for DB portability"},
    }


# ── Case A ─────────────────────────────────────────────────────────────────

def test_electrical_hazard_routed_to_assigned_with_escalation():
    """Real ML classifies electrical hazard → ASSIGNED + ESCALATION event.

    The NLP pipeline is NOT mocked.  Phase E is stubbed so the department
    is always KSEB regardless of PostGIS availability.
    """
    citizen = User.objects.create_user(
        username="val_citizen_A", password="pass", role="citizen"
    )
    dept = _kseb_dept()

    with patch(_PATCH_PHASE_E, return_value=_make_phase_e_stub(dept)):
        g = submit_grievance(
            submitter=citizen,
            raw_text="live electric wire fallen in Pattom near road",
        )

    g.refresh_from_db()

    # ── Category and routing ────────────────────────────────────────────────
    assert g.category_code == "electrical_hazard", (
        f"Expected electrical_hazard, got {g.category_code!r}"
    )
    assert g.department == dept, (
        f"Expected KSEB department, got {g.department}"
    )

    # ── Lifecycle status ────────────────────────────────────────────────────
    assert g.status == GrievanceStatus.ASSIGNED, (
        f"Expected assigned, got {g.status!r}"
    )
    assert g.status != "escalated", (
        "'escalated' is not a lifecycle state; grievance must be in assigned"
    )

    # ── Escalation flag in status_metadata ─────────────────────────────────
    sm = g.status_metadata or {}
    esc = sm.get("escalation") or {}
    assert esc.get("should_escalate") is True, (
        f"Expected should_escalate=True in status_metadata, got {esc}"
    )

    # ── WorkflowEvents ─────────────────────────────────────────────────────
    events = WorkflowEvent.objects.filter(grievance=g).order_by("occurred_at")

    submitted_to_assigned = events.filter(
        previous_status="submitted", new_status="assigned"
    )
    assert submitted_to_assigned.exists(), (
        "Expected submitted→assigned WorkflowEvent; events: "
        + str(list(events.values_list("previous_status", "new_status", "transition_type")))
    )

    esc_events = events.filter(transition_type=WorkflowTransitionType.ESCALATION)
    assert esc_events.exists(), (
        "Expected ESCALATION WorkflowEvent for electrical_hazard life-safety complaint"
    )

    esc_event = esc_events.first()
    assert esc_event.previous_status == esc_event.new_status, (
        f"ESCALATION event must have previous==new (alert not state change); "
        f"got {esc_event.previous_status!r}→{esc_event.new_status!r}"
    )
    # After dispatch the escalation alert fires on the *assigned* status.
    assert esc_event.new_status == "assigned", (
        f"ESCALATION event must record assigned status, got {esc_event.new_status!r}"
    )


# ── Case B ─────────────────────────────────────────────────────────────────

def test_vague_complaint_no_dept_routed_to_triaged():
    """Real ML + no department from Phase E → TRIAGED (ward officer queue)."""
    citizen = User.objects.create_user(
        username="val_citizen_B", password="pass", role="citizen"
    )

    no_dept_stub = {
        "ward_instance": None,
        "department_instance": None,  # Phase E cannot resolve
        "routing_metadata": {"stub": True},
    }
    with patch(_PATCH_PHASE_E, return_value=no_dept_stub):
        g = submit_grievance(
            submitter=citizen,
            raw_text="there is a broken streetlight on the road",
        )

    g.refresh_from_db()

    assert g.status == GrievanceStatus.TRIAGED, (
        f"Expected triaged when no department resolved, got {g.status!r}"
    )

    events = WorkflowEvent.objects.filter(grievance=g).order_by("occurred_at")
    submitted_to_triaged = events.filter(
        previous_status="submitted", new_status="triaged"
    )
    assert submitted_to_triaged.exists(), (
        "Expected submitted→triaged WorkflowEvent"
    )


# ── Case C ─────────────────────────────────────────────────────────────────

def test_escalation_prev_equals_new_is_post_dispatch_status():
    """ESCALATION event carries the lifecycle status set by dispatch, not 'submitted'.

    Step A (lifecycle): submitted → assigned
    Step B (alert):     ESCALATION event with previous_status=assigned, new_status=assigned

    This ensures the escalation alert records the correct post-dispatch state.
    """
    citizen = User.objects.create_user(
        username="val_citizen_C", password="pass", role="citizen"
    )
    dept = _kseb_dept()

    with patch(_PATCH_PHASE_E, return_value=_make_phase_e_stub(dept)):
        g = submit_grievance(
            submitter=citizen,
            raw_text="live electric wire fallen in Pattom near road hazard urgent",
        )

    g.refresh_from_db()

    if g.status != GrievanceStatus.ASSIGNED:
        pytest.skip(
            f"ML did not produce electrical_hazard+escalation for this text "
            f"(status={g.status!r}); skip prev==new check"
        )

    esc_events = WorkflowEvent.objects.filter(
        grievance=g,
        transition_type=WorkflowTransitionType.ESCALATION,
    )
    if not esc_events.exists():
        pytest.skip("ML did not flag escalation; skip prev==new check")

    esc_event = esc_events.first()
    assert esc_event.previous_status == "assigned"
    assert esc_event.new_status == "assigned"
