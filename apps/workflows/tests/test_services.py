"""Service tests for workflow history creation."""
from __future__ import annotations

import pytest

from apps.grievances.services import submit_grievance
from apps.workflows.models import WorkflowTransitionType
from apps.workflows.services import record_workflow_comment, transition_grievance

pytestmark = pytest.mark.django_db


def test_transition_updates_grievance_and_records_event(django_user_model) -> None:
    actor = django_user_model.objects.create_user(username="actor", password="password")
    grievance = submit_grievance(submitter=actor, raw_text="Drain blocked.")

    event = transition_grievance(
        grievance=grievance,
        actor=actor,
        new_status="triaged",
        transition_type=WorkflowTransitionType.STATUS_CHANGE,
        transition_reason="Reviewed by officer.",
    )

    grievance.refresh_from_db()
    assert grievance.status == "triaged"
    assert event.previous_status == "submitted"
    assert event.new_status == "triaged"
    assert event.event_code.startswith("WFE-")


def test_assignment_preserves_assignee_context(django_user_model) -> None:
    actor = django_user_model.objects.create_user(username="actor", password="password")
    assignee = django_user_model.objects.create_user(username="assignee", password="password")
    grievance = submit_grievance(submitter=actor, raw_text="Road damaged.")

    event = transition_grievance(
        grievance=grievance,
        actor=actor,
        assignee=assignee,
        new_status="assigned",
        transition_type=WorkflowTransitionType.ASSIGNMENT,
        assignment_metadata={"source": "manual"},
    )

    assert event.assignee == assignee
    assert event.assignment_metadata["assignee_user_id"] == assignee.pk
    assert event.assignment_metadata["source"] == "manual"


def test_comment_event_keeps_grievance_status(django_user_model) -> None:
    actor = django_user_model.objects.create_user(username="actor", password="password")
    grievance = submit_grievance(submitter=actor, raw_text="Streetlight out.")

    event = record_workflow_comment(grievance=grievance, actor=actor, remarks="Awaiting photo.")

    grievance.refresh_from_db()
    assert grievance.status == "submitted"
    assert event.previous_status == event.new_status == "submitted"
    assert event.transition_type == WorkflowTransitionType.COMMENT


# ---------------------------------------------------------------------------
# RETURN transition — Task 1
# ---------------------------------------------------------------------------

def test_return_clears_department_and_sets_triaged(django_user_model) -> None:
    from apps.departments.models import Department
    from apps.workflows.services import return_grievance_to_intake
    from django.contrib.gis.geos import Point

    actor = django_user_model.objects.create_user(
        username="dept_off", password="password", role="department_officer"
    )
    dept = Department.objects.create(
        code="roads_and_drainage", name="Roads and Drainage"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Broken road.")

    # Simulate the grievance having been assigned to a department.
    grievance.department = dept
    grievance.status = "assigned"
    grievance.save(update_fields=["department", "status"])

    event = return_grievance_to_intake(
        grievance=grievance,
        actor=actor,
        transition_reason="Wrong department — should go to Water Authority.",
    )

    grievance.refresh_from_db()
    assert grievance.status == "triaged", "RETURN must move status to TRIAGED"
    assert grievance.department is None, "RETURN must clear the department FK"
    assert event.transition_type == WorkflowTransitionType.RETURN
    assert event.previous_status == "assigned"
    assert event.new_status == "triaged"
    assert "Wrong department" in event.transition_reason


def test_return_records_reason_in_status_metadata(django_user_model) -> None:
    from apps.workflows.services import return_grievance_to_intake

    actor = django_user_model.objects.create_user(username="officer2", password="password")
    grievance = submit_grievance(submitter=actor, raw_text="Water supply issue.")
    grievance.status = "in_progress"
    grievance.save(update_fields=["status"])

    return_grievance_to_intake(
        grievance=grievance,
        actor=actor,
        transition_reason="Misclassified — belongs to Water Authority.",
        remarks="Field inspection confirmed wrong routing.",
    )

    grievance.refresh_from_db()
    assert grievance.status_metadata.get("return_reason") == "Misclassified — belongs to Water Authority."


def test_return_is_not_rejection_complaint_stays_open(django_user_model) -> None:
    from apps.workflows.services import return_grievance_to_intake

    actor = django_user_model.objects.create_user(username="officer3", password="password")
    grievance = submit_grievance(submitter=actor, raw_text="Park light out.")

    return_grievance_to_intake(
        grievance=grievance,
        actor=actor,
        transition_reason="Wrong department.",
    )

    grievance.refresh_from_db()
    # TRIAGED means open and awaiting routing — not closed/rejected.
    assert grievance.status not in ("rejected", "closed")
