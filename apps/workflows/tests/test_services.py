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
