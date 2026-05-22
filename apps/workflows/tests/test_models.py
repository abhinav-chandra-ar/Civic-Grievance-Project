"""Model tests for workflow event history."""
from __future__ import annotations

import pytest
from django.core.exceptions import ValidationError
from django.utils import timezone

from apps.grievances.services import submit_grievance
from apps.workflows.models import WorkflowEvent, WorkflowTransitionType

pytestmark = pytest.mark.django_db


def test_workflow_transition_types_are_closed() -> None:
    assert set(WorkflowTransitionType.values) == {
        "status_change",
        "assignment",
        "reassignment",
        "escalation",
        "resolution",
        "rejection",
        "closure",
        "comment",
    }


def test_workflow_metadata_hooks_must_be_mappings(django_user_model) -> None:
    actor = django_user_model.objects.create_user(username="actor", password="password")
    grievance = submit_grievance(submitter=actor, raw_text="Water leak.")
    event = WorkflowEvent(
        event_code="WFE-2026-000001",
        grievance=grievance,
        actor=actor,
        transition_type=WorkflowTransitionType.STATUS_CHANGE,
        previous_status="submitted",
        new_status="triaged",
        assignment_metadata=["assignment"],
        occurred_at=timezone.now(),
    )

    with pytest.raises(ValidationError) as exc_info:
        event.full_clean()

    assert "assignment_metadata" in exc_info.value.message_dict
