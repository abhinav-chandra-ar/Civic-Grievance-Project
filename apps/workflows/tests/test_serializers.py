"""Serializer tests for workflow transitions."""
from __future__ import annotations

import pytest

from apps.grievances.services import submit_grievance
from apps.workflows.serializers import WorkflowTransitionSerializer

pytestmark = pytest.mark.django_db


def test_transition_serializer_records_event(django_user_model) -> None:
    actor = django_user_model.objects.create_user(username="actor", password="password")
    grievance = submit_grievance(submitter=actor, raw_text="Waste overflow.")
    serializer = WorkflowTransitionSerializer(
        data={
            "grievance": grievance.pk,
            "transition_type": "status_change",
            "new_status": "triaged",
            "remarks": "Checked.",
        }
    )

    assert serializer.is_valid(), serializer.errors
    event = serializer.save(actor=actor)
    assert event.grievance == grievance
    assert event.new_status == "triaged"


def test_transition_serializer_rejects_comment_transition(django_user_model) -> None:
    actor = django_user_model.objects.create_user(username="actor", password="password")
    grievance = submit_grievance(submitter=actor, raw_text="Comment target.")
    serializer = WorkflowTransitionSerializer(
        data={
            "grievance": grievance.pk,
            "transition_type": "comment",
            "new_status": "submitted",
        }
    )

    assert not serializer.is_valid()
    assert "transition_type" in serializer.errors
