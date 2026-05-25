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


# ---------------------------------------------------------------------------
# RETURN transition serializer tests — Task 1
# ---------------------------------------------------------------------------

def test_return_transition_accepted_for_dept_officer(django_user_model) -> None:
    from dataclasses import dataclass
    from rest_framework.test import APIRequestFactory

    actor = django_user_model.objects.create_user(
        username="dept_off2", password="password", role="department_officer"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Wrong routing test.")

    request = APIRequestFactory().post("/workflows/")
    request.user = actor

    serializer = WorkflowTransitionSerializer(
        data={
            "grievance": grievance.pk,
            "transition_type": "return",
            "new_status": "triaged",
            "transition_reason": "Complaint belongs to a different department.",
        },
        context={"request": request},
    )

    assert serializer.is_valid(), serializer.errors
    event = serializer.save(actor=actor)
    assert event.transition_type == "return"
    assert event.new_status == "triaged"


def test_return_transition_rejected_for_ward_officer(django_user_model) -> None:
    from rest_framework.test import APIRequestFactory

    actor = django_user_model.objects.create_user(
        username="ward_off2", password="password", role="ward_officer"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Ward officer RETURN attempt.")

    request = APIRequestFactory().post("/workflows/")
    request.user = actor

    serializer = WorkflowTransitionSerializer(
        data={
            "grievance": grievance.pk,
            "transition_type": "return",
            "new_status": "triaged",
            "transition_reason": "Wrong routing.",
        },
        context={"request": request},
    )

    assert not serializer.is_valid()
    # Ward officers cannot RETURN — they should use a workflow comment instead.
    assert "transition_type" in serializer.errors or "non_field_errors" in serializer.errors


def test_return_transition_requires_reason(django_user_model) -> None:
    from rest_framework.test import APIRequestFactory

    actor = django_user_model.objects.create_user(
        username="dept_off3", password="password", role="department_officer"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Reason missing test.")

    request = APIRequestFactory().post("/workflows/")
    request.user = actor

    serializer = WorkflowTransitionSerializer(
        data={
            "grievance": grievance.pk,
            "transition_type": "return",
            "new_status": "triaged",
            "transition_reason": "   ",  # blank reason
        },
        context={"request": request},
    )

    assert not serializer.is_valid()
    assert "transition_reason" in serializer.errors or "non_field_errors" in serializer.errors


def test_return_transition_enforces_triaged_status(django_user_model) -> None:
    from rest_framework.test import APIRequestFactory

    actor = django_user_model.objects.create_user(
        username="dept_off4", password="password", role="department_officer"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Wrong status test.")

    request = APIRequestFactory().post("/workflows/")
    request.user = actor

    serializer = WorkflowTransitionSerializer(
        data={
            "grievance": grievance.pk,
            "transition_type": "return",
            "new_status": "resolved",  # wrong — must be triaged
            "transition_reason": "Wrong department.",
        },
        context={"request": request},
    )

    assert not serializer.is_valid()
    assert "new_status" in serializer.errors or "non_field_errors" in serializer.errors
