"""View tests for workflow history visibility."""
from __future__ import annotations

import pytest
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from apps.grievances.services import submit_grievance
from apps.workflows.services import transition_grievance
from apps.workflows.views import WorkflowEventViewSet

pytestmark = pytest.mark.django_db


def test_citizen_list_returns_own_workflow_events(django_user_model) -> None:
    citizen = django_user_model.objects.create_user(username="citizen", password="password")
    other = django_user_model.objects.create_user(username="other", password="password")
    own = submit_grievance(submitter=citizen, raw_text="Own grievance.")
    other_grievance = submit_grievance(submitter=other, raw_text="Other grievance.")
    transition_grievance(
        grievance=own,
        actor=citizen,
        new_status="triaged",
        transition_type="status_change",
    )
    transition_grievance(
        grievance=other_grievance,
        actor=other,
        new_status="triaged",
        transition_type="status_change",
    )
    request = APIRequestFactory().get("/workflows/")
    force_authenticate(request, user=citizen)

    response = WorkflowEventViewSet.as_view({"get": "list"})(request)

    assert response.status_code == status.HTTP_200_OK
    assert len(response.data["results"]) == 1
