"""View tests for grievance visibility."""
from __future__ import annotations

import pytest
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from apps.grievances.services import submit_grievance
from apps.grievances.views import GrievanceViewSet

pytestmark = pytest.mark.django_db


def test_citizen_list_returns_only_own_grievances(django_user_model) -> None:
    citizen = django_user_model.objects.create_user(username="citizen", password="password")
    other = django_user_model.objects.create_user(username="other", password="password")
    submit_grievance(submitter=citizen, raw_text="Own grievance.")
    submit_grievance(submitter=other, raw_text="Other grievance.")
    request = APIRequestFactory().get("/grievances/")
    force_authenticate(request, user=citizen)

    response = GrievanceViewSet.as_view({"get": "list"})(request)

    assert response.status_code == status.HTTP_200_OK
    assert len(response.data["results"]) == 1
    assert response.data["results"][0]["raw_text"] == "Own grievance."
