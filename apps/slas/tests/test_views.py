"""View tests for SLA visibility."""
from __future__ import annotations

import pytest
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from apps.grievances.services import submit_grievance
from apps.slas.services import create_sla_for_grievance
from apps.slas.views import SLAViewSet

pytestmark = pytest.mark.django_db


def test_citizen_list_returns_own_grievance_sla(django_user_model) -> None:
    citizen = django_user_model.objects.create_user(username="citizen", password="password")
    other = django_user_model.objects.create_user(username="other", password="password")
    own_grievance = submit_grievance(submitter=citizen, raw_text="Own issue.")
    other_grievance = submit_grievance(submitter=other, raw_text="Other issue.")
    now = timezone.now()
    create_sla_for_grievance(
        grievance=own_grievance,
        response_due_at=now + timezone.timedelta(hours=1),
        resolution_due_at=now + timezone.timedelta(days=1),
    )
    create_sla_for_grievance(
        grievance=other_grievance,
        response_due_at=now + timezone.timedelta(hours=1),
        resolution_due_at=now + timezone.timedelta(days=1),
    )
    request = APIRequestFactory().get("/slas/")
    force_authenticate(request, user=citizen)

    response = SLAViewSet.as_view({"get": "list"})(request)

    assert response.status_code == status.HTTP_200_OK
    assert len(response.data["results"]) == 1
