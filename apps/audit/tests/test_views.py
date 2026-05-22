"""View tests for read-only audit API."""
from __future__ import annotations

import pytest
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from apps.audit.services import record_audit_event
from apps.audit.views import AuditLogViewSet

pytestmark = pytest.mark.django_db


def test_audit_viewset_lists_logs_for_reader_role(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="operator",
        password="password",
        role="system_operator",
    )
    record_audit_event(
        actor=actor,
        target_model="grievances.Grievance",
        target_object_id="1",
        action_type="update",
    )
    request = APIRequestFactory().get("/audit/")
    force_authenticate(request, user=actor)

    response = AuditLogViewSet.as_view({"get": "list"})(request)

    assert response.status_code == status.HTTP_200_OK
    assert len(response.data["results"]) == 1


def test_audit_viewset_has_no_create_action() -> None:
    assert "post" not in AuditLogViewSet().get_extra_actions()
