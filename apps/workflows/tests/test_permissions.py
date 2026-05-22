"""Permission tests for workflow transitions."""
from __future__ import annotations

from dataclasses import dataclass

from rest_framework.test import APIRequestFactory

from apps.workflows.permissions import IsWorkflowOperatorRole


@dataclass
class UserStub:
    role: str
    is_authenticated: bool = True


def test_workflow_operator_permission_accepts_ward_officer() -> None:
    request = APIRequestFactory().post("/workflows/")
    request.user = UserStub(role="ward_officer")

    assert IsWorkflowOperatorRole().has_permission(request, view=None)


def test_workflow_operator_permission_rejects_citizen() -> None:
    request = APIRequestFactory().post("/workflows/")
    request.user = UserStub(role="citizen")

    assert not IsWorkflowOperatorRole().has_permission(request, view=None)
