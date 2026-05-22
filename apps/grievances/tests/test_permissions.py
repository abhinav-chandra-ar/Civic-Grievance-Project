"""Permission tests for grievance access roles."""
from __future__ import annotations

from dataclasses import dataclass

from rest_framework.test import APIRequestFactory

from apps.grievances.permissions import IsGrievanceOperatorRole


@dataclass
class UserStub:
    role: str
    is_authenticated: bool = True


def test_grievance_operator_permission_accepts_department_officer() -> None:
    request = APIRequestFactory().patch("/grievances/1/")
    request.user = UserStub(role="department_officer")

    assert IsGrievanceOperatorRole().has_permission(request, view=None)


def test_grievance_operator_permission_rejects_citizen() -> None:
    request = APIRequestFactory().patch("/grievances/1/")
    request.user = UserStub(role="citizen")

    assert not IsGrievanceOperatorRole().has_permission(request, view=None)
