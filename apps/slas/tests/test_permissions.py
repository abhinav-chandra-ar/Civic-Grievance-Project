"""Permission tests for SLA operations."""
from __future__ import annotations

from dataclasses import dataclass

from rest_framework.test import APIRequestFactory

from apps.slas.permissions import IsSLAOperatorRole


@dataclass
class UserStub:
    role: str
    is_authenticated: bool = True


def test_sla_operator_permission_accepts_system_operator() -> None:
    request = APIRequestFactory().post("/slas/")
    request.user = UserStub(role="system_operator")

    assert IsSLAOperatorRole().has_permission(request, view=None)


def test_sla_operator_permission_rejects_citizen() -> None:
    request = APIRequestFactory().post("/slas/")
    request.user = UserStub(role="citizen")

    assert not IsSLAOperatorRole().has_permission(request, view=None)
