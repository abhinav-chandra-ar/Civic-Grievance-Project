"""Permission tests for ward endpoints."""
from __future__ import annotations

from dataclasses import dataclass

from rest_framework.test import APIRequestFactory

from apps.wards.permissions import IsWardManagerRole


@dataclass
class UserStub:
    role: str
    is_authenticated: bool = True


def test_ward_manager_permission_accepts_municipal_admin() -> None:
    request = APIRequestFactory().post("/wards/")
    request.user = UserStub(role="municipal_admin")

    assert IsWardManagerRole().has_permission(request, view=None)


def test_ward_manager_permission_rejects_ward_officer() -> None:
    request = APIRequestFactory().post("/wards/")
    request.user = UserStub(role="ward_officer")

    assert not IsWardManagerRole().has_permission(request, view=None)
