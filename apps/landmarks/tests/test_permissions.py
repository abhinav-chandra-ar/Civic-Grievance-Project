"""Permission tests for landmark endpoints."""
from __future__ import annotations

from dataclasses import dataclass

from rest_framework.test import APIRequestFactory

from apps.landmarks.permissions import IsLandmarkManagerRole


@dataclass
class UserStub:
    role: str
    is_authenticated: bool = True


def test_landmark_manager_permission_accepts_super_admin() -> None:
    request = APIRequestFactory().post("/landmarks/")
    request.user = UserStub(role="super_admin")

    assert IsLandmarkManagerRole().has_permission(request, view=None)


def test_landmark_manager_permission_rejects_field_verifier() -> None:
    request = APIRequestFactory().post("/landmarks/")
    request.user = UserStub(role="field_verifier")

    assert not IsLandmarkManagerRole().has_permission(request, view=None)
