"""Permission tests for user role hooks."""
from __future__ import annotations

from rest_framework.test import APIRequestFactory

from apps.users.models import User, UserRole
from apps.users.permissions import IsUserAdminRole, user_has_any_role


def test_role_helper_requires_authenticated_user() -> None:
    assert not user_has_any_role(object(), {UserRole.SUPER_ADMIN})


def test_admin_permission_accepts_municipal_admin() -> None:
    request = APIRequestFactory().get("/users/")
    request.user = User(username="municipal", role=UserRole.MUNICIPAL_ADMIN)

    assert IsUserAdminRole().has_permission(request, view=None)


def test_admin_permission_rejects_citizen() -> None:
    request = APIRequestFactory().get("/users/")
    request.user = User(username="citizen", role=UserRole.CITIZEN)

    assert not IsUserAdminRole().has_permission(request, view=None)
