"""DRF permissions for landmark endpoints."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rest_framework.permissions import BasePermission
from rest_framework.request import Request
from rest_framework.views import APIView

LANDMARK_MANAGER_ROLES = frozenset({"municipal_admin", "super_admin", "system_operator"})


def user_has_landmark_role(user: Any, roles: Iterable[str]) -> bool:
    """Check roles without importing the users app model."""
    return bool(
        getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) in frozenset(roles)
    )


class IsLandmarkManagerRole(BasePermission):
    """Allow administrative roles to maintain landmark catalog data."""

    def has_permission(self, request: Request, view: APIView) -> bool:
        return user_has_landmark_role(request.user, LANDMARK_MANAGER_ROLES)
