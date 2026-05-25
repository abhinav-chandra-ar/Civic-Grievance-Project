"""DRF permissions for ward endpoints."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rest_framework.permissions import BasePermission
from rest_framework.request import Request
from rest_framework.views import APIView

# Ward boundary create/edit (GIS import) is platform-level; municipal_admin has read-only oversight.
WARD_MANAGER_ROLES = frozenset({"super_admin", "system_operator"})


def user_has_ward_role(user: Any, roles: Iterable[str]) -> bool:
    """Check roles without importing a concrete user model."""
    return bool(
        getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) in frozenset(roles)
    )


class IsWardManagerRole(BasePermission):
    """Allow administrative roles to maintain ward boundaries."""

    def has_permission(self, request: Request, view: APIView) -> bool:
        return user_has_ward_role(request.user, WARD_MANAGER_ROLES)
