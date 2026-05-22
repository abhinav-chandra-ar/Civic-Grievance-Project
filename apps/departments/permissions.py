"""DRF permissions for department endpoints."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rest_framework.permissions import BasePermission
from rest_framework.request import Request
from rest_framework.views import APIView

DEPARTMENT_MANAGER_ROLES = frozenset({"municipal_admin", "super_admin", "system_operator"})


def user_has_department_role(user: Any, roles: Iterable[str]) -> bool:
    """Check a role without coupling departments to another app's model."""
    return bool(
        getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) in frozenset(roles)
    )


class IsDepartmentManagerRole(BasePermission):
    """Allow department maintenance to administrative operational roles."""

    def has_permission(self, request: Request, view: APIView) -> bool:
        return user_has_department_role(request.user, DEPARTMENT_MANAGER_ROLES)
