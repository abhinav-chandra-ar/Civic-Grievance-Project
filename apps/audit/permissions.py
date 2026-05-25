"""DRF permissions for audit log reads."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rest_framework.permissions import BasePermission, SAFE_METHODS
from rest_framework.request import Request
from rest_framework.views import APIView

# Audit logs are system governance data — municipal_admin does not have access.
AUDIT_READER_ROLES = frozenset({"super_admin", "system_operator"})


def user_has_audit_role(user: Any, roles: Iterable[str]) -> bool:
    """Check role values without importing the users app model."""
    return bool(
        getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) in frozenset(roles)
    )


class IsAuditReaderRole(BasePermission):
    """Allow read-only audit access to investigation roles."""

    def has_permission(self, request: Request, view: APIView) -> bool:
        return request.method in SAFE_METHODS and user_has_audit_role(request.user, AUDIT_READER_ROLES)
