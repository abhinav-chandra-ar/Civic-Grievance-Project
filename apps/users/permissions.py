"""Role-aware DRF permissions for user endpoints."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rest_framework.permissions import BasePermission
from rest_framework.request import Request
from rest_framework.views import APIView

from .models import User, UserRole

# Both admin tiers share read access to the user directory.
ADMIN_ROLES = frozenset({UserRole.MUNICIPAL_ADMIN, UserRole.SUPER_ADMIN})

OFFICER_ROLES = frozenset(
    {
        UserRole.WARD_OFFICER,
        UserRole.DEPARTMENT_OFFICER,
        UserRole.FIELD_VERIFIER,
    }
)


def user_has_any_role(user: Any, roles: Iterable[str | UserRole]) -> bool:
    """Return whether an authenticated user has one of the requested roles."""
    return bool(
        getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) in {str(role) for role in roles}
    )


class HasUserRole(BasePermission):
    """Base permission for views that declare allowed_roles."""

    allowed_roles: frozenset[str | UserRole] = frozenset()

    def has_permission(self, request: Request, view: APIView) -> bool:
        return user_has_any_role(request.user, self.allowed_roles)


class IsUserAdminRole(HasUserRole):
    """Allow both admin tiers to read the user directory."""

    allowed_roles = ADMIN_ROLES


class IsSuperAdminOnly(HasUserRole):
    """Restrict to super_admin — system governance, audit, security, ML controls."""

    allowed_roles = frozenset({UserRole.SUPER_ADMIN})


class IsMunicipalAdminOnly(HasUserRole):
    """Restrict to municipal_admin — city operations, grievance management, dept oversight."""

    allowed_roles = frozenset({UserRole.MUNICIPAL_ADMIN})


class IsOfficerRole(HasUserRole):
    """Allow operational roles that handle or verify complaints."""

    allowed_roles = OFFICER_ROLES


class IsSelfOrUserAdminRole(BasePermission):
    """Allow profile access to the user itself or a user administrator."""

    def has_object_permission(self, request: Request, view: APIView, obj: User) -> bool:
        return bool(
            getattr(request.user, "is_authenticated", False)
            and (request.user.pk == obj.pk or user_has_any_role(request.user, ADMIN_ROLES))
        )
