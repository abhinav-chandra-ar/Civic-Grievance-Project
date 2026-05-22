"""DRF permissions for grievance endpoints."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rest_framework.permissions import BasePermission
from rest_framework.request import Request
from rest_framework.views import APIView

from .models import Grievance

GRIEVANCE_OPERATOR_ROLES = frozenset(
    {
        "ward_officer",
        "department_officer",
        "municipal_admin",
        "super_admin",
        "field_verifier",
        "system_operator",
    }
)


def user_has_grievance_role(user: Any, roles: Iterable[str]) -> bool:
    """Check role values without importing the concrete user model."""
    return bool(
        getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) in frozenset(roles)
    )


class IsGrievanceOperatorRole(BasePermission):
    """Allow enrichment and status maintenance to operational roles."""

    def has_permission(self, request: Request, view: APIView) -> bool:
        return user_has_grievance_role(request.user, GRIEVANCE_OPERATOR_ROLES)


class IsSubmitterOrGrievanceOperatorRole(BasePermission):
    """Allow grievance reads to its submitter or an operational role."""

    def has_object_permission(self, request: Request, view: APIView, obj: Grievance) -> bool:
        return bool(
            getattr(request.user, "is_authenticated", False)
            and (
                obj.submitter_id == request.user.pk
                or user_has_grievance_role(request.user, GRIEVANCE_OPERATOR_ROLES)
            )
        )
