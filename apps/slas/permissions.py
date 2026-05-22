"""DRF permissions for SLA endpoints."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rest_framework.permissions import BasePermission
from rest_framework.request import Request
from rest_framework.views import APIView

from .models import SLA

SLA_OPERATOR_ROLES = frozenset(
    {
        "ward_officer",
        "department_officer",
        "municipal_admin",
        "super_admin",
        "field_verifier",
        "system_operator",
    }
)


def user_has_sla_role(user: Any, roles: Iterable[str]) -> bool:
    """Check role values without importing the users app model."""
    return bool(
        getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) in frozenset(roles)
    )


class IsSLAOperatorRole(BasePermission):
    """Allow SLA maintenance to operational roles."""

    def has_permission(self, request: Request, view: APIView) -> bool:
        return user_has_sla_role(request.user, SLA_OPERATOR_ROLES)


class IsSLAVisible(BasePermission):
    """Allow reads to grievance submitters and operational roles."""

    def has_object_permission(self, request: Request, view: APIView, obj: SLA) -> bool:
        return bool(
            getattr(request.user, "is_authenticated", False)
            and (
                obj.grievance.submitter_id == request.user.pk
                or user_has_sla_role(request.user, SLA_OPERATOR_ROLES)
            )
        )
