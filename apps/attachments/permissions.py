"""DRF permissions for attachment endpoints."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rest_framework.permissions import BasePermission
from rest_framework.request import Request
from rest_framework.views import APIView

from .models import Attachment

ATTACHMENT_OPERATOR_ROLES = frozenset(
    {
        "ward_officer",
        "department_officer",
        "municipal_admin",
        "super_admin",
        "field_verifier",
        "system_operator",
    }
)


def user_has_attachment_role(user: Any, roles: Iterable[str]) -> bool:
    """Check role values without importing the users app."""
    return bool(
        getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) in frozenset(roles)
    )


class IsAttachmentOperatorRole(BasePermission):
    """Allow metadata validation updates to operational roles."""

    def has_permission(self, request: Request, view: APIView) -> bool:
        return user_has_attachment_role(request.user, ATTACHMENT_OPERATOR_ROLES)


class IsAttachmentOwnerOrOperatorRole(BasePermission):
    """Allow reads to grievance submitters, uploaders, and operators."""

    def has_object_permission(self, request: Request, view: APIView, obj: Attachment) -> bool:
        return bool(
            getattr(request.user, "is_authenticated", False)
            and (
                obj.uploader_id == request.user.pk
                or obj.grievance.submitter_id == request.user.pk
                or user_has_attachment_role(request.user, ATTACHMENT_OPERATOR_ROLES)
            )
        )
