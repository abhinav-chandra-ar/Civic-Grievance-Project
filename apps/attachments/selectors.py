"""Read-side queries for attachments."""
from __future__ import annotations

from typing import Any

from django.db.models import QuerySet

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


def attachment_list(*, active_only: bool = True) -> QuerySet[Attachment]:
    """Return attachments with parent and uploader relations loaded."""
    attachments = Attachment.objects.select_related("grievance", "uploader")
    if active_only:
        return attachments.filter(is_active=True)
    return attachments


def attachment_list_for_grievance(
    *, grievance, active_only: bool = True
) -> QuerySet[Attachment]:
    """Return attachments registered for one grievance."""
    return attachment_list(active_only=active_only).filter(grievance=grievance)


def attachment_list_by_content_hash(
    *, content_hash: str, active_only: bool = True
) -> QuerySet[Attachment]:
    """Return hash matches for future duplicate-image detection."""
    return attachment_list(active_only=active_only).filter(content_hash=content_hash)


def attachment_list_visible_to_user(*, user: Any) -> QuerySet[Attachment]:
    """Return uploader/grievance-owned rows unless the user is operational."""
    attachments = attachment_list()
    if getattr(user, "role", None) in ATTACHMENT_OPERATOR_ROLES:
        return attachments
    return attachments.filter(grievance__submitter=user)
