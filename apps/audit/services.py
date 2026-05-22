"""Create-only services for audit logging."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from .models import AuditActionType, AuditLog
from .signals import audit_log_recorded

AUDIT_SEQUENCE_PATTERN = re.compile(r"^AUD-(?P<year>\d{4})-(?P<sequence>\d{6})$")


def format_audit_code(*, year: int, sequence: int) -> str:
    """Format an internal audit code."""
    if sequence < 1 or sequence > 999999:
        raise ValidationError("Audit code sequence is out of range.")
    return f"AUD-{year:04d}-{sequence:06d}"


def generate_audit_code(*, created_at=None) -> str:
    """Generate the next year-scoped audit code."""
    created_at = created_at or timezone.now()
    year = timezone.localtime(created_at).year
    latest_code = (
        AuditLog.objects.select_for_update()
        .filter(audit_code__startswith=f"AUD-{year:04d}-")
        .order_by("-audit_code")
        .values_list("audit_code", flat=True)
        .first()
    )
    if latest_code is None:
        return format_audit_code(year=year, sequence=1)

    match = AUDIT_SEQUENCE_PATTERN.fullmatch(latest_code)
    if match is None:
        raise ValidationError("Existing audit code cannot be sequenced.")
    return format_audit_code(year=year, sequence=int(match["sequence"]) + 1)


@transaction.atomic
def record_audit_event(
    *,
    actor,
    target_model: str,
    target_object_id: str,
    action_type: str | AuditActionType,
    request_metadata: Mapping[str, Any] | None = None,
    change_metadata: Mapping[str, Any] | None = None,
    security_metadata: Mapping[str, Any] | None = None,
    remarks: str = "",
) -> AuditLog:
    """Record one append-oriented audit event."""
    audit_log = AuditLog(
        audit_code=generate_audit_code(),
        actor=actor,
        target_model=target_model,
        target_object_id=str(target_object_id),
        action_type=str(action_type),
        request_metadata=dict(request_metadata or {}),
        change_metadata=dict(change_metadata or {}),
        security_metadata=dict(security_metadata or {}),
        remarks=remarks,
    )
    audit_log.full_clean()
    audit_log.save()
    audit_log_recorded.send(sender=AuditLog, audit_log=audit_log)
    return audit_log


def record_system_audit_event(
    *,
    target_model: str,
    target_object_id: str,
    action_type: str | AuditActionType = AuditActionType.SYSTEM_EVENT,
    request_metadata: Mapping[str, Any] | None = None,
    change_metadata: Mapping[str, Any] | None = None,
    security_metadata: Mapping[str, Any] | None = None,
    remarks: str = "",
) -> AuditLog:
    """Record a system-generated audit event with no actor."""
    return record_audit_event(
        actor=None,
        target_model=target_model,
        target_object_id=target_object_id,
        action_type=action_type,
        request_metadata=request_metadata,
        change_metadata=change_metadata,
        security_metadata=security_metadata,
        remarks=remarks,
    )
