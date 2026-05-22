"""Read-side queries for audit logs."""
from __future__ import annotations

from .models import AuditLog


def audit_log_list():
    """Return audit logs with actor loaded for investigation views."""
    return AuditLog.objects.select_related("actor")


def audit_log_list_for_actor(*, actor):
    """Return audit history for one actor."""
    return audit_log_list().filter(actor=actor)


def audit_log_list_for_target(*, target_model: str, target_object_id: str):
    """Return audit history for a referenced object."""
    return audit_log_list().filter(target_model=target_model, target_object_id=str(target_object_id))


def audit_log_list_by_action_type(*, action_type: str):
    """Return audit logs for one action type."""
    return audit_log_list().filter(action_type=action_type)


def audit_log_list_recent(*, since=None, limit: int = 100):
    """Return recent audit activity, optionally after a timestamp."""
    logs = audit_log_list()
    if since is not None:
        logs = logs.filter(created_at__gte=since)
    return logs[:limit]
