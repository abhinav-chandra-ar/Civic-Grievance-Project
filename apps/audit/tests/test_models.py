"""Model tests for audit logs."""
from __future__ import annotations

import pytest
from django.core.exceptions import ValidationError

from apps.audit.models import AuditActionType, AuditLog


def test_audit_action_types_are_closed_and_no_read_action() -> None:
    assert set(AuditActionType.values) == {
        "create",
        "update",
        "delete",
        "login",
        "logout",
        "permission_denied",
        "status_change",
        "assignment",
        "escalation",
        "breach",
        "export",
        "admin_action",
        "system_event",
    }
    assert "read" not in AuditActionType.values


def test_audit_log_rejects_blank_target_reference() -> None:
    audit_log = AuditLog(
        audit_code="AUD-2026-000001",
        target_model=" ",
        target_object_id="1",
        action_type=AuditActionType.CREATE,
    )

    with pytest.raises(ValidationError) as exc_info:
        audit_log.full_clean()

    assert "target_model" in exc_info.value.message_dict


def test_audit_metadata_hooks_must_be_mappings() -> None:
    audit_log = AuditLog(
        audit_code="AUD-2026-000001",
        target_model="grievances.Grievance",
        target_object_id="1",
        action_type=AuditActionType.UPDATE,
        change_metadata=["bad"],
    )

    with pytest.raises(ValidationError) as exc_info:
        audit_log.full_clean()

    assert "change_metadata" in exc_info.value.message_dict
