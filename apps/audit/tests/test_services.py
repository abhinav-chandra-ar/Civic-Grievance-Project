"""Service tests for audit event recording."""
from __future__ import annotations

import pytest

from apps.audit.models import AuditActionType
from apps.audit.services import record_audit_event, record_system_audit_event

pytestmark = pytest.mark.django_db


def test_record_audit_event_generates_code_and_preserves_actor(django_user_model) -> None:
    actor = django_user_model.objects.create_user(username="operator", password="password")

    audit_log = record_audit_event(
        actor=actor,
        target_model="grievances.Grievance",
        target_object_id="1",
        action_type=AuditActionType.STATUS_CHANGE,
        request_metadata={"path": "/api/v1/grievances/1/"},
    )

    assert audit_log.audit_code.startswith("AUD-")
    assert audit_log.actor == actor
    assert audit_log.action_type == AuditActionType.STATUS_CHANGE


def test_record_system_audit_event_has_null_actor() -> None:
    audit_log = record_system_audit_event(
        target_model="slas.SLA",
        target_object_id="1",
        remarks="Scheduled breach check completed.",
    )

    assert audit_log.actor is None
    assert audit_log.action_type == AuditActionType.SYSTEM_EVENT
