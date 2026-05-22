"""Selector tests for audit history lookup."""
from __future__ import annotations

import pytest

from apps.audit.selectors import (
    audit_log_list_by_action_type,
    audit_log_list_for_actor,
    audit_log_list_for_target,
    audit_log_list_recent,
)
from apps.audit.services import record_audit_event, record_system_audit_event

pytestmark = pytest.mark.django_db


def test_audit_selectors_filter_actor_target_action_and_recent(django_user_model) -> None:
    actor = django_user_model.objects.create_user(username="operator", password="password")
    actor_log = record_audit_event(
        actor=actor,
        target_model="grievances.Grievance",
        target_object_id="42",
        action_type="update",
    )
    record_system_audit_event(target_model="slas.SLA", target_object_id="7")

    assert list(audit_log_list_for_actor(actor=actor)) == [actor_log]
    assert list(
        audit_log_list_for_target(target_model="grievances.Grievance", target_object_id="42")
    ) == [actor_log]
    assert list(audit_log_list_by_action_type(action_type="update")) == [actor_log]
    assert actor_log in list(audit_log_list_recent(limit=10))
