"""Selector tests for SLA deadline lookup."""
from __future__ import annotations

import pytest
from django.utils import timezone

from apps.grievances.services import submit_grievance
from apps.slas.selectors import (
    sla_get_for_grievance,
    sla_list_requiring_breach_check,
    sla_list_with_escalation_due,
    sla_list_with_upcoming_response_deadline,
)
from apps.slas.services import create_sla_for_grievance, mark_sla_breached

pytestmark = pytest.mark.django_db


def test_selectors_find_grievance_sla_and_upcoming_deadlines(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Streetlight failed.")
    now = timezone.now()
    sla = create_sla_for_grievance(
        grievance=grievance,
        response_due_at=now + timezone.timedelta(minutes=30),
        resolution_due_at=now + timezone.timedelta(days=1),
        next_escalation_due_at=now - timezone.timedelta(minutes=1),
    )

    assert sla_get_for_grievance(grievance=grievance) == sla
    assert list(sla_list_with_upcoming_response_deadline(before=now + timezone.timedelta(hours=1))) == [sla]
    assert list(sla_list_with_escalation_due(now=now)) == [sla]


def test_breached_rows_are_excluded_from_breach_check(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Drain overflow.")
    now = timezone.now()
    sla = create_sla_for_grievance(
        grievance=grievance,
        response_due_at=now - timezone.timedelta(hours=1),
        resolution_due_at=now + timezone.timedelta(days=1),
    )
    mark_sla_breached(sla=sla, breach_type="response")

    assert list(sla_list_requiring_breach_check(now=now)) == []
