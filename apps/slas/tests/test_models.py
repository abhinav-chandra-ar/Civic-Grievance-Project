"""Model tests for SLA state."""
from __future__ import annotations

import pytest
from django.core.exceptions import ValidationError
from django.utils import timezone

from apps.grievances.services import submit_grievance
from apps.slas.models import SLA, SLABreachType, SLAStatus

pytestmark = pytest.mark.django_db


def test_sla_status_and_breach_type_choices_are_simple() -> None:
    assert set(SLAStatus.values) == {"active", "breached", "paused", "satisfied", "cancelled"}
    assert set(SLABreachType.values) == {"none", "response", "resolution", "both"}


def test_sla_rejects_resolution_before_response(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Drain blocked.")
    now = timezone.now()
    sla = SLA(
        sla_code="SLA-2026-000001",
        grievance=grievance,
        response_due_at=now,
        resolution_due_at=now - timezone.timedelta(hours=1),
    )

    with pytest.raises(ValidationError) as exc_info:
        sla.full_clean()

    assert "resolution_due_at" in exc_info.value.message_dict


def test_breached_sla_requires_breach_type_and_time(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Road damaged.")
    now = timezone.now()
    sla = SLA(
        sla_code="SLA-2026-000001",
        grievance=grievance,
        response_due_at=now,
        resolution_due_at=now + timezone.timedelta(hours=1),
        is_breached=True,
    )

    with pytest.raises(ValidationError) as exc_info:
        sla.full_clean()

    assert "breached_at" in exc_info.value.message_dict
    assert "breach_type" in exc_info.value.message_dict


def test_metadata_hooks_must_be_mappings(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Water interrupted.")
    now = timezone.now()
    sla = SLA(
        sla_code="SLA-2026-000001",
        grievance=grievance,
        response_due_at=now,
        resolution_due_at=now + timezone.timedelta(hours=1),
        breach_metadata=["bad"],
    )

    with pytest.raises(ValidationError) as exc_info:
        sla.full_clean()

    assert "breach_metadata" in exc_info.value.message_dict
