"""Service tests for SLA state updates."""
from __future__ import annotations

import pytest
from django.utils import timezone

from apps.grievances.services import submit_grievance
from apps.slas.models import SLABreachType, SLAStatus
from apps.slas.services import compute_breach_type, create_sla_for_grievance, mark_sla_breached

pytestmark = pytest.mark.django_db


def test_create_sla_generates_code_and_one_to_one_row(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Waste not collected.")
    now = timezone.now()

    sla = create_sla_for_grievance(
        grievance=grievance,
        response_due_at=now + timezone.timedelta(hours=2),
        resolution_due_at=now + timezone.timedelta(days=2),
        policy_snapshot_metadata={"policy": "standard"},
    )

    assert sla.sla_code.startswith("SLA-")
    assert sla.grievance == grievance
    assert sla.policy_snapshot_metadata == {"policy": "standard"}


def test_mark_sla_breached_sets_explicit_breach_state(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Road crater.")
    now = timezone.now()
    sla = create_sla_for_grievance(
        grievance=grievance,
        response_due_at=now + timezone.timedelta(hours=2),
        resolution_due_at=now + timezone.timedelta(days=2),
    )

    mark_sla_breached(
        sla=sla,
        breach_type=SLABreachType.RESPONSE,
        breach_metadata={"source": "scheduled_check"},
    )

    assert sla.is_breached
    assert sla.sla_status == SLAStatus.BREACHED
    assert sla.breach_type == SLABreachType.RESPONSE
    assert sla.breached_at is not None


def test_compute_breach_type_does_not_mutate_state(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Water issue.")
    now = timezone.now()
    sla = create_sla_for_grievance(
        grievance=grievance,
        response_due_at=now - timezone.timedelta(minutes=5),
        resolution_due_at=now + timezone.timedelta(days=1),
    )

    assert compute_breach_type(sla=sla, now=now) == SLABreachType.RESPONSE
    assert not sla.is_breached
