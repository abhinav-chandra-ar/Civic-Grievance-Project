"""Serializer tests for SLA creation and breach marking."""
from __future__ import annotations

import pytest
from django.utils import timezone

from apps.grievances.services import submit_grievance
from apps.slas.serializers import SLABreachMarkSerializer, SLACreateSerializer

pytestmark = pytest.mark.django_db


def test_create_serializer_creates_sla(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Road blocked.")
    now = timezone.now()
    serializer = SLACreateSerializer(
        data={
            "grievance": grievance.pk,
            "response_due_at": now + timezone.timedelta(hours=1),
            "resolution_due_at": now + timezone.timedelta(days=1),
            "policy_snapshot_metadata": {"policy": "p1"},
        }
    )

    assert serializer.is_valid(), serializer.errors
    sla = serializer.save()
    assert sla.grievance == grievance
    assert sla.sla_code.startswith("SLA-")


def test_breach_serializer_marks_breach(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Waste issue.")
    now = timezone.now()
    create_serializer = SLACreateSerializer(
        data={
            "grievance": grievance.pk,
            "response_due_at": now + timezone.timedelta(hours=1),
            "resolution_due_at": now + timezone.timedelta(days=1),
        }
    )
    assert create_serializer.is_valid(), create_serializer.errors
    sla = create_serializer.save()
    breach_serializer = SLABreachMarkSerializer(
        sla,
        data={"breach_type": "resolution", "breach_metadata": {"source": "manual"}},
        partial=True,
    )

    assert breach_serializer.is_valid(), breach_serializer.errors
    updated = breach_serializer.save()
    assert updated.is_breached
    assert updated.breach_type == "resolution"
