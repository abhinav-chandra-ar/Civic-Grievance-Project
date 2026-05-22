"""Model tests for grievance foundation fields."""
from __future__ import annotations

import pytest
from django.core.exceptions import ValidationError
from django.utils import timezone

from apps.grievances.models import Grievance, GrievancePriority, GrievanceStatus

pytestmark = pytest.mark.django_db


def test_grievance_rejects_whitespace_raw_text(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = Grievance(
        tracking_code="GRV-2026-000001",
        submitter=submitter,
        raw_text="   ",
        submitted_at=timezone.now(),
    )

    with pytest.raises(ValidationError) as exc_info:
        grievance.full_clean()

    assert "raw_text" in exc_info.value.message_dict


def test_grievance_accepts_metadata_and_closed_defaults(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = Grievance(
        tracking_code="GRV-2026-000001",
        submitter=submitter,
        raw_text="Street light is broken.",
        submitted_at=timezone.now(),
        image_attachment_metadata={"reference": "pending"},
    )

    grievance.full_clean()
    assert grievance.priority == GrievancePriority.MEDIUM
    assert grievance.status == GrievanceStatus.SUBMITTED


def test_metadata_hooks_must_be_mappings(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = Grievance(
        tracking_code="GRV-2026-000001",
        submitter=submitter,
        raw_text="Waste pickup missed.",
        submitted_at=timezone.now(),
        duplicate_detection_metadata=["candidate"],
    )

    with pytest.raises(ValidationError) as exc_info:
        grievance.full_clean()

    assert "duplicate_detection_metadata" in exc_info.value.message_dict
