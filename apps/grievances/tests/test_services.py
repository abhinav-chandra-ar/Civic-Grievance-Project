"""Service tests for grievance submission and duplicate hooks."""
from __future__ import annotations

import pytest

from apps.grievances.services import change_grievance_status, submit_grievance

pytestmark = pytest.mark.django_db


def test_submission_generates_year_scoped_tracking_codes(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")

    first = submit_grievance(submitter=submitter, raw_text="Road damage near school.")
    second = submit_grievance(submitter=submitter, raw_text="Overflowing drain.")

    assert first.tracking_code.startswith("GRV-")
    assert first.tracking_code.endswith("000001")
    assert second.tracking_code.endswith("000002")


def test_possible_duplicate_link_is_nullable_self_relation(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    original = submit_grievance(submitter=submitter, raw_text="Street light broken.")
    duplicate = submit_grievance(submitter=submitter, raw_text="Same street light broken.")
    duplicate.possible_duplicate_of = original
    duplicate.full_clean()
    duplicate.save(update_fields=["possible_duplicate_of"])

    assert duplicate.possible_duplicate_of == original


def test_status_change_records_audit_safe_status_fields(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Water supply interrupted.")

    change_grievance_status(
        grievance=grievance,
        status="triaged",
        reason="Mapped for review.",
        metadata={"source": "manual"},
    )

    assert grievance.status == "triaged"
    assert grievance.status_reason == "Mapped for review."
    assert grievance.last_status_changed_at is not None
