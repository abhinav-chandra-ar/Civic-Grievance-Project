"""Serializer tests for grievance submission and enrichment role enforcement."""
from __future__ import annotations

import pytest
from django.contrib.gis.geos import Polygon
from rest_framework.test import APIRequestFactory

from apps.departments.models import Department
from apps.grievances.serializers import GrievanceEnrichmentSerializer, GrievanceSubmitSerializer
from apps.grievances.services import submit_grievance
from apps.wards.models import Ward

pytestmark = pytest.mark.django_db

_BOUNDARY = Polygon(
    ((76.90, 8.40), (76.95, 8.40), (76.95, 8.45), (76.90, 8.45), (76.90, 8.40)),
    srid=4326,
)


# ---------------------------------------------------------------------------
# Submission tests (existing)
# ---------------------------------------------------------------------------

def test_submit_serializer_creates_grievance_with_tracking_code(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    serializer = GrievanceSubmitSerializer(
        data={
            "raw_text": "Waste pile near junction.",
            "landmark_mention": "Main junction",
            "image_attachment_metadata": {"reference": "pending-upload"},
        }
    )

    assert serializer.is_valid(), serializer.errors
    grievance = serializer.save(submitter=submitter)
    assert grievance.tracking_code.startswith("GRV-")
    assert grievance.submitter == submitter


def test_submit_serializer_rejects_empty_raw_text() -> None:
    serializer = GrievanceSubmitSerializer(data={"raw_text": " "})

    assert not serializer.is_valid()
    assert "raw_text" in serializer.errors


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _enrichment_serializer(grievance, user, data):
    request = APIRequestFactory().patch(f"/grievances/{grievance.pk}/")
    request.user = user
    return GrievanceEnrichmentSerializer(
        grievance, data=data, partial=True, context={"request": request}
    )


# ---------------------------------------------------------------------------
# Task 2 — ward_officer routing field restrictions
# ---------------------------------------------------------------------------

def test_ward_officer_cannot_change_department(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="wo1", password="password", role="ward_officer"
    )
    dept = Department.objects.create(code="roads", name="Roads")
    grievance = submit_grievance(submitter=actor, raw_text="Road broken.")

    serializer = _enrichment_serializer(grievance, actor, {"department": dept.pk})
    assert not serializer.is_valid()
    assert "department" in serializer.errors


def test_ward_officer_cannot_change_ward(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="wo2", password="password", role="ward_officer"
    )
    ward = Ward.objects.create(code="tvm_001", name="Ward 1", boundary=_BOUNDARY)
    grievance = submit_grievance(submitter=actor, raw_text="Light broken.")

    serializer = _enrichment_serializer(grievance, actor, {"ward": ward.pk})
    assert not serializer.is_valid()
    assert "ward" in serializer.errors


def test_ward_officer_cannot_change_category_code(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="wo3", password="password", role="ward_officer"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Sewage overflow.")

    serializer = _enrichment_serializer(grievance, actor, {"category_code": "water_supply"})
    assert not serializer.is_valid()
    assert "category_code" in serializer.errors


def test_ward_officer_can_raise_priority(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="wo4", password="password", role="ward_officer"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Pothole.")
    # Force to medium so the AI classifier result doesn't affect the raise test.
    grievance.priority = "medium"
    grievance.save(update_fields=["priority"])

    serializer = _enrichment_serializer(grievance, actor, {"priority": "high"})
    assert serializer.is_valid(), serializer.errors


def test_ward_officer_cannot_lower_priority(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="wo5", password="password", role="ward_officer"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Minor scratch on road.")
    grievance.priority = "high"
    grievance.save(update_fields=["priority"])

    serializer = _enrichment_serializer(grievance, actor, {"priority": "low"})
    assert not serializer.is_valid()
    assert "priority" in serializer.errors


# ---------------------------------------------------------------------------
# Task 3 & 4 — department_officer category bounding and routing restriction
# ---------------------------------------------------------------------------

def test_dept_officer_cannot_change_department_fk(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="do1", password="password", role="department_officer"
    )
    target_dept = Department.objects.create(code="water", name="Water Authority")
    grievance = submit_grievance(submitter=actor, raw_text="Water pipe broken.")

    serializer = _enrichment_serializer(grievance, actor, {"department": target_dept.pk})
    assert not serializer.is_valid()
    assert "department" in serializer.errors


def test_dept_officer_cannot_change_ward_fk(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="do2", password="password", role="department_officer"
    )
    ward = Ward.objects.create(code="tvm_002", name="Ward 2", boundary=_BOUNDARY)
    grievance = submit_grievance(submitter=actor, raw_text="Drain blocked.")

    serializer = _enrichment_serializer(grievance, actor, {"ward": ward.pk})
    assert not serializer.is_valid()
    assert "ward" in serializer.errors


def test_dept_officer_category_outside_scope_rejected(django_user_model) -> None:
    dept = Department.objects.create(
        code="electrical",
        name="Electrical",
        handled_categories=["street_light", "electrical_hazard"],
    )
    actor = django_user_model.objects.create_user(
        username="do3", password="password", role="department_officer"
    )
    actor.assigned_department = dept
    actor.save(update_fields=["assigned_department"])
    grievance = submit_grievance(submitter=actor, raw_text="Street light problem.")

    # water_supply is outside electrical department scope
    serializer = _enrichment_serializer(grievance, actor, {"category_code": "water_supply"})
    assert not serializer.is_valid()
    assert "category_code" in serializer.errors


def test_dept_officer_category_within_scope_accepted(django_user_model) -> None:
    dept = Department.objects.create(
        code="electrical2",
        name="Electrical Dept 2",
        handled_categories=["street_light", "electrical_hazard"],
    )
    actor = django_user_model.objects.create_user(
        username="do4", password="password", role="department_officer"
    )
    actor.assigned_department = dept
    actor.save(update_fields=["assigned_department"])
    grievance = submit_grievance(submitter=actor, raw_text="Street light out.")

    serializer = _enrichment_serializer(grievance, actor, {"category_code": "street_light"})
    assert serializer.is_valid(), serializer.errors


# ---------------------------------------------------------------------------
# Task 5 — priority direction rule for department_officer
# ---------------------------------------------------------------------------

def test_dept_officer_lowering_priority_requires_reason(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="do5", password="password", role="department_officer"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Urgent drain.")
    grievance.priority = "urgent"
    grievance.save(update_fields=["priority"])

    # No status_reason provided — should reject
    serializer = _enrichment_serializer(grievance, actor, {"priority": "medium"})
    assert not serializer.is_valid()
    assert "status_reason" in serializer.errors


def test_dept_officer_lowering_priority_with_reason_accepted(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="do6", password="password", role="department_officer"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Minor issue.")
    grievance.priority = "urgent"
    grievance.save(update_fields=["priority"])

    serializer = _enrichment_serializer(
        grievance,
        actor,
        {
            "priority": "medium",
            "status_reason": "Field inspection confirmed low severity — no immediate danger.",
        },
    )
    assert serializer.is_valid(), serializer.errors


# ---------------------------------------------------------------------------
# Task 6 — municipal_admin has no enrichment restrictions
# ---------------------------------------------------------------------------

def test_municipal_admin_can_change_department(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="madmin1", password="password", role="municipal_admin"
    )
    dept = Department.objects.create(code="health_dept", name="Health Dept")
    grievance = submit_grievance(submitter=actor, raw_text="Health centre issue.")

    serializer = _enrichment_serializer(grievance, actor, {"department": dept.pk})
    assert serializer.is_valid(), serializer.errors


def test_municipal_admin_can_change_ward(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="madmin2", password="password", role="municipal_admin"
    )
    ward = Ward.objects.create(code="tvm_010", name="Ward 10", boundary=_BOUNDARY)
    grievance = submit_grievance(submitter=actor, raw_text="Ward issue.")

    serializer = _enrichment_serializer(grievance, actor, {"ward": ward.pk})
    assert serializer.is_valid(), serializer.errors


def test_municipal_admin_can_lower_priority_without_reason(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="madmin3", password="password", role="municipal_admin"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Overstated complaint.")
    grievance.priority = "critical"
    grievance.save(update_fields=["priority"])

    serializer = _enrichment_serializer(grievance, actor, {"priority": "low"})
    assert serializer.is_valid(), serializer.errors


def test_municipal_admin_can_change_category_freely(django_user_model) -> None:
    actor = django_user_model.objects.create_user(
        username="madmin4", password="password", role="municipal_admin"
    )
    grievance = submit_grievance(submitter=actor, raw_text="Misclassified complaint.")

    serializer = _enrichment_serializer(grievance, actor, {"category_code": "road_damage"})
    assert serializer.is_valid(), serializer.errors
