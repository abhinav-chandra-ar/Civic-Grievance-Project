"""Serializer tests for department service writes."""
from __future__ import annotations

import pytest

from apps.departments.serializers import DepartmentWriteSerializer

pytestmark = pytest.mark.django_db


def test_department_write_serializer_creates_valid_department() -> None:
    serializer = DepartmentWriteSerializer(
        data={
            "code": "public_works",
            "name": "Public Works",
            "handled_categories": ["road_damage", "water_supply"],
            "translated_names": {"ml": "Public Works Malayalam"},
            "escalation_metadata": {"tier": "municipal"},
            "sla_metadata": {"policy_code": "standard"},
        }
    )

    assert serializer.is_valid(), serializer.errors
    department = serializer.save()
    assert department.handles_category("road_damage")
    assert department.is_active


def test_department_write_serializer_rejects_metadata_arrays() -> None:
    serializer = DepartmentWriteSerializer(
        data={
            "code": "water",
            "name": "Water",
            "escalation_metadata": ["tier"],
        }
    )

    assert not serializer.is_valid()
    assert "escalation_metadata" in serializer.errors
