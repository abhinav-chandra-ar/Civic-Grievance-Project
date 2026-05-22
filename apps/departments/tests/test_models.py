"""Model tests for departments."""
from __future__ import annotations

import pytest
from django.core.exceptions import ValidationError

from apps.departments.models import Department


def test_department_accepts_category_codes_and_translated_names() -> None:
    department = Department(
        code="public_works",
        name="Public Works",
        handled_categories=["road_damage", "water_supply"],
        translated_names={"ml": "Public Works Malayalam"},
    )

    department.full_clean()


def test_handled_categories_must_be_category_code_list() -> None:
    department = Department(
        code="waste",
        name="Waste",
        handled_categories={"code": "waste"},
    )

    with pytest.raises(ValidationError) as exc_info:
        department.full_clean()

    assert "handled_categories" in exc_info.value.message_dict


def test_handled_categories_reject_duplicates() -> None:
    department = Department(
        code="roads",
        name="Roads",
        handled_categories=["road_damage", "road_damage"],
    )

    with pytest.raises(ValidationError) as exc_info:
        department.full_clean()

    assert "handled_categories" in exc_info.value.message_dict


def test_translated_names_must_be_mapping() -> None:
    department = Department(code="water", name="Water", translated_names=["ml"])

    with pytest.raises(ValidationError) as exc_info:
        department.full_clean()

    assert "translated_names" in exc_info.value.message_dict
