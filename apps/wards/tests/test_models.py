"""Model tests for ward boundaries."""
from __future__ import annotations

import pytest
from django.contrib.gis.geos import Polygon
from django.core.exceptions import ValidationError

from apps.wards.models import Ward


def square_boundary() -> Polygon:
    return Polygon(((76.0, 10.0), (76.0, 10.1), (76.1, 10.1), (76.1, 10.0), (76.0, 10.0)), srid=4326)


def test_ward_accepts_valid_polygon_and_translated_names() -> None:
    ward = Ward(
        code="ward_01",
        name="Ward 01",
        boundary=square_boundary(),
        translated_names={"ml": "Ward 01 Malayalam"},
    )

    ward.full_clean()


def test_ward_rejects_empty_boundary() -> None:
    ward = Ward(code="ward_01", name="Ward 01", boundary=Polygon())

    with pytest.raises(ValidationError) as exc_info:
        ward.full_clean()

    assert "boundary" in exc_info.value.message_dict


def test_ward_rejects_invalid_polygon() -> None:
    ward = Ward(
        code="ward_01",
        name="Ward 01",
        boundary=Polygon(((0, 0), (1, 1), (1, 0), (0, 1), (0, 0)), srid=4326),
    )

    with pytest.raises(ValidationError) as exc_info:
        ward.full_clean()

    assert "boundary" in exc_info.value.message_dict


def test_ward_metadata_hooks_must_be_mappings() -> None:
    ward = Ward(
        code="ward_01",
        name="Ward 01",
        boundary=square_boundary(),
        officer_assignment_metadata=["officer"],
    )

    with pytest.raises(ValidationError) as exc_info:
        ward.full_clean()

    assert "officer_assignment_metadata" in exc_info.value.message_dict
