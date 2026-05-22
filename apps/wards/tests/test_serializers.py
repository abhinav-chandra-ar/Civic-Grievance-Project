"""Serializer tests for service-backed ward writes."""
from __future__ import annotations

import pytest

from apps.wards.serializers import WardWriteSerializer

pytestmark = pytest.mark.django_db


def ward_feature() -> dict[str, object]:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [76.0, 10.0],
                [76.0, 10.1],
                [76.1, 10.1],
                [76.1, 10.0],
                [76.0, 10.0],
            ]],
        },
        "properties": {
            "code": "ward_01",
            "name": "Ward 01",
            "translated_names": {"ml": "Ward 01 Malayalam"},
            "officer_assignment_metadata": {"assignment_policy": "future"},
            "landmark_mapping_metadata": {"mapping_policy": "future"},
        },
    }


def test_ward_write_serializer_creates_polygon_ward() -> None:
    serializer = WardWriteSerializer(data=ward_feature())

    assert serializer.is_valid(), serializer.errors
    ward = serializer.save()
    assert ward.boundary.srid == 4326
    assert ward.boundary.valid


def test_ward_write_serializer_rejects_metadata_arrays() -> None:
    payload = ward_feature()
    payload["properties"]["landmark_mapping_metadata"] = ["landmark"]  # type: ignore[index]
    serializer = WardWriteSerializer(data=payload)

    assert not serializer.is_valid()
    assert "landmark_mapping_metadata" in serializer.errors
