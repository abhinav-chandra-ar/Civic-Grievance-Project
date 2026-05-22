"""Serializer tests for service-backed landmark writes."""
from __future__ import annotations

import pytest

from apps.landmarks.models import LandmarkType
from apps.landmarks.serializers import LandmarkWriteSerializer

pytestmark = pytest.mark.django_db


def landmark_feature() -> dict[str, object]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [76.27, 9.98]},
        "properties": {
            "code": "central_market",
            "primary_name": "Central Market",
            "aliases_en": ["Main Market"],
            "aliases_ml": ["Central Market Malayalam"],
            "landmark_type": LandmarkType.MARKET,
            "normalized_search_tokens": ["central", "market"],
            "fuzzy_lookup_metadata": {"source": "curated"},
            "ward_mapping_metadata": {"mapping_policy": "future"},
        },
    }


def test_landmark_write_serializer_creates_point_landmark() -> None:
    serializer = LandmarkWriteSerializer(data=landmark_feature())

    assert serializer.is_valid(), serializer.errors
    landmark = serializer.save()
    assert landmark.location.srid == 4326
    assert landmark.location.valid


def test_landmark_write_serializer_rejects_token_objects() -> None:
    payload = landmark_feature()
    payload["properties"]["normalized_search_tokens"] = [{"value": "market"}]  # type: ignore[index]
    serializer = LandmarkWriteSerializer(data=payload)

    assert not serializer.is_valid()
    assert "normalized_search_tokens" in serializer.errors
