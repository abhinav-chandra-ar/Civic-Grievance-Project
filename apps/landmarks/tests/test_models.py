"""Model tests for landmark geometry and search metadata."""
from __future__ import annotations

import pytest
from django.contrib.gis.geos import Point
from django.core.exceptions import ValidationError

from apps.landmarks.models import Landmark, LandmarkType


def test_landmark_accepts_point_aliases_and_search_tokens() -> None:
    landmark = Landmark(
        code="central_market",
        primary_name="Central Market",
        location=Point(76.27, 9.98, srid=4326),
        landmark_type=LandmarkType.MARKET,
        aliases_en=["Main Market"],
        aliases_ml=["Central Market Malayalam"],
        normalized_search_tokens=["central", "market"],
    )

    landmark.full_clean()


def test_landmark_rejects_empty_point() -> None:
    landmark = Landmark(code="market", primary_name="Market", location=Point())

    with pytest.raises(ValidationError) as exc_info:
        landmark.full_clean()

    assert "location" in exc_info.value.message_dict


def test_landmark_rejects_empty_alias_values() -> None:
    landmark = Landmark(
        code="market",
        primary_name="Market",
        location=Point(76.27, 9.98, srid=4326),
        aliases_en=[""],
    )

    with pytest.raises(ValidationError) as exc_info:
        landmark.full_clean()

    assert "aliases_en" in exc_info.value.message_dict


def test_landmark_type_uses_closed_choices() -> None:
    landmark = Landmark(
        code="market",
        primary_name="Market",
        location=Point(76.27, 9.98, srid=4326),
        landmark_type="free_text_type",
    )

    with pytest.raises(ValidationError) as exc_info:
        landmark.full_clean()

    assert "landmark_type" in exc_info.value.message_dict


def test_landmark_metadata_hooks_must_be_mappings() -> None:
    landmark = Landmark(
        code="market",
        primary_name="Market",
        location=Point(76.27, 9.98, srid=4326),
        fuzzy_lookup_metadata=["future"],
    )

    with pytest.raises(ValidationError) as exc_info:
        landmark.full_clean()

    assert "fuzzy_lookup_metadata" in exc_info.value.message_dict
