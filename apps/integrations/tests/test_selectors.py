"""Selector tests for integration-local enrichment reads."""
from __future__ import annotations

import pytest
from django.contrib.gis.geos import Point

from apps.integrations.selectors import local_landmark_candidates_for_mention
from apps.landmarks.models import Landmark, LandmarkType

pytestmark = pytest.mark.django_db


def test_local_landmark_candidates_for_mention_returns_read_shape() -> None:
    Landmark.objects.create(
        code="central_market",
        primary_name="Central Market",
        location=Point(76.27, 9.98, srid=4326),
        landmark_type=LandmarkType.MARKET,
    )

    candidates = local_landmark_candidates_for_mention(mention="Central Market")

    assert candidates == [
        {
            "code": "central_market",
            "primary_name": "Central Market",
            "landmark_type": "market",
            "source": "local_landmark_catalog",
        }
    ]


def test_local_landmark_candidates_for_blank_mention_returns_empty() -> None:
    assert local_landmark_candidates_for_mention(mention=" ") == []
