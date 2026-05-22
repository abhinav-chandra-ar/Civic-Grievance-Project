"""Orchestration tests for integrations."""
from __future__ import annotations

import pytest
from django.contrib.gis.geos import Point

from apps.integrations.services import analyze_attachment_image, analyze_grievance_submission
from apps.landmarks.models import Landmark, LandmarkType

pytestmark = pytest.mark.django_db


def test_grievance_analysis_returns_enrichment_payload_without_writes() -> None:
    payload = analyze_grievance_submission(
        raw_text="Water pipe leak near school",
        landmark_mention="",
        citizen_location_text="near school",
    )

    assert payload["category_code"] == "water_supply"
    assert payload["priority"] == "medium"
    assert "landmark_resolution_metadata" in payload
    assert "duplicate_detection_metadata" in payload
    assert isinstance(payload["provider_metadata"], dict)


def test_grievance_analysis_can_enrich_with_local_landmark_candidates() -> None:
    Landmark.objects.create(
        code="central_market",
        primary_name="Central Market",
        location=Point(76.27, 9.98, srid=4326),
        landmark_type=LandmarkType.MARKET,
    )

    payload = analyze_grievance_submission(
        raw_text="Garbage dumped",
        landmark_mention="Central Market",
    )

    candidates = payload["landmark_resolution_metadata"]["local_candidates"]
    assert candidates[0]["code"] == "central_market"


def test_attachment_analysis_returns_metadata_payload() -> None:
    payload = analyze_attachment_image(
        storage_reference="attachments/photo.png",
        content_type="image/png",
        content_hash="b" * 64,
    )

    assert payload["image_validation_metadata"]["is_valid"] is True
    assert payload["moderation_metadata"]["status"] == "pending"
    assert isinstance(payload["image_issue_classification_metadata"], dict)
