"""Client hook tests for integrations."""
from __future__ import annotations

from apps.integrations.clients.duplicates import detect_possible_duplicates
from apps.integrations.clients.images import validate_grievance_image
from apps.integrations.clients.landmarks import resolve_landmark_mention
from apps.integrations.clients.nlp import classify_grievance_text


def test_nlp_client_returns_classification_shape() -> None:
    result = classify_grievance_text(raw_text="Road has a pothole", language_hint="en")

    assert result["category_code"] == "road_damage"
    assert result["priority"] == "high"
    assert isinstance(result["metadata"], dict)


def test_landmark_client_returns_provider_style_hints_only() -> None:
    result = resolve_landmark_mention(mention="Central Market")

    assert result["landmark_code"] is None
    assert result["candidates"] == []
    assert isinstance(result["metadata"], dict)


def test_image_client_returns_attachment_ready_shape() -> None:
    result = validate_grievance_image(
        storage_reference="attachments/photo.jpg",
        content_type="image/jpeg",
        content_hash="a" * 64,
    )

    assert result["is_valid"] is True
    assert result["moderation_status"] == "pending"
    assert isinstance(result["metadata"], dict)


def test_duplicate_client_returns_candidate_shape() -> None:
    result = detect_possible_duplicates(raw_text="Waste near junction", category_code="waste")

    assert result["possible_duplicate_tracking_code"] is None
    assert result["candidates"] == []
    assert isinstance(result["metadata"], dict)
