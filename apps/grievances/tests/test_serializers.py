"""Serializer tests for grievance submission."""
from __future__ import annotations

import pytest

from apps.grievances.serializers import GrievanceSubmitSerializer

pytestmark = pytest.mark.django_db


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
