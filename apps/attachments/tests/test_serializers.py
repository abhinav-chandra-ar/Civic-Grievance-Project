"""Serializer tests for attachment registration."""
from __future__ import annotations

import pytest

from apps.attachments.serializers import AttachmentRegisterSerializer
from apps.grievances.services import submit_grievance

pytestmark = pytest.mark.django_db


def test_register_serializer_creates_attachment_record(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Water leak.")
    serializer = AttachmentRegisterSerializer(
        data={
            "grievance": grievance.pk,
            "storage_reference": "grievances/water/photo.png",
            "original_filename": "photo.png",
            "content_type": "image/png",
            "file_size_bytes": 1024,
            "content_hash": "c" * 64,
        }
    )

    assert serializer.is_valid(), serializer.errors
    attachment = serializer.save(uploader=submitter)
    assert attachment.grievance == grievance
    assert attachment.attachment_code.startswith("ATT-")


def test_register_serializer_rejects_non_image_content_type() -> None:
    serializer = AttachmentRegisterSerializer(data={"content_type": "text/plain"})

    assert not serializer.is_valid()
    assert "content_type" in serializer.errors
