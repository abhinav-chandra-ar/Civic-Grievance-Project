"""Model tests for attachment metadata."""
from __future__ import annotations

import pytest
from django.core.exceptions import ValidationError
from django.utils import timezone

from apps.attachments.models import Attachment
from apps.grievances.services import submit_grievance

pytestmark = pytest.mark.django_db

SHA256_EXAMPLE = "a" * 64


def build_attachment(*, submitter, grievance, **overrides) -> Attachment:
    values = {
        "attachment_code": "ATT-2026-000001",
        "grievance": grievance,
        "uploader": submitter,
        "storage_reference": "grievances/road/photo.jpg",
        "original_filename": "photo.jpg",
        "content_type": "image/jpeg",
        "file_size_bytes": 2048,
        "content_hash": SHA256_EXAMPLE,
        "uploaded_at": timezone.now(),
    }
    values.update(overrides)
    return Attachment(**values)


def test_attachment_accepts_image_metadata(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Road is damaged.")
    attachment = build_attachment(submitter=submitter, grievance=grievance)

    attachment.full_clean()
    assert attachment.is_common_image_type


def test_attachment_rejects_non_image_content_type(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Road is damaged.")
    attachment = build_attachment(
        submitter=submitter,
        grievance=grievance,
        content_type="application/pdf",
    )

    with pytest.raises(ValidationError) as exc_info:
        attachment.full_clean()

    assert "content_type" in exc_info.value.message_dict


def test_attachment_rejects_invalid_sha256_hash(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Road is damaged.")
    attachment = build_attachment(submitter=submitter, grievance=grievance, content_hash="bad")

    with pytest.raises(ValidationError) as exc_info:
        attachment.full_clean()

    assert "content_hash" in exc_info.value.message_dict


def test_attachment_rejects_blank_storage_reference(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Road is damaged.")
    attachment = build_attachment(submitter=submitter, grievance=grievance, storage_reference=" ")

    with pytest.raises(ValidationError) as exc_info:
        attachment.full_clean()

    assert "storage_reference" in exc_info.value.message_dict
