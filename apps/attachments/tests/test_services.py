"""Service tests for attachment registration."""
from __future__ import annotations

import pytest

from apps.attachments.services import register_attachment, update_attachment_metadata
from apps.grievances.services import submit_grievance

pytestmark = pytest.mark.django_db

SHA256_EXAMPLE = "b" * 64


def registration_values(*, grievance) -> dict[str, object]:
    return {
        "grievance": grievance,
        "storage_reference": "grievances/waste/photo.webp",
        "original_filename": "photo.webp",
        "content_type": "image/webp",
        "file_size_bytes": 4096,
        "content_hash": SHA256_EXAMPLE,
    }


def test_registration_generates_attachment_codes(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Waste near market.")

    first = register_attachment(uploader=submitter, **registration_values(grievance=grievance))
    second = register_attachment(uploader=submitter, **registration_values(grievance=grievance))

    assert first.attachment_code.startswith("ATT-")
    assert first.attachment_code.endswith("000001")
    assert second.attachment_code.endswith("000002")
    assert grievance.attachments.count() == 2


def test_metadata_update_keeps_validation_hooks_object_shaped(django_user_model) -> None:
    submitter = django_user_model.objects.create_user(username="citizen", password="password")
    grievance = submit_grievance(submitter=submitter, raw_text="Drain overflow.")
    attachment = register_attachment(uploader=submitter, **registration_values(grievance=grievance))

    update_attachment_metadata(
        attachment=attachment,
        values={"image_validation_metadata": {"result": "pending"}},
    )

    assert attachment.image_validation_metadata == {"result": "pending"}
