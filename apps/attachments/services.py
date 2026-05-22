"""Write-side services for attachments."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from .models import Attachment
from .signals import attachment_registered, attachment_updated

ATTACHMENT_SEQUENCE_PATTERN = re.compile(r"^ATT-(?P<year>\d{4})-(?P<sequence>\d{6})$")
ATTACHMENT_CREATE_FIELDS = frozenset(
    {
        "grievance",
        "storage_reference",
        "original_filename",
        "content_type",
        "file_size_bytes",
        "content_hash",
        "attachment_metadata",
    }
)
ATTACHMENT_METADATA_FIELDS = frozenset(
    {
        "attachment_metadata",
        "image_validation_metadata",
        "image_issue_classification_metadata",
        "image_text_consistency_metadata",
        "moderation_metadata",
        "is_active",
    }
)


def format_attachment_code(*, year: int, sequence: int) -> str:
    """Format an internal attachment identifier."""
    if sequence < 1 or sequence > 999999:
        raise ValidationError("Attachment code sequence is out of range.")
    return f"ATT-{year:04d}-{sequence:06d}"


def generate_attachment_code(*, uploaded_at=None) -> str:
    """Generate the next year-scoped attachment code."""
    uploaded_at = uploaded_at or timezone.now()
    year = timezone.localtime(uploaded_at).year
    latest_code = (
        Attachment.objects.select_for_update()
        .filter(attachment_code__startswith=f"ATT-{year:04d}-")
        .order_by("-attachment_code")
        .values_list("attachment_code", flat=True)
        .first()
    )
    if latest_code is None:
        return format_attachment_code(year=year, sequence=1)

    match = ATTACHMENT_SEQUENCE_PATTERN.fullmatch(latest_code)
    if match is None:
        raise ValidationError("Existing attachment code cannot be sequenced.")
    return format_attachment_code(year=year, sequence=int(match["sequence"]) + 1)


def _prepare_values(values: Mapping[str, Any], allowed_fields: frozenset[str]) -> dict[str, Any]:
    unknown_fields = set(values) - allowed_fields
    if unknown_fields:
        raise ValidationError(f"Unsupported attachment fields: {', '.join(sorted(unknown_fields))}")
    return {field: values[field] for field in allowed_fields if field in values}


@transaction.atomic
def register_attachment(*, uploader, **values: Any) -> Attachment:
    """Register an image attachment with storage managed outside the model."""
    uploaded_at = timezone.now()
    prepared = _prepare_values(values, ATTACHMENT_CREATE_FIELDS)
    attachment = Attachment(
        uploader=uploader,
        uploaded_at=uploaded_at,
        attachment_code=generate_attachment_code(uploaded_at=uploaded_at),
        **prepared,
    )
    attachment.full_clean()
    attachment.save()
    attachment_registered.send(sender=Attachment, attachment=attachment)
    return attachment


@transaction.atomic
def update_attachment_metadata(
    *, attachment: Attachment, values: Mapping[str, Any]
) -> Attachment:
    """Update validation and moderation hooks after upload."""
    prepared = _prepare_values(values, ATTACHMENT_METADATA_FIELDS)
    for field, value in prepared.items():
        setattr(attachment, field, value)

    if prepared:
        attachment.full_clean()
        attachment.save(update_fields=[*prepared, "updated_at"])
        attachment_updated.send(
            sender=Attachment,
            attachment=attachment,
            updated_fields=frozenset(prepared),
        )
    return attachment
