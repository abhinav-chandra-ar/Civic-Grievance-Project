"""Write-side services for attachments."""
from __future__ import annotations

import hashlib
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
def register_attachment_with_file(
    *,
    uploader,
    grievance,
    image_file_obj: Any,
    attachment_metadata: dict | None = None,
) -> Attachment:
    """Register an attachment by writing binary image bytes to local disk.

    This is the **direct-upload** path.  The caller passes a Django
    ``InMemoryUploadedFile`` (or any file-like object that has ``.read()``,
    ``.seek()``, ``.size``, ``.name``, and ``.content_type``).  The service:

    1. Computes SHA-256 from the raw bytes.
    2. Saves the file to ``MEDIA_ROOT`` via Django's ``FileField``.
    3. Sets ``storage_reference`` to the resulting relative path so the
       attachment is addressable via ``MEDIA_URL``.
    4. Fires ``attachment_registered`` — the signal handler in
       ``receivers.py`` will read the file bytes and run CLIP analysis.

    Parameters
    ----------
    uploader
        The :class:`~apps.users.models.User` performing the upload.
    grievance
        The :class:`~apps.grievances.models.Grievance` the image belongs to.
    image_file_obj
        A Django ``UploadedFile`` (``InMemoryUploadedFile`` or
        ``TemporaryUploadedFile``) received from a multipart request.
    attachment_metadata
        Optional caller-supplied metadata dict (default ``{}``).

    Returns
    -------
    :class:`~apps.attachments.models.Attachment`
        Saved instance with ``image_file`` and ``storage_reference`` set.
    """
    # ── 1. Read bytes to compute SHA-256 ────────────────────────────────────
    image_file_obj.seek(0)
    raw_bytes = image_file_obj.read()
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    image_file_obj.seek(0)

    content_type: str = (
        getattr(image_file_obj, "content_type", None)
        or "image/jpeg"
    ).strip().lower()
    file_size_bytes: int = len(raw_bytes)
    original_filename: str = (
        getattr(image_file_obj, "name", None) or "upload"
    ).strip() or "upload"

    # ── 2. Build and save the Attachment instance ────────────────────────────
    uploaded_at = timezone.now()
    attachment_code = generate_attachment_code(uploaded_at=uploaded_at)

    attachment = Attachment(
        uploader=uploader,
        grievance=grievance,
        image_file=image_file_obj,         # FileField → Django writes to MEDIA_ROOT on .save()
        storage_reference=original_filename,  # temporary; overwritten after save below
        original_filename=original_filename,
        content_type=content_type,
        file_size_bytes=file_size_bytes,
        content_hash=sha256_hex,
        attachment_metadata=dict(attachment_metadata or {}),
        uploaded_at=uploaded_at,
        attachment_code=attachment_code,
    )
    # Validate all fields except storage_reference (which we'll update after save)
    attachment.full_clean(exclude=["storage_reference"])
    attachment.save()  # Django FileField writes bytes to MEDIA_ROOT here

    # ── 3. Back-fill storage_reference with the real on-disk path ───────────
    storage_path: str = attachment.image_file.name if attachment.image_file else original_filename
    attachment.storage_reference = storage_path
    attachment.save(update_fields=["storage_reference", "updated_at"])

    # ── 4. Signal (triggers CLIP analysis in receivers.py) ──────────────────
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
