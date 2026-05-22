"""Attachment model for image-first grievance uploads."""
from __future__ import annotations

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator, RegexValidator
from django.db import models

ATTACHMENT_CODE_VALIDATOR = RegexValidator(
    regex=r"^ATT-\d{4}-\d{6}$",
    message="Attachment codes must use the ATT-YYYY-NNNNNN format.",
)
SHA256_HEX_VALIDATOR = RegexValidator(
    regex=r"^[a-fA-F0-9]{64}$",
    message="Content hash must be a SHA-256 hexadecimal digest.",
)
COMMON_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/avif",
        "image/gif",
        "image/heic",
        "image/heif",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
    }
)


def validate_non_empty_text(value: str) -> None:
    """Reject blank storage and filename values."""
    if not value.strip():
        raise ValidationError("This value must not be empty.")


def validate_image_content_type(value: str) -> None:
    """Require image MIME types while allowing image formats beyond common ones."""
    normalized = value.strip().lower()
    if not normalized.startswith("image/") or len(normalized) <= len("image/"):
        raise ValidationError("Attachment content type must be an image MIME type.")


def validate_metadata_mapping(value: object) -> None:
    """Keep image service hooks object-shaped until services exist."""
    if not isinstance(value, dict):
        raise ValidationError("Metadata hooks must be a JSON object.")


class Attachment(models.Model):
    """Image-first stored attachment for a grievance."""

    attachment_code = models.CharField(
        max_length=15,
        unique=True,
        db_index=True,
        blank=True,
        validators=[ATTACHMENT_CODE_VALIDATOR],
    )
    grievance = models.ForeignKey(
        "grievances.Grievance",
        on_delete=models.CASCADE,
        related_name="attachments",
    )
    uploader = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="uploaded_attachments",
    )
    storage_reference = models.CharField(
        max_length=512,
        db_index=True,
        validators=[validate_non_empty_text],
    )
    original_filename = models.CharField(max_length=255, validators=[validate_non_empty_text])
    content_type = models.CharField(
        max_length=127,
        db_index=True,
        validators=[validate_image_content_type],
        help_text="Image MIME type, such as image/jpeg or image/png.",
    )
    file_size_bytes = models.PositiveBigIntegerField(validators=[MinValueValidator(1)])
    content_hash = models.CharField(
        max_length=64,
        db_index=True,
        validators=[SHA256_HEX_VALIDATOR],
        help_text="SHA-256 hexadecimal content fingerprint.",
    )
    attachment_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    image_validation_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    image_issue_classification_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    image_text_consistency_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    moderation_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    is_active = models.BooleanField(default=True)
    uploaded_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "attachments_attachment"
        verbose_name = "Attachment"
        verbose_name_plural = "Attachments"
        ordering = ("-uploaded_at", "-id")
        indexes = [
            models.Index(fields=["grievance", "is_active"], name="attach_grievance_active_idx"),
            models.Index(fields=["uploader", "uploaded_at"], name="attach_uploader_uploaded_idx"),
        ]

    def __str__(self) -> str:
        return self.attachment_code or f"Attachment {self.pk or 'unsaved'}"

    @property
    def is_common_image_type(self) -> bool:
        """Return whether the MIME type is one of the common image formats."""
        return self.content_type.lower() in COMMON_IMAGE_CONTENT_TYPES


__all__ = [
    "Attachment",
    "COMMON_IMAGE_CONTENT_TYPES",
    "validate_image_content_type",
    "validate_metadata_mapping",
]
