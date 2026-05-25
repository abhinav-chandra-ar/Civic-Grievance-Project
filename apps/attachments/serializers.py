"""DRF serializers for attachment records."""
from __future__ import annotations

from typing import Any

from rest_framework import serializers

from apps.grievances.models import Grievance
from .models import Attachment
from .services import register_attachment, register_attachment_with_file, update_attachment_metadata


class AttachmentSerializer(serializers.ModelSerializer[Attachment]):
    """Read representation for stored attachment metadata."""

    class Meta:
        model = Attachment
        fields = (
            "id",
            "attachment_code",
            "grievance",
            "uploader",
            "storage_reference",
            "original_filename",
            "content_type",
            "file_size_bytes",
            "content_hash",
            "attachment_metadata",
            "image_validation_metadata",
            "image_issue_classification_metadata",
            "image_text_consistency_metadata",
            "moderation_metadata",
            "is_active",
            "uploaded_at",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields


class AttachmentRegisterSerializer(serializers.ModelSerializer[Attachment]):
    """Register storage-backed image metadata through the service layer."""

    class Meta:
        model = Attachment
        fields = (
            "grievance",
            "storage_reference",
            "original_filename",
            "content_type",
            "file_size_bytes",
            "content_hash",
            "attachment_metadata",
        )

    def create(self, validated_data: dict[str, Any]) -> Attachment:
        return register_attachment(**validated_data)


class AttachmentUploadSerializer(serializers.Serializer):
    """Accept a real multipart image upload and create a fully-analysed Attachment.

    The client sends ``multipart/form-data`` with:

    * ``grievance`` — integer PK of the parent grievance
    * ``image_file`` — the raw image binary (JPEG, PNG, WEBP, …)
    * ``attachment_metadata`` — optional JSON object (default ``{}``)

    The service layer computes SHA-256, MIME type, and file size from the
    uploaded bytes, saves the file to ``MEDIA_ROOT``, then fires the
    ``attachment_registered`` signal which triggers CLIP analysis.
    """

    grievance = serializers.PrimaryKeyRelatedField(queryset=Grievance.objects.all())
    image_file = serializers.ImageField(
        allow_empty_file=False,
        help_text="Raw image bytes — JPEG, PNG, WEBP, GIF, etc.",
    )
    attachment_metadata = serializers.JSONField(required=False, default=dict)

    def create(self, validated_data: dict[str, Any]) -> Attachment:
        return register_attachment_with_file(
            uploader=validated_data["uploader"],
            grievance=validated_data["grievance"],
            image_file_obj=validated_data["image_file"],
            attachment_metadata=validated_data.get("attachment_metadata", {}),
        )

    def save(self, **kwargs: Any) -> Attachment:  # type: ignore[override]
        validated_data = {**self.validated_data, **kwargs}
        return self.create(validated_data)


class AttachmentMetadataSerializer(serializers.ModelSerializer[Attachment]):
    """Operational post-upload validation metadata updates."""

    class Meta:
        model = Attachment
        fields = (
            "attachment_metadata",
            "image_validation_metadata",
            "image_issue_classification_metadata",
            "image_text_consistency_metadata",
            "moderation_metadata",
            "is_active",
        )

    def update(self, instance: Attachment, validated_data: dict[str, Any]) -> Attachment:
        return update_attachment_metadata(attachment=instance, values=validated_data)
