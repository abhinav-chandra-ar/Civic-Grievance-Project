"""DRF serializers for attachment records."""
from __future__ import annotations

from typing import Any

from rest_framework import serializers

from .models import Attachment
from .services import register_attachment, update_attachment_metadata


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
