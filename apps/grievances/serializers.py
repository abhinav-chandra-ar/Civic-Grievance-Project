"""DRF serializers for grievance submission and enrichment."""
from __future__ import annotations

from typing import Any

from rest_framework import serializers

from .models import Grievance
from .services import create_grievance_with_foundation_records, update_grievance_enrichment


class GrievanceSerializer(serializers.ModelSerializer[Grievance]):
    """Read representation for citizen and operational grievance views."""

    class Meta:
        model = Grievance
        fields = (
            "id",
            "tracking_code",
            "submitter",
            "raw_text",
            "landmark_mention",
            "citizen_location_text",
            "image_attachment_metadata",
            "normalized_summary",
            "category_code",
            "department",
            "ward",
            "resolved_landmark",
            "landmark_resolution_metadata",
            "priority",
            "possible_duplicate_of",
            "duplicate_detection_metadata",
            "image_validation_metadata",
            "status",
            "status_reason",
            "status_metadata",
            "submitted_at",
            "last_status_changed_at",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields


class GrievanceSubmitSerializer(serializers.ModelSerializer[Grievance]):
    """Citizen submission fields backed by tracking-code generation."""

    class Meta:
        model = Grievance
        fields = (
            "raw_text",
            "landmark_mention",
            "citizen_location_text",
            "image_attachment_metadata",
        )

    def create(self, validated_data: dict[str, Any]) -> Grievance:
        return create_grievance_with_foundation_records(**validated_data)


class GrievanceEnrichmentSerializer(serializers.ModelSerializer[Grievance]):
    """Operational mapping and metadata fields before workflow ownership exists."""

    class Meta:
        model = Grievance
        fields = (
            "normalized_summary",
            "category_code",
            "department",
            "ward",
            "resolved_landmark",
            "landmark_resolution_metadata",
            "priority",
            "possible_duplicate_of",
            "duplicate_detection_metadata",
            "image_validation_metadata",
            "status_reason",
            "status_metadata",
        )

    def update(self, instance: Grievance, validated_data: dict[str, Any]) -> Grievance:
        return update_grievance_enrichment(grievance=instance, values=validated_data)
