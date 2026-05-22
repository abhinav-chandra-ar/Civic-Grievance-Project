"""DRF GIS serializers for wards."""
from __future__ import annotations

from typing import Any

from rest_framework_gis.serializers import GeoFeatureModelSerializer

from .models import Ward
from .services import create_ward, update_ward


class WardSerializer(GeoFeatureModelSerializer[Ward]):
    """Read a ward as a GeoJSON feature with its polygon boundary."""

    class Meta:
        model = Ward
        geo_field = "boundary"
        fields = (
            "id",
            "code",
            "name",
            "translated_names",
            "is_active",
            "officer_assignment_metadata",
            "landmark_mapping_metadata",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields


class WardWriteSerializer(GeoFeatureModelSerializer[Ward]):
    """Write ward geometry and metadata through services."""

    class Meta:
        model = Ward
        geo_field = "boundary"
        fields = (
            "code",
            "name",
            "translated_names",
            "is_active",
            "officer_assignment_metadata",
            "landmark_mapping_metadata",
        )

    def create(self, validated_data: dict[str, Any]) -> Ward:
        return create_ward(**validated_data)

    def update(self, instance: Ward, validated_data: dict[str, Any]) -> Ward:
        return update_ward(ward=instance, values=validated_data)
