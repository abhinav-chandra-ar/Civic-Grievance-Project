"""DRF GIS serializers for landmarks."""
from __future__ import annotations

from typing import Any

from rest_framework_gis.serializers import GeoFeatureModelSerializer

from .models import Landmark
from .services import create_landmark, update_landmark


class LandmarkSerializer(GeoFeatureModelSerializer[Landmark]):
    """Read a landmark as a GeoJSON feature."""

    class Meta:
        model = Landmark
        geo_field = "location"
        fields = (
            "id",
            "code",
            "primary_name",
            "aliases_en",
            "aliases_ml",
            "landmark_type",
            "normalized_search_tokens",
            "fuzzy_lookup_metadata",
            "ward_mapping_metadata",
            "is_active",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields


class LandmarkWriteSerializer(GeoFeatureModelSerializer[Landmark]):
    """Write landmark point and lookup metadata through services."""

    class Meta:
        model = Landmark
        geo_field = "location"
        fields = (
            "code",
            "primary_name",
            "aliases_en",
            "aliases_ml",
            "landmark_type",
            "normalized_search_tokens",
            "fuzzy_lookup_metadata",
            "ward_mapping_metadata",
            "is_active",
        )

    def create(self, validated_data: dict[str, Any]) -> Landmark:
        return create_landmark(**validated_data)

    def update(self, instance: Landmark, validated_data: dict[str, Any]) -> Landmark:
        return update_landmark(landmark=instance, values=validated_data)
