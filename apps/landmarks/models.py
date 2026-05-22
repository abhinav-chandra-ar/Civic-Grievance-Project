"""Landmark point model and search metadata validation."""
from __future__ import annotations

from django.contrib.gis.db import models
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator

LANDMARK_CODE_VALIDATOR = RegexValidator(
    regex=r"^[a-z][a-z0-9_]*$",
    message="Use a lowercase landmark code containing letters, numbers, and underscores.",
)


class LandmarkType(models.TextChoices):
    JUNCTION = "junction", "Junction"
    HOSPITAL = "hospital", "Hospital"
    SCHOOL = "school", "School"
    GOVERNMENT_OFFICE = "government_office", "Government office"
    MARKET = "market", "Market"
    TEMPLE = "temple", "Temple"
    MOSQUE = "mosque", "Mosque"
    CHURCH = "church", "Church"
    BUS_STOP = "bus_stop", "Bus stop"
    RAILWAY_STATION = "railway_station", "Railway station"
    MALL = "mall", "Mall"
    OTHER = "other", "Other"


def validate_non_empty_string_list(value: object) -> None:
    """Require predictable string lists for alias and search-token fields."""
    if not isinstance(value, list):
        raise ValidationError("Expected a list of strings.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValidationError("List values must be non-empty strings.")


def validate_metadata_mapping(value: object) -> None:
    """Keep fuzzy and ward hooks object-shaped until integrations exist."""
    if not isinstance(value, dict):
        raise ValidationError("Metadata hooks must be a JSON object.")


class Landmark(models.Model):
    """Searchable civic landmark represented by a PostGIS point."""

    code = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        validators=[LANDMARK_CODE_VALIDATOR],
    )
    primary_name = models.CharField(max_length=255, db_index=True)
    aliases_en = models.JSONField(
        blank=True,
        default=list,
        validators=[validate_non_empty_string_list],
    )
    aliases_ml = models.JSONField(
        blank=True,
        default=list,
        validators=[validate_non_empty_string_list],
    )
    location = models.PointField(srid=4326, spatial_index=True)
    landmark_type = models.CharField(
        max_length=32,
        choices=LandmarkType.choices,
        default=LandmarkType.OTHER,
        db_index=True,
    )
    normalized_search_tokens = models.JSONField(
        blank=True,
        default=list,
        validators=[validate_non_empty_string_list],
        help_text="Canonical tokens prepared for fuzzy and mixed-language lookup.",
    )
    fuzzy_lookup_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    ward_mapping_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "landmarks_landmark"
        verbose_name = "Landmark"
        verbose_name_plural = "Landmarks"
        ordering = ("primary_name", "code")
        indexes = [
            models.Index(fields=["is_active", "primary_name"], name="landmarks_active_name_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(landmark_type__in=LandmarkType.values),
                name="landmarks_type_valid",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.primary_name} ({self.code})"

    def clean(self) -> None:
        """Reject absent, empty, or invalid point geometry before persistence."""
        super().clean()
        if self.location is None or self.location.empty:
            raise ValidationError({"location": "Landmark location must be a non-empty point."})
        if not self.location.valid:
            raise ValidationError({"location": "Landmark location must be a valid point."})


__all__ = ["Landmark", "LandmarkType", "validate_metadata_mapping", "validate_non_empty_string_list"]
