"""Ward model and GeoDjango boundary validation."""
from __future__ import annotations

from django.contrib.gis.db import models
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator

WARD_CODE_VALIDATOR = RegexValidator(
    regex=r"^[a-z][a-z0-9_]*$",
    message="Use a lowercase ward code containing letters, numbers, and underscores.",
)


def validate_translated_names(value: object) -> None:
    """Keep translated ward names keyed by language code."""
    if not isinstance(value, dict):
        raise ValidationError("Translated names must be a JSON object.")
    if any(not isinstance(language, str) or not isinstance(name, str) for language, name in value.items()):
        raise ValidationError("Translated names must map language codes to name strings.")


def validate_metadata_mapping(value: object) -> None:
    """Keep relation hooks object-shaped until domain relations exist."""
    if not isinstance(value, dict):
        raise ValidationError("Metadata hooks must be a JSON object.")


class Ward(models.Model):
    """Municipal ward with a validated PostGIS polygon boundary."""

    code = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        validators=[WARD_CODE_VALIDATOR],
    )
    name = models.CharField(max_length=255, db_index=True)
    translated_names = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_translated_names],
    )
    boundary = models.PolygonField(srid=4326, spatial_index=True)
    is_active = models.BooleanField(default=True)
    officer_assignment_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    landmark_mapping_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "wards_ward"
        verbose_name = "Ward"
        verbose_name_plural = "Wards"
        ordering = ("name", "code")
        indexes = [
            models.Index(fields=["is_active", "name"], name="wards_active_name_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.code})"

    def clean(self) -> None:
        """Reject empty or invalid ward polygons before persistence."""
        super().clean()
        if self.boundary is None or self.boundary.empty:
            raise ValidationError({"boundary": "Ward boundary must be a non-empty polygon."})
        if not self.boundary.valid:
            raise ValidationError({"boundary": "Ward boundary must be a valid polygon."})


__all__ = ["Ward", "validate_metadata_mapping", "validate_translated_names"]
