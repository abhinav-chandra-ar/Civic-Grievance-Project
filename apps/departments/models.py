"""Department model and metadata validation."""
from __future__ import annotations

import re

from django.contrib.postgres.indexes import GinIndex
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator
from django.db import models

CATEGORY_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
# Department codes use standard abbreviations for Kerala civic agencies (e.g. KSEB, KWA, PWD).
# Allow mixed-case alphanumeric codes with optional underscores.
DEPARTMENT_CODE_VALIDATOR = RegexValidator(
    regex=r"^[A-Za-z][A-Za-z0-9_]*$",
    message="Use an alphanumeric department code (e.g. KSEB, KWA, CENGG). Letters, numbers, underscores only.",
)


def validate_category_codes(value: object) -> None:
    """Require a deduplicated list of grievance category codes."""
    if not isinstance(value, list):
        raise ValidationError("Handled categories must be a list of category codes.")
    if any(not isinstance(code, str) or not CATEGORY_CODE_PATTERN.fullmatch(code) for code in value):
        raise ValidationError(
            "Each handled category must be a lowercase code containing letters, numbers, and underscores."
        )
    if len(value) != len(set(value)):
        raise ValidationError("Handled category codes must not contain duplicates.")


def validate_translated_names(value: object) -> None:
    """Keep translated department names as a JSON object keyed by language."""
    if not isinstance(value, dict):
        raise ValidationError("Translated names must be a JSON object.")
    if any(not isinstance(language, str) or not isinstance(name, str) for language, name in value.items()):
        raise ValidationError("Translated names must map language codes to name strings.")


def validate_metadata_mapping(value: object) -> None:
    """Keep metadata hooks object-shaped for future domain services."""
    if not isinstance(value, dict):
        raise ValidationError("Metadata hooks must be a JSON object.")


class Department(models.Model):
    """Municipal department that can own grievance category routing metadata."""

    code = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        validators=[DEPARTMENT_CODE_VALIDATOR],
    )
    name = models.CharField(max_length=255, db_index=True)
    translated_names = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_translated_names],
    )
    handled_categories = models.JSONField(
        blank=True,
        default=list,
        validators=[validate_category_codes],
        help_text="List of grievance category codes handled by this department.",
    )
    is_active = models.BooleanField(default=True)
    escalation_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    sla_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "departments_department"
        verbose_name = "Department"
        verbose_name_plural = "Departments"
        ordering = ("name", "code")
        indexes = [
            models.Index(fields=["is_active", "name"], name="departments_active_name_idx"),
            # Enables indexed JSONB @> (contains) lookups for department_list_for_category().
            GinIndex(fields=["handled_categories"], name="dept_categories_gin_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.code})"

    def handles_category(self, category_code: str) -> bool:
        """Return whether this department declares a category code."""
        return category_code in self.handled_categories


__all__ = [
    "Department",
    "validate_category_codes",
    "validate_metadata_mapping",
    "validate_translated_names",
]
