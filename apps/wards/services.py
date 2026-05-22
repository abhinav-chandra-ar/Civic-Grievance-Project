"""Write-side services for wards."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from django.core.exceptions import ValidationError
from django.db import transaction

from .models import Ward
from .signals import ward_created, ward_updated

WARD_WRITE_FIELDS = frozenset(
    {
        "code",
        "name",
        "translated_names",
        "boundary",
        "is_active",
        "officer_assignment_metadata",
        "landmark_mapping_metadata",
    }
)


def _prepare_ward_values(values: Mapping[str, Any]) -> dict[str, Any]:
    unknown_fields = set(values) - WARD_WRITE_FIELDS
    if unknown_fields:
        raise ValidationError(f"Unsupported ward fields: {', '.join(sorted(unknown_fields))}")
    return {field: values[field] for field in WARD_WRITE_FIELDS if field in values}


@transaction.atomic
def create_ward(**values: Any) -> Ward:
    """Create a validated ward boundary and metadata record."""
    prepared = _prepare_ward_values(values)
    ward = Ward(**prepared)
    ward.full_clean()
    ward.save()
    ward_created.send(sender=Ward, ward=ward)
    return ward


@transaction.atomic
def update_ward(*, ward: Ward, values: Mapping[str, Any]) -> Ward:
    """Update ward fields without binding to future relation models."""
    prepared = _prepare_ward_values(values)
    for field, value in prepared.items():
        setattr(ward, field, value)

    if prepared:
        ward.full_clean()
        ward.save(update_fields=[*prepared, "updated_at"])
        ward_updated.send(sender=Ward, ward=ward, updated_fields=frozenset(prepared))
    return ward


def set_ward_active(*, ward: Ward, is_active: bool) -> Ward:
    """Toggle ward availability for trusted call sites."""
    return update_ward(ward=ward, values={"is_active": is_active})
