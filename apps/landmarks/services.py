"""Write-side services for landmarks."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from django.core.exceptions import ValidationError
from django.db import transaction

from .models import Landmark
from .signals import landmark_created, landmark_updated

LANDMARK_WRITE_FIELDS = frozenset(
    {
        "code",
        "primary_name",
        "aliases_en",
        "aliases_ml",
        "location",
        "landmark_type",
        "normalized_search_tokens",
        "fuzzy_lookup_metadata",
        "ward_mapping_metadata",
        "is_active",
    }
)


def _prepare_landmark_values(values: Mapping[str, Any]) -> dict[str, Any]:
    unknown_fields = set(values) - LANDMARK_WRITE_FIELDS
    if unknown_fields:
        raise ValidationError(f"Unsupported landmark fields: {', '.join(sorted(unknown_fields))}")
    return {field: values[field] for field in LANDMARK_WRITE_FIELDS if field in values}


@transaction.atomic
def create_landmark(**values: Any) -> Landmark:
    """Create a validated landmark point and search metadata record."""
    prepared = _prepare_landmark_values(values)
    landmark = Landmark(**prepared)
    landmark.full_clean()
    landmark.save()
    landmark_created.send(sender=Landmark, landmark=landmark)
    return landmark


@transaction.atomic
def update_landmark(*, landmark: Landmark, values: Mapping[str, Any]) -> Landmark:
    """Update a landmark without ward or grievance dependencies."""
    prepared = _prepare_landmark_values(values)
    for field, value in prepared.items():
        setattr(landmark, field, value)

    if prepared:
        landmark.full_clean()
        landmark.save(update_fields=[*prepared, "updated_at"])
        landmark_updated.send(
            sender=Landmark,
            landmark=landmark,
            updated_fields=frozenset(prepared),
        )
    return landmark


def set_landmark_active(*, landmark: Landmark, is_active: bool) -> Landmark:
    """Toggle landmark availability for trusted call sites."""
    return update_landmark(landmark=landmark, values={"is_active": is_active})
