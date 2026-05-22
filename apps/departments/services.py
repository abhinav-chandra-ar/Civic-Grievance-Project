"""Write-side services for departments."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from django.core.exceptions import ValidationError
from django.db import transaction

from .models import Department
from .signals import department_created, department_updated

DEPARTMENT_WRITE_FIELDS = frozenset(
    {
        "code",
        "name",
        "translated_names",
        "handled_categories",
        "is_active",
        "escalation_metadata",
        "sla_metadata",
    }
)


def _prepare_department_values(values: Mapping[str, Any]) -> dict[str, Any]:
    unknown_fields = set(values) - DEPARTMENT_WRITE_FIELDS
    if unknown_fields:
        raise ValidationError(f"Unsupported department fields: {', '.join(sorted(unknown_fields))}")
    return {field: values[field] for field in DEPARTMENT_WRITE_FIELDS if field in values}


@transaction.atomic
def create_department(**values: Any) -> Department:
    """Create a validated department without grievance-model dependencies."""
    prepared = _prepare_department_values(values)
    department = Department(**prepared)
    department.full_clean()
    department.save()
    department_created.send(sender=Department, department=department)
    return department


@transaction.atomic
def update_department(*, department: Department, values: Mapping[str, Any]) -> Department:
    """Update department routing and metadata values."""
    prepared = _prepare_department_values(values)
    for field, value in prepared.items():
        setattr(department, field, value)

    if prepared:
        department.full_clean()
        department.save(update_fields=[*prepared, "updated_at"])
        department_updated.send(
            sender=Department,
            department=department,
            updated_fields=frozenset(prepared),
        )
    return department


def set_department_active(*, department: Department, is_active: bool) -> Department:
    """Toggle department availability for trusted call sites."""
    return update_department(department=department, values={"is_active": is_active})
