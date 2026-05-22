"""Read-side queries for departments."""
from __future__ import annotations

from django.db.models import QuerySet

from .models import Department


def department_list(*, active_only: bool = True) -> QuerySet[Department]:
    """Return departments using the model's stable display ordering."""
    departments = Department.objects.all()
    if active_only:
        return departments.filter(is_active=True)
    return departments


def department_get_by_code(*, code: str) -> Department:
    """Return a department by its stable external code."""
    return Department.objects.get(code=code)


def department_list_for_category(
    *, category_code: str, active_only: bool = True
) -> QuerySet[Department]:
    """Return departments that declare a grievance category code."""
    return department_list(active_only=active_only).filter(
        handled_categories__contains=[category_code]
    )
