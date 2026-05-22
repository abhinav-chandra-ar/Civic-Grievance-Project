"""Read-side queries for grievances."""
from __future__ import annotations

from typing import Any

from django.db.models import QuerySet

from .models import Grievance

OPERATIONAL_ROLES = frozenset(
    {
        "ward_officer",
        "department_officer",
        "municipal_admin",
        "super_admin",
        "field_verifier",
        "system_operator",
    }
)


def grievance_list() -> QuerySet[Grievance]:
    """Return grievances with mapping relations ready for display."""
    return Grievance.objects.select_related(
        "submitter",
        "department",
        "ward",
        "resolved_landmark",
        "possible_duplicate_of",
    )


def grievance_list_visible_to_user(*, user: Any) -> QuerySet[Grievance]:
    """Return submitter-owned grievances unless the role is operational."""
    grievances = grievance_list()
    if getattr(user, "role", None) in OPERATIONAL_ROLES:
        return grievances
    return grievances.filter(submitter=user)


def grievance_get_by_tracking_code(*, tracking_code: str) -> Grievance:
    """Return a grievance by citizen-facing tracking code."""
    return grievance_list().get(tracking_code=tracking_code)
