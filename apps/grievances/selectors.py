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


def grievance_list_summary() -> QuerySet[Grievance]:
    """Return grievances with the three relations needed for list and queue views.

    Fetches submitter, department, and ward in a single query.  The two
    optional relations — resolved_landmark and possible_duplicate_of — are
    intentionally omitted: ``GrievanceSerializer`` renders both as integer PKs
    (reads the ``_id`` column directly), so no JOIN is needed.  Use
    ``grievance_list_detail()`` when the related *objects* themselves must be
    accessed (e.g. landmark name for a detail panel, duplicate tracking code).
    """
    return Grievance.objects.select_related(
        "submitter",
        "department",
        "ward",
    )


def grievance_list_detail() -> QuerySet[Grievance]:
    """Return grievances with all five mapping relations pre-fetched.

    Includes the self-referential possible_duplicate_of JOIN and
    resolved_landmark.  Use for single-object detail fetches or any code
    path that needs to traverse those relations without extra queries.
    """
    return Grievance.objects.select_related(
        "submitter",
        "department",
        "ward",
        "resolved_landmark",
        "possible_duplicate_of",
    )


def grievance_list() -> QuerySet[Grievance]:
    """Backward-compatible alias for ``grievance_list_detail()``.

    Prefer ``grievance_list_summary()`` for list/queue views and
    ``grievance_list_detail()`` for detail views.
    """
    return grievance_list_detail()


def grievance_list_visible_to_user(*, user: Any) -> QuerySet[Grievance]:
    """Return submitter-owned grievances unless the role is operational.

    Uses ``grievance_list_summary()`` — the three core JOINs are sufficient
    for all serialized list and retrieve responses.
    """
    grievances = grievance_list_summary()
    if getattr(user, "role", None) in OPERATIONAL_ROLES:
        return grievances
    return grievances.filter(submitter=user)


def grievance_get_by_tracking_code(*, tracking_code: str) -> Grievance:
    """Return a grievance by citizen-facing tracking code with all relations."""
    return grievance_list_detail().get(tracking_code=tracking_code)
