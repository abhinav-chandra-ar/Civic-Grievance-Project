"""Read-side queries for grievances."""
from __future__ import annotations

from typing import Any

from django.db.models import QuerySet

from .models import Grievance

# All non-citizen roles — used by permission checks and other parts of the app.
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

# Roles that see all grievances regardless of ward/department assignment.
# field_verifier and system_operator are platform-level roles that need
# full visibility for verification and operations tasks.
_WIDE_VISIBILITY_ROLES = frozenset(
    {
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
    """Return the grievances visible to *user* based on their role assignment.

    Visibility rules
    ----------------
    citizen (or no role)
        Sees only their own submitted grievances.

    ward_officer
        Sees only grievances whose ``ward`` FK matches ``user.assigned_ward``.
        If no ward is assigned, returns an empty queryset — never the full set.

    department_officer
        Sees only grievances whose ``department`` FK matches
        ``user.assigned_department``.  If no department is assigned, returns
        an empty queryset.

    municipal_admin / super_admin / field_verifier / system_operator
        Full visibility — no ward or department filter applied.

    No officer identity is ever exposed through the queryset itself; the
    routing metadata stored in ``status_metadata`` omits assignee details.

    Uses ``grievance_list_summary()`` — the three core JOINs are sufficient
    for all serialized list and retrieve responses.
    """
    grievances = grievance_list_summary()
    role = getattr(user, "role", None)

    # ── Citizens and unauthenticated actors: own grievances only ────────────
    if role not in OPERATIONAL_ROLES:
        return grievances.filter(submitter=user)

    # ── Wide-visibility admin/operator roles: full queryset ──────────────────
    if role in _WIDE_VISIBILITY_ROLES:
        return grievances

    # ── Ward officer: scoped to assigned ward only ───────────────────────────
    if role == "ward_officer":
        ward_id = getattr(user, "assigned_ward_id", None)
        if ward_id is not None:
            return grievances.filter(ward_id=ward_id)
        # Unassigned ward officer: return nothing rather than everything.
        return grievances.none()

    # ── Department officer: scoped to assigned department only ───────────────
    if role == "department_officer":
        dept_id = getattr(user, "assigned_department_id", None)
        if dept_id is not None:
            return grievances.filter(department_id=dept_id)
        # Unassigned department officer: return nothing rather than everything.
        return grievances.none()

    # Fallback for any future role additions: safe default is no access.
    return grievances.none()


def grievance_get_by_tracking_code(*, tracking_code: str) -> Grievance:
    """Return a grievance by citizen-facing tracking code with all relations."""
    return grievance_list_detail().get(tracking_code=tracking_code)


# ---------------------------------------------------------------------------
# Role-specific action queues
# ---------------------------------------------------------------------------
# These selectors return the subset of grievances that require a specific
# actor's attention right now.  They narrow the full visibility queryset
# by both role scope (ward/department FK) and status, so each actor's inbox
# contains only actionable items.
#
# Visibility rule (full list read-access) is unchanged — use
# grievance_list_visible_to_user() for the general list view.
# ---------------------------------------------------------------------------


def grievance_queue_for_ward_officer(*, user: Any) -> QuerySet[Grievance]:
    """TRIAGED grievances in the ward officer's assigned ward.

    TRIAGED = enriched complaints that the AI could not route with confidence,
    or complaints with no department resolved.  The ward officer reviews,
    confirms routing, and forwards to the correct department (→ ASSIGNED)
    or rejects invalid complaints (→ REJECTED).
    """
    from .models import GrievanceStatus  # noqa: PLC0415

    ward_id = getattr(user, "assigned_ward_id", None)
    if ward_id is None:
        return Grievance.objects.none()
    return grievance_list_summary().filter(
        ward_id=ward_id,
        status=GrievanceStatus.TRIAGED,
    )


def grievance_queue_for_dept_officer(*, user: Any) -> QuerySet[Grievance]:
    """ASSIGNED and IN_PROGRESS grievances for the department officer's department.

    ASSIGNED   = newly routed to this department by AI or ward officer.
    IN_PROGRESS = officer has accepted the work and started resolution.
    """
    from .models import GrievanceStatus  # noqa: PLC0415

    dept_id = getattr(user, "assigned_department_id", None)
    if dept_id is None:
        return Grievance.objects.none()
    return grievance_list_summary().filter(
        department_id=dept_id,
        status__in=[GrievanceStatus.ASSIGNED, GrievanceStatus.IN_PROGRESS],
    )


def grievance_escalation_alert_queue() -> QuerySet[Grievance]:
    """Grievances that carry an AI escalation flag — municipal admin alert queue.

    Escalation is NOT a lifecycle status; it is an urgency marker stored in
    ``status_metadata.escalation.should_escalate``.  These complaints may be
    in any active lifecycle state (typically ASSIGNED for routed emergencies,
    or TRIAGED when the department could not be resolved).

    Ordered oldest-first so the longest-waiting emergencies surface at the top.
    """
    return grievance_list_summary().filter(
        status_metadata__escalation__should_escalate=True,
    ).order_by("submitted_at")


def grievance_duplicate_flagged_queue() -> QuerySet[Grievance]:
    """DUPLICATE_FLAGGED grievances awaiting deduplication review by admin."""
    from .models import GrievanceStatus  # noqa: PLC0415

    return grievance_list_summary().filter(status=GrievanceStatus.DUPLICATE_FLAGGED)
