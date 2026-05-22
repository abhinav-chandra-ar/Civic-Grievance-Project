"""Read-side queries for SLA state."""
from __future__ import annotations

from typing import Any

from django.db.models import Q, QuerySet

from .models import SLA, SLAStatus

SLA_OPERATOR_ROLES = frozenset(
    {
        "ward_officer",
        "department_officer",
        "municipal_admin",
        "super_admin",
        "field_verifier",
        "system_operator",
    }
)


def sla_list() -> QuerySet[SLA]:
    """Return SLA rows with grievance context ready for display."""
    return SLA.objects.select_related("grievance", "grievance__submitter")


def sla_get_for_grievance(*, grievance) -> SLA:
    """Return the one current SLA row for a grievance."""
    return sla_list().get(grievance=grievance)


def sla_list_visible_to_user(*, user: Any) -> QuerySet[SLA]:
    """Return grievance-owned SLA rows unless the role is operational."""
    slas = sla_list()
    if getattr(user, "role", None) in SLA_OPERATOR_ROLES:
        return slas
    return slas.filter(grievance__submitter=user)


def sla_list_breached() -> QuerySet[SLA]:
    """Return explicitly breached SLA rows."""
    return sla_list().filter(is_breached=True)


def sla_list_with_upcoming_response_deadline(*, before) -> QuerySet[SLA]:
    """Return active rows whose response deadline is due before a threshold."""
    return sla_list().filter(
        is_breached=False,
        sla_status=SLAStatus.ACTIVE,
        response_due_at__lte=before,
    )


def sla_list_with_upcoming_resolution_deadline(*, before) -> QuerySet[SLA]:
    """Return active rows whose resolution deadline is due before a threshold."""
    return sla_list().filter(
        is_breached=False,
        sla_status=SLAStatus.ACTIVE,
        resolution_due_at__lte=before,
    )


def sla_list_requiring_breach_check(*, now) -> QuerySet[SLA]:
    """Return active rows whose deadlines have passed and need breach evaluation."""
    return sla_list().filter(
        Q(response_due_at__lt=now) | Q(resolution_due_at__lt=now),
        is_breached=False,
        sla_status=SLAStatus.ACTIVE,
    )


def sla_list_with_escalation_due(*, now) -> QuerySet[SLA]:
    """Return rows whose next escalation hook is due for future task processing."""
    return sla_list().filter(next_escalation_due_at__isnull=False, next_escalation_due_at__lte=now)
