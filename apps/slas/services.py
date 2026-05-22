"""Write-side services for SLA state."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from .models import SLA, SLABreachType, SLAStatus
from .signals import sla_breached, sla_created, sla_updated

SLA_SEQUENCE_PATTERN = re.compile(r"^SLA-(?P<year>\d{4})-(?P<sequence>\d{6})$")


def format_sla_code(*, year: int, sequence: int) -> str:
    """Format an internal SLA code."""
    if sequence < 1 or sequence > 999999:
        raise ValidationError("SLA code sequence is out of range.")
    return f"SLA-{year:04d}-{sequence:06d}"


def generate_sla_code(*, created_at=None) -> str:
    """Generate the next year-scoped SLA code."""
    created_at = created_at or timezone.now()
    year = timezone.localtime(created_at).year
    latest_code = (
        SLA.objects.select_for_update()
        .filter(sla_code__startswith=f"SLA-{year:04d}-")
        .order_by("-sla_code")
        .values_list("sla_code", flat=True)
        .first()
    )
    if latest_code is None:
        return format_sla_code(year=year, sequence=1)

    match = SLA_SEQUENCE_PATTERN.fullmatch(latest_code)
    if match is None:
        raise ValidationError("Existing SLA code cannot be sequenced.")
    return format_sla_code(year=year, sequence=int(match["sequence"]) + 1)


@transaction.atomic
def create_sla_for_grievance(
    *,
    grievance,
    response_due_at,
    resolution_due_at,
    policy_snapshot_metadata: Mapping[str, Any] | None = None,
    escalation_metadata: Mapping[str, Any] | None = None,
    next_escalation_due_at=None,
) -> SLA:
    """Create the one current SLA row for a grievance."""
    sla = SLA(
        sla_code=generate_sla_code(),
        grievance=grievance,
        response_due_at=response_due_at,
        resolution_due_at=resolution_due_at,
        policy_snapshot_metadata=dict(policy_snapshot_metadata or {}),
        escalation_metadata=dict(escalation_metadata or {}),
        next_escalation_due_at=next_escalation_due_at,
    )
    sla.full_clean()
    sla.save()
    sla_created.send(sender=SLA, sla=sla)
    return sla


@transaction.atomic
def update_sla_deadlines(
    *,
    sla: SLA,
    response_due_at=None,
    resolution_due_at=None,
    policy_snapshot_metadata: Mapping[str, Any] | None = None,
    escalation_metadata: Mapping[str, Any] | None = None,
    next_escalation_due_at=None,
) -> SLA:
    """Update current SLA deadlines and metadata hooks."""
    update_fields = []
    if response_due_at is not None:
        sla.response_due_at = response_due_at
        update_fields.append("response_due_at")
    if resolution_due_at is not None:
        sla.resolution_due_at = resolution_due_at
        update_fields.append("resolution_due_at")
    if policy_snapshot_metadata is not None:
        sla.policy_snapshot_metadata = dict(policy_snapshot_metadata)
        update_fields.append("policy_snapshot_metadata")
    if escalation_metadata is not None:
        sla.escalation_metadata = dict(escalation_metadata)
        update_fields.append("escalation_metadata")
    if next_escalation_due_at is not None:
        sla.next_escalation_due_at = next_escalation_due_at
        update_fields.append("next_escalation_due_at")

    if update_fields:
        sla.full_clean()
        sla.save(update_fields=[*update_fields, "updated_at"])
        sla_updated.send(sender=SLA, sla=sla, updated_fields=frozenset(update_fields))
    return sla


@transaction.atomic
def update_sla_status(*, sla: SLA, sla_status: str | SLAStatus) -> SLA:
    """Update SLA operational status without inferring breach state."""
    sla.sla_status = str(sla_status)
    sla.full_clean()
    sla.save(update_fields=["sla_status", "updated_at"])
    sla_updated.send(sender=SLA, sla=sla, updated_fields=frozenset({"sla_status"}))
    return sla


@transaction.atomic
def mark_sla_breached(
    *,
    sla: SLA,
    breach_type: str | SLABreachType,
    breach_metadata: Mapping[str, Any] | None = None,
    breached_at=None,
) -> SLA:
    """Explicitly mark an SLA as breached."""
    sla.is_breached = True
    sla.sla_status = SLAStatus.BREACHED
    sla.breach_type = str(breach_type)
    sla.breached_at = breached_at or timezone.now()
    sla.breach_metadata = dict(breach_metadata or {})
    sla.full_clean()
    sla.save(
        update_fields=[
            "is_breached",
            "sla_status",
            "breach_type",
            "breached_at",
            "breach_metadata",
            "updated_at",
        ]
    )
    sla_breached.send(sender=SLA, sla=sla)
    return sla


def compute_breach_type(*, sla: SLA, now=None) -> SLABreachType:
    """Compute response/resolution breach type without mutating state."""
    now = now or timezone.now()
    response_breached = sla.response_due_at < now
    resolution_breached = sla.resolution_due_at < now
    if response_breached and resolution_breached:
        return SLABreachType.BOTH
    if response_breached:
        return SLABreachType.RESPONSE
    if resolution_breached:
        return SLABreachType.RESOLUTION
    return SLABreachType.NONE


def refresh_sla_deadline_status(*, sla: SLA, now=None) -> SLA:
    """Mark a breach when deadlines have passed; otherwise leave state intact."""
    breach_type = compute_breach_type(sla=sla, now=now)
    if breach_type == SLABreachType.NONE:
        return sla
    return mark_sla_breached(
        sla=sla,
        breach_type=breach_type,
        breach_metadata={"source": "deadline_refresh"},
        breached_at=now,
    )
