"""Write-side services for grievances."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from .models import Grievance, GrievanceStatus
from .signals import grievance_status_changed, grievance_submitted, grievance_updated

TRACKING_SEQUENCE_PATTERN = re.compile(r"^GRV-(?P<year>\d{4})-(?P<sequence>\d{6})$")
GRIEVANCE_CREATE_FIELDS = frozenset(
    {
        "raw_text",
        "landmark_mention",
        "citizen_location_text",
        "image_attachment_metadata",
    }
)
GRIEVANCE_ENRICHMENT_FIELDS = frozenset(
    {
        "normalized_summary",
        "category_code",
        "department",
        "ward",
        "resolved_landmark",
        "landmark_resolution_metadata",
        "priority",
        "possible_duplicate_of",
        "duplicate_detection_metadata",
        "image_validation_metadata",
        "status_reason",
        "status_metadata",
    }
)
GRIEVANCE_ADMIN_CREATE_FIELDS = GRIEVANCE_CREATE_FIELDS | GRIEVANCE_ENRICHMENT_FIELDS | frozenset(
    {
        "tracking_code",
        "submitted_at",
        "status",
    }
)
DEFAULT_SLA_DEADLINE_DELTAS = {
    "low": timezone.timedelta(days=7),
    "medium": timezone.timedelta(days=5),
    "high": timezone.timedelta(days=3),
    "urgent": timezone.timedelta(days=1),
    "critical": timezone.timedelta(hours=12),
}


def format_tracking_code(*, year: int, sequence: int) -> str:
    """Format a citizen-facing tracking code."""
    if sequence < 1 or sequence > 999999:
        raise ValidationError("Tracking code sequence is out of range.")
    return f"GRV-{year:04d}-{sequence:06d}"


def generate_tracking_code(*, submitted_at=None) -> str:
    """Generate the next year-scoped tracking code.

    The unique database constraint remains the collision guard for concurrent
    writers until a dedicated database sequence is introduced by migrations.
    """
    submitted_at = submitted_at or timezone.now()
    year = timezone.localtime(submitted_at).year
    latest_code = (
        Grievance.objects.select_for_update()
        .filter(tracking_code__startswith=f"GRV-{year:04d}-")
        .order_by("-tracking_code")
        .values_list("tracking_code", flat=True)
        .first()
    )
    if latest_code is None:
        return format_tracking_code(year=year, sequence=1)

    match = TRACKING_SEQUENCE_PATTERN.fullmatch(latest_code)
    if match is None:
        raise ValidationError("Existing tracking code cannot be sequenced.")
    return format_tracking_code(year=year, sequence=int(match["sequence"]) + 1)


def _prepare_values(values: Mapping[str, Any], allowed_fields: frozenset[str]) -> dict[str, Any]:
    unknown_fields = set(values) - allowed_fields
    if unknown_fields:
        raise ValidationError(f"Unsupported grievance fields: {', '.join(sorted(unknown_fields))}")
    return {field: values[field] for field in allowed_fields if field in values}


@transaction.atomic
def submit_grievance(*, submitter, **values: Any) -> Grievance:
    """Submit a citizen grievance and run creation orchestration."""
    return create_grievance_with_foundation_records(submitter=submitter, actor=submitter, **values)


def _default_sla_deadlines(*, grievance: Grievance) -> tuple[Any, Any]:
    deadline_delta = DEFAULT_SLA_DEADLINE_DELTAS.get(
        grievance.priority,
        DEFAULT_SLA_DEADLINE_DELTAS["medium"],
    )
    due_at = grievance.submitted_at + deadline_delta
    return due_at, due_at


@transaction.atomic
def create_grievance_with_foundation_records(
    *,
    submitter,
    actor=None,
    response_due_at=None,
    resolution_due_at=None,
    **values: Any,
) -> Grievance:
    """Create a grievance and its initial workflow, SLA, and audit records."""
    submitted_at = timezone.now()
    prepared = _prepare_values(values, GRIEVANCE_ADMIN_CREATE_FIELDS)
    submitted_at = prepared.pop("submitted_at", submitted_at) or submitted_at
    tracking_code = prepared.pop("tracking_code", "") or generate_tracking_code(
        submitted_at=submitted_at
    )
    grievance = Grievance(
        submitter=submitter,
        submitted_at=submitted_at,
        tracking_code=tracking_code,
        **prepared,
    )
    grievance.full_clean()
    grievance.save()
    grievance_submitted.send(sender=Grievance, grievance=grievance)

    from apps.audit.models import AuditActionType
    from apps.audit.services import record_audit_event
    from apps.slas.services import create_sla_for_grievance
    from apps.workflows.models import WorkflowTransitionType
    from apps.workflows.services import transition_grievance

    actor = actor or submitter
    transition_grievance(
        grievance=grievance,
        actor=actor,
        new_status=grievance.status,
        transition_type=WorkflowTransitionType.STATUS_CHANGE,
        transition_reason="Grievance created.",
        remarks="Initial grievance creation event.",
        status_metadata={"source": "grievance_creation"},
    )
    if response_due_at is None or resolution_due_at is None:
        default_response_due_at, default_resolution_due_at = _default_sla_deadlines(
            grievance=grievance
        )
        response_due_at = response_due_at or default_response_due_at
        resolution_due_at = resolution_due_at or default_resolution_due_at
    create_sla_for_grievance(
        grievance=grievance,
        response_due_at=response_due_at,
        resolution_due_at=resolution_due_at,
        policy_snapshot_metadata={
            "source": "temporary_priority_rule",
            "priority": grievance.priority,
        },
    )
    record_audit_event(
        actor=actor,
        target_model="grievances.Grievance",
        target_object_id=str(grievance.pk),
        action_type=AuditActionType.CREATE,
        change_metadata={
            "tracking_code": grievance.tracking_code,
            "status": grievance.status,
            "priority": grievance.priority,
        },
        remarks="Grievance created through orchestration service.",
    )
    return grievance


@transaction.atomic
def update_grievance_enrichment(*, grievance: Grievance, values: Mapping[str, Any]) -> Grievance:
    """Update nullable mapping and metadata fields from trusted enrichment paths."""
    prepared = _prepare_values(values, GRIEVANCE_ENRICHMENT_FIELDS)
    for field, value in prepared.items():
        setattr(grievance, field, value)

    if prepared:
        grievance.full_clean()
        grievance.save(update_fields=[*prepared, "updated_at"])
        grievance_updated.send(
            sender=Grievance,
            grievance=grievance,
            updated_fields=frozenset(prepared),
        )
    return grievance


@transaction.atomic
def change_grievance_status(
    *,
    grievance: Grievance,
    status: str | GrievanceStatus,
    reason: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> Grievance:
    """Record status changes locally until workflow ownership exists."""
    previous_status = grievance.status
    grievance.status = str(status)
    grievance.status_reason = reason
    grievance.status_metadata = dict(metadata or {})
    grievance.last_status_changed_at = timezone.now()
    grievance.full_clean()
    grievance.save(
        update_fields=[
            "status",
            "status_reason",
            "status_metadata",
            "last_status_changed_at",
            "updated_at",
        ]
    )
    grievance_status_changed.send(
        sender=Grievance,
        grievance=grievance,
        previous_status=previous_status,
    )
    return grievance
