"""Write-side transition services for workflow history."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from apps.grievances.services import change_grievance_status

from .models import WorkflowEvent, WorkflowTransitionType
from .signals import workflow_event_recorded

WORKFLOW_SEQUENCE_PATTERN = re.compile(r"^WFE-(?P<year>\d{4})-(?P<sequence>\d{6})$")
ASSIGNMENT_TRANSITION_TYPES = frozenset(
    {WorkflowTransitionType.ASSIGNMENT, WorkflowTransitionType.REASSIGNMENT}
)


def format_workflow_event_code(*, year: int, sequence: int) -> str:
    """Format an internal workflow event code."""
    if sequence < 1 or sequence > 999999:
        raise ValidationError("Workflow event code sequence is out of range.")
    return f"WFE-{year:04d}-{sequence:06d}"


def generate_workflow_event_code(*, occurred_at=None) -> str:
    """Generate the next year-scoped workflow event code."""
    occurred_at = occurred_at or timezone.now()
    year = timezone.localtime(occurred_at).year
    latest_code = (
        WorkflowEvent.objects.select_for_update()
        .filter(event_code__startswith=f"WFE-{year:04d}-")
        .order_by("-event_code")
        .values_list("event_code", flat=True)
        .first()
    )
    if latest_code is None:
        return format_workflow_event_code(year=year, sequence=1)

    match = WORKFLOW_SEQUENCE_PATTERN.fullmatch(latest_code)
    if match is None:
        raise ValidationError("Existing workflow event code cannot be sequenced.")
    return format_workflow_event_code(year=year, sequence=int(match["sequence"]) + 1)


def _assignment_metadata_with_assignee_context(
    *,
    assignment_metadata: Mapping[str, Any] | None,
    assignee,
    transition_type: str | WorkflowTransitionType,
) -> dict[str, Any]:
    metadata = dict(assignment_metadata or {})
    if str(transition_type) in {str(value) for value in ASSIGNMENT_TRANSITION_TYPES}:
        metadata.setdefault("assignee_user_id", assignee.pk if assignee is not None else None)
    return metadata


@transaction.atomic
def transition_grievance(
    *,
    grievance,
    actor,
    new_status: str,
    transition_type: str | WorkflowTransitionType,
    assignee=None,
    transition_reason: str = "",
    remarks: str = "",
    assignment_metadata: Mapping[str, Any] | None = None,
    escalation_metadata: Mapping[str, Any] | None = None,
    sla_metadata: Mapping[str, Any] | None = None,
    status_metadata: Mapping[str, Any] | None = None,
) -> WorkflowEvent:
    """Update grievance status and record the workflow history atomically."""
    occurred_at = timezone.now()
    previous_status = grievance.status
    assignment_context = _assignment_metadata_with_assignee_context(
        assignment_metadata=assignment_metadata,
        assignee=assignee,
        transition_type=transition_type,
    )
    change_grievance_status(
        grievance=grievance,
        status=new_status,
        reason=transition_reason,
        metadata=status_metadata,
    )
    event = WorkflowEvent(
        event_code=generate_workflow_event_code(occurred_at=occurred_at),
        grievance=grievance,
        actor=actor,
        assignee=assignee,
        transition_type=str(transition_type),
        previous_status=previous_status,
        new_status=str(new_status),
        transition_reason=transition_reason,
        remarks=remarks,
        assignment_metadata=assignment_context,
        escalation_metadata=dict(escalation_metadata or {}),
        sla_metadata=dict(sla_metadata or {}),
        occurred_at=occurred_at,
    )
    event.full_clean()
    event.save()
    workflow_event_recorded.send(sender=WorkflowEvent, workflow_event=event)
    return event


@transaction.atomic
def escalate_grievance_from_system(
    *,
    grievance,
    transition_reason: str,
    escalation_metadata: Mapping[str, Any] | None = None,
) -> WorkflowEvent:
    """Record a system-generated escalation alert without changing grievance status.

    Escalation is an **urgency marker and municipal-admin alert**, not a
    lifecycle state.  The grievance continues through its normal lifecycle
    (submitted → triaged/assigned → in_progress → resolved → closed).
    This function records the escalation fact so that:

    * The audit trail shows when and why the escalation was triggered.
    * Downstream signal handlers can notify municipal_admin.
    * The ``WorkflowEvent.escalation_metadata`` carries the AI reasoning.
    * ``previous_status == new_status`` so no citizen email is fired.

    A ``__system__`` user (role ``system_operator``) is get-or-created as the
    actor so this function can be called from automated paths (AI enrichment,
    SLA breach monitors) without a human user in context.

    Parameters
    ----------
    grievance
        A :class:`~apps.grievances.models.Grievance` instance.
        Its ``status`` is **not** changed.
    transition_reason
        Human-readable reason for the escalation (stored on the event).
    escalation_metadata
        Optional JSON-serialisable dict persisted in
        :attr:`WorkflowEvent.escalation_metadata`.
    """
    User = get_user_model()
    system_user, _ = User.objects.get_or_create(
        username="__system__",
        defaults={
            "role": "system_operator",
            "is_active": True,
            "is_staff": False,
        },
    )

    occurred_at = timezone.now()
    event = WorkflowEvent(
        event_code=generate_workflow_event_code(occurred_at=occurred_at),
        grievance=grievance,
        actor=system_user,
        transition_type=WorkflowTransitionType.ESCALATION,
        # previous_status == new_status: escalation is an alert, not a state change.
        previous_status=grievance.status,
        new_status=grievance.status,
        transition_reason=transition_reason,
        remarks="Automatically escalated by system.",
        escalation_metadata=dict(escalation_metadata or {}),
        occurred_at=occurred_at,
    )
    event.full_clean()
    event.save()
    workflow_event_recorded.send(sender=WorkflowEvent, workflow_event=event)

    # Lazy import — avoids circular dependency at module level.
    from apps.audit.models import AuditActionType  # noqa: PLC0415
    from apps.audit.services import record_system_audit_event  # noqa: PLC0415

    record_system_audit_event(
        target_model="grievances.Grievance",
        target_object_id=str(grievance.pk),
        action_type=AuditActionType.ESCALATION,
        change_metadata={
            "tracking_code": grievance.tracking_code,
            "escalation_reason": transition_reason,
            "source": (dict(escalation_metadata or {})).get("source", "system"),
        },
        remarks="System-generated escalation alert persisted to workflow history.",
    )
    return event


@transaction.atomic
def return_grievance_to_intake(
    *,
    grievance,
    actor,
    transition_reason: str,
    remarks: str = "",
) -> WorkflowEvent:
    """Return a wrongly-routed complaint to TRIAGED status for re-routing.

    Governance rule: a department_officer who receives a complaint that does
    not belong to their department may not reroute it directly.  They use this
    action to send it back to the TRIAGED state so that municipal_admin can
    re-assign it to the correct department.

    Side effects
    ------------
    * ``grievance.department`` is cleared (set to NULL) — the complaint is
      explicitly unassigned so municipal_admin can see it needs re-routing.
    * ``grievance.status`` moves to TRIAGED.
    * A ``RETURN`` workflow event is recorded with the supplied reason.

    This is NOT rejection — the complaint remains open and visible.
    """
    from apps.grievances.models import GrievanceStatus
    from apps.grievances.services import change_grievance_status

    previous_status = grievance.status

    # Clear department assignment so the complaint is visibly unrouted.
    grievance.department = None
    from django.utils import timezone as _tz
    grievance.save(update_fields=["department", "updated_at"])

    change_grievance_status(
        grievance=grievance,
        status=GrievanceStatus.TRIAGED,
        reason=transition_reason,
        metadata={
            "return_reason": transition_reason,
            "returned_by_role": getattr(actor, "role", "unknown"),
        },
    )

    occurred_at = _tz.now()
    event = WorkflowEvent(
        event_code=generate_workflow_event_code(occurred_at=occurred_at),
        grievance=grievance,
        actor=actor,
        transition_type=WorkflowTransitionType.RETURN,
        previous_status=previous_status,
        new_status=GrievanceStatus.TRIAGED,
        transition_reason=transition_reason,
        remarks=remarks,
        occurred_at=occurred_at,
    )
    event.full_clean()
    event.save()
    workflow_event_recorded.send(sender=WorkflowEvent, workflow_event=event)
    return event


@transaction.atomic
def record_workflow_comment(
    *,
    grievance,
    actor,
    remarks: str,
    transition_reason: str = "",
    sla_metadata: Mapping[str, Any] | None = None,
) -> WorkflowEvent:
    """Record a history comment without changing grievance current state."""
    occurred_at = timezone.now()
    event = WorkflowEvent(
        event_code=generate_workflow_event_code(occurred_at=occurred_at),
        grievance=grievance,
        actor=actor,
        transition_type=WorkflowTransitionType.COMMENT,
        previous_status=grievance.status,
        new_status=grievance.status,
        transition_reason=transition_reason,
        remarks=remarks,
        sla_metadata=dict(sla_metadata or {}),
        occurred_at=occurred_at,
    )
    event.full_clean()
    event.save()
    workflow_event_recorded.send(sender=WorkflowEvent, workflow_event=event)
    return event
