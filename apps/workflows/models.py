"""Workflow history events for grievance lifecycle transitions."""
from __future__ import annotations

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator
from django.db import models

from apps.grievances.models import GrievanceStatus

WORKFLOW_EVENT_CODE_VALIDATOR = RegexValidator(
    regex=r"^WFE-\d{4}-\d{6}$",
    message="Workflow event codes must use the WFE-YYYY-NNNNNN format.",
)


class WorkflowTransitionType(models.TextChoices):
    STATUS_CHANGE = "status_change", "Status change"
    ASSIGNMENT = "assignment", "Assignment"
    REASSIGNMENT = "reassignment", "Reassignment"
    ESCALATION = "escalation", "Escalation"
    RESOLUTION = "resolution", "Resolution"
    REJECTION = "rejection", "Rejection"
    CLOSURE = "closure", "Closure"
    COMMENT = "comment", "Comment"
    # RETURN: complaint is valid but wrongly routed; sends back to TRIAGED for
    # municipal_admin re-routing. Not the same as REJECTION (complaint stays open).
    RETURN = "return", "Return to intake"


def validate_metadata_mapping(value: object) -> None:
    """Keep workflow integration hooks object-shaped."""
    if not isinstance(value, dict):
        raise ValidationError("Metadata hooks must be a JSON object.")


class WorkflowEvent(models.Model):
    """Immutable-oriented event history for one grievance transition."""

    event_code = models.CharField(
        max_length=15,
        unique=True,
        db_index=True,
        blank=True,
        validators=[WORKFLOW_EVENT_CODE_VALIDATOR],
    )
    grievance = models.ForeignKey(
        "grievances.Grievance",
        on_delete=models.CASCADE,
        related_name="workflow_events",
    )
    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="workflow_events_as_actor",
    )
    assignee = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="workflow_events_as_assignee",
    )
    transition_type = models.CharField(
        max_length=32,
        choices=WorkflowTransitionType.choices,
        db_index=True,
    )
    previous_status = models.CharField(max_length=32, choices=GrievanceStatus.choices)
    new_status = models.CharField(max_length=32, choices=GrievanceStatus.choices)
    transition_reason = models.TextField(blank=True)
    remarks = models.TextField(blank=True)
    assignment_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    escalation_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    sla_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    occurred_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "workflows_workflow_event"
        verbose_name = "Workflow event"
        verbose_name_plural = "Workflow events"
        ordering = ("-occurred_at", "-id")
        indexes = [
            models.Index(fields=["grievance", "occurred_at"], name="wf_grievance_occurred_idx"),
            models.Index(fields=["grievance", "new_status"], name="wf_grievance_status_idx"),
            models.Index(fields=["actor", "occurred_at"], name="wf_actor_occurred_idx"),
            models.Index(fields=["assignee", "occurred_at"], name="wf_assignee_occurred_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(transition_type__in=WorkflowTransitionType.values),
                name="wf_transition_type_valid",
            ),
            models.CheckConstraint(
                condition=models.Q(previous_status__in=GrievanceStatus.values),
                name="wf_previous_status_valid",
            ),
            models.CheckConstraint(
                condition=models.Q(new_status__in=GrievanceStatus.values),
                name="wf_new_status_valid",
            ),
        ]

    def __str__(self) -> str:
        return self.event_code or f"Workflow event {self.pk or 'unsaved'}"


__all__ = ["WorkflowEvent", "WorkflowTransitionType", "validate_metadata_mapping"]

