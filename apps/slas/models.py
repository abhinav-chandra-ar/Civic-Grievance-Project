"""SLA state model for grievance-level deadlines."""
from __future__ import annotations

from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator
from django.db import models

SLA_CODE_VALIDATOR = RegexValidator(
    regex=r"^SLA-\d{4}-\d{6}$",
    message="SLA codes must use the SLA-YYYY-NNNNNN format.",
)


class SLAStatus(models.TextChoices):
    ACTIVE = "active", "Active"
    BREACHED = "breached", "Breached"
    PAUSED = "paused", "Paused"
    SATISFIED = "satisfied", "Satisfied"
    CANCELLED = "cancelled", "Cancelled"


class SLABreachType(models.TextChoices):
    NONE = "none", "None"
    RESPONSE = "response", "Response"
    RESOLUTION = "resolution", "Resolution"
    BOTH = "both", "Both"


def validate_metadata_mapping(value: object) -> None:
    """Keep SLA hooks object-shaped until policy/escalation domains exist."""
    if not isinstance(value, dict):
        raise ValidationError("Metadata hooks must be a JSON object.")


class SLA(models.Model):
    """Current operational SLA state for one grievance."""

    sla_code = models.CharField(
        max_length=15,
        unique=True,
        db_index=True,
        blank=True,
        validators=[SLA_CODE_VALIDATOR],
    )
    grievance = models.OneToOneField(
        "grievances.Grievance",
        on_delete=models.CASCADE,
        related_name="sla",
    )
    response_due_at = models.DateTimeField()
    resolution_due_at = models.DateTimeField()
    sla_status = models.CharField(
        max_length=16,
        choices=SLAStatus.choices,
        default=SLAStatus.ACTIVE,
        db_index=True,
    )
    is_breached = models.BooleanField(default=False, db_index=True)
    breach_type = models.CharField(
        max_length=16,
        choices=SLABreachType.choices,
        default=SLABreachType.NONE,
        db_index=True,
    )
    breached_at = models.DateTimeField(blank=True, null=True)
    next_escalation_due_at = models.DateTimeField(blank=True, null=True)
    breach_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    escalation_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    policy_snapshot_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "slas_sla"
        verbose_name = "SLA"
        verbose_name_plural = "SLAs"
        ordering = ("response_due_at", "resolution_due_at")
        indexes = [
            models.Index(fields=["is_breached", "breached_at"], name="sla_breached_at_idx"),
            models.Index(fields=["sla_status", "response_due_at"], name="sla_status_response_idx"),
            models.Index(fields=["sla_status", "resolution_due_at"], name="sla_status_resolution_idx"),
            models.Index(fields=["next_escalation_due_at"], name="sla_next_escalation_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(sla_status__in=SLAStatus.values),
                name="sla_status_valid",
            ),
            models.CheckConstraint(
                condition=models.Q(breach_type__in=SLABreachType.values),
                name="sla_breach_type_valid",
            ),
        ]

    def clean(self) -> None:
        """Validate SLA deadline and breach-state consistency."""
        super().clean()
        errors = {}
        if self.response_due_at and self.resolution_due_at and self.resolution_due_at < self.response_due_at:
            errors["resolution_due_at"] = "Resolution deadline must not be before response deadline."
        if self.is_breached and self.breached_at is None:
            errors["breached_at"] = "Breached SLAs must include breached_at."
        if not self.is_breached and self.breached_at is not None:
            errors["breached_at"] = "Non-breached SLAs must not include breached_at."
        if not self.is_breached and self.breach_type != SLABreachType.NONE:
            errors["breach_type"] = "Non-breached SLAs must use breach type none."
        if self.is_breached and self.breach_type == SLABreachType.NONE:
            errors["breach_type"] = "Breached SLAs must include a concrete breach type."
        if errors:
            raise ValidationError(errors)

    def __str__(self) -> str:
        return self.sla_code or f"SLA {self.pk or 'unsaved'}"


__all__ = ["SLA", "SLABreachType", "SLAStatus", "validate_metadata_mapping"]
