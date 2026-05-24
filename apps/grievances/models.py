"""Core grievance model and foundation validation."""
from __future__ import annotations

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator
from django.db import models

TRACKING_CODE_VALIDATOR = RegexValidator(
    regex=r"^GRV-\d{4}-\d{6}$",
    message="Tracking codes must use the GRV-YYYY-NNNNNN format.",
)
CATEGORY_CODE_VALIDATOR = RegexValidator(
    regex=r"^[a-z][a-z0-9_]*$",
    message="Use a lowercase category code containing letters, numbers, and underscores.",
)


class GrievancePriority(models.TextChoices):
    LOW = "low", "Low"
    MEDIUM = "medium", "Medium"
    HIGH = "high", "High"
    URGENT = "urgent", "Urgent"
    CRITICAL = "critical", "Critical"


class GrievanceStatus(models.TextChoices):
    SUBMITTED = "submitted", "Submitted"
    ENRICHMENT_PENDING = "enrichment_pending", "Enrichment pending"
    TRIAGED = "triaged", "Triaged"
    ASSIGNED = "assigned", "Assigned"
    IN_PROGRESS = "in_progress", "In progress"
    RESOLVED = "resolved", "Resolved"
    REJECTED = "rejected", "Rejected"
    CLOSED = "closed", "Closed"


def validate_non_empty_text(value: str) -> None:
    """Reject citizen input that contains only whitespace."""
    if not value.strip():
        raise ValidationError("Grievance text must not be empty.")


def validate_metadata_mapping(value: object) -> None:
    """Keep foundation hooks object-shaped until adjacent domains exist."""
    if not isinstance(value, dict):
        raise ValidationError("Metadata hooks must be a JSON object.")


class Grievance(models.Model):
    """Citizen-submitted grievance with nullable enrichment mappings."""

    tracking_code = models.CharField(
        max_length=15,
        unique=True,
        db_index=True,
        blank=True,
        validators=[TRACKING_CODE_VALIDATOR],
    )
    submitter = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="submitted_grievances",
    )
    raw_text = models.TextField(validators=[validate_non_empty_text])
    landmark_mention = models.CharField(max_length=255, blank=True)
    citizen_location_text = models.TextField(blank=True)
    image_attachment_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    normalized_summary = models.TextField(blank=True)
    category_code = models.CharField(
        max_length=64,
        blank=True,
        db_index=True,
        validators=[CATEGORY_CODE_VALIDATOR],
    )
    department = models.ForeignKey(
        "departments.Department",
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="grievances",
    )
    ward = models.ForeignKey(
        "wards.Ward",
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="grievances",
    )
    resolved_landmark = models.ForeignKey(
        "landmarks.Landmark",
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="resolved_grievances",
    )
    landmark_resolution_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    priority = models.CharField(
        max_length=16,
        choices=GrievancePriority.choices,
        default=GrievancePriority.MEDIUM,
        db_index=True,
    )
    possible_duplicate_of = models.ForeignKey(
        "self",
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="possible_duplicates",
    )
    duplicate_detection_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    image_validation_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    status = models.CharField(
        max_length=32,
        choices=GrievanceStatus.choices,
        default=GrievanceStatus.SUBMITTED,
        db_index=True,
    )
    status_reason = models.TextField(blank=True)
    status_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    submitted_at = models.DateTimeField()
    last_status_changed_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "grievances_grievance"
        verbose_name = "Grievance"
        verbose_name_plural = "Grievances"
        ordering = ("-submitted_at", "-id")
        indexes = [
            models.Index(fields=["status", "submitted_at"], name="grievance_status_submitted_idx"),
            models.Index(fields=["department", "status"], name="grv_department_status_idx"),
            models.Index(fields=["ward", "status"], name="grievance_ward_status_idx"),
            # Covers citizen-facing list queries: WHERE submitter_id=X ORDER BY submitted_at DESC.
            models.Index(fields=["submitter", "submitted_at"], name="grv_submitter_submitted_idx"),
            # Covers unfiltered ORDER BY submitted_at DESC (e.g. duplicate-context selector).
            models.Index(fields=["-submitted_at"], name="grv_submitted_at_desc_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(priority__in=GrievancePriority.values),
                name="grievance_priority_valid",
            ),
            models.CheckConstraint(
                condition=models.Q(status__in=GrievanceStatus.values),
                name="grievance_status_valid",
            ),
        ]

    def __str__(self) -> str:
        return self.tracking_code or f"Grievance {self.pk or 'unsaved'}"


__all__ = [
    "Grievance",
    "GrievancePriority",
    "GrievanceStatus",
    "validate_metadata_mapping",
    "validate_non_empty_text",
]
