"""Append-oriented audit log model."""
from __future__ import annotations

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator
from django.db import models

AUDIT_CODE_VALIDATOR = RegexValidator(
    regex=r"^AUD-\d{4}-\d{6}$",
    message="Audit codes must use the AUD-YYYY-NNNNNN format.",
)


class AuditActionType(models.TextChoices):
    CREATE = "create", "Create"
    UPDATE = "update", "Update"
    DELETE = "delete", "Delete"
    LOGIN = "login", "Login"
    LOGOUT = "logout", "Logout"
    PERMISSION_DENIED = "permission_denied", "Permission denied"
    STATUS_CHANGE = "status_change", "Status change"
    ASSIGNMENT = "assignment", "Assignment"
    ESCALATION = "escalation", "Escalation"
    BREACH = "breach", "Breach"
    EXPORT = "export", "Export"
    ADMIN_ACTION = "admin_action", "Admin action"
    SYSTEM_EVENT = "system_event", "System event"


def validate_non_empty_text(value: str) -> None:
    """Reject blank target reference values."""
    if not value.strip():
        raise ValidationError("This value must not be empty.")


def validate_metadata_mapping(value: object) -> None:
    """Keep audit context hooks object-shaped."""
    if not isinstance(value, dict):
        raise ValidationError("Metadata hooks must be a JSON object.")


class AuditLog(models.Model):
    """System activity/security log row created through audit services."""

    audit_code = models.CharField(
        max_length=15,
        unique=True,
        db_index=True,
        blank=True,
        validators=[AUDIT_CODE_VALIDATOR],
    )
    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="audit_logs",
    )
    target_model = models.CharField(
        max_length=128,
        db_index=True,
        validators=[validate_non_empty_text],
    )
    target_object_id = models.CharField(
        max_length=64,
        db_index=True,
        validators=[validate_non_empty_text],
    )
    action_type = models.CharField(
        max_length=32,
        choices=AuditActionType.choices,
        db_index=True,
    )
    request_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    change_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    security_metadata = models.JSONField(
        blank=True,
        default=dict,
        validators=[validate_metadata_mapping],
    )
    remarks = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "audit_audit_log"
        verbose_name = "Audit log"
        verbose_name_plural = "Audit logs"
        ordering = ("-created_at", "-id")
        indexes = [
            models.Index(fields=["actor", "created_at"], name="audit_actor_created_idx"),
            models.Index(
                fields=["target_model", "target_object_id", "created_at"],
                name="audit_target_created_idx",
            ),
            models.Index(fields=["action_type", "created_at"], name="audit_action_created_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(action_type__in=AuditActionType.values),
                name="audit_action_type_valid",
            ),
        ]

    def __str__(self) -> str:
        return self.audit_code or f"Audit log {self.pk or 'unsaved'}"


__all__ = ["AuditActionType", "AuditLog", "validate_metadata_mapping", "validate_non_empty_text"]
