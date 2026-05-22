"""DRF serializers for audit logs."""
from __future__ import annotations

from rest_framework import serializers

from .models import AuditLog


class AuditLogSerializer(serializers.ModelSerializer[AuditLog]):
    """Read-only audit log representation."""

    class Meta:
        model = AuditLog
        fields = (
            "id",
            "audit_code",
            "actor",
            "target_model",
            "target_object_id",
            "action_type",
            "request_metadata",
            "change_metadata",
            "security_metadata",
            "remarks",
            "created_at",
        )
        read_only_fields = fields
