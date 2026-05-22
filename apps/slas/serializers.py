"""DRF serializers for SLA state."""
from __future__ import annotations

from typing import Any

from rest_framework import serializers

from .models import SLA
from .services import create_sla_for_grievance, mark_sla_breached, update_sla_deadlines


class SLASerializer(serializers.ModelSerializer[SLA]):
    """Read representation for current grievance SLA state."""

    class Meta:
        model = SLA
        fields = (
            "id",
            "sla_code",
            "grievance",
            "response_due_at",
            "resolution_due_at",
            "sla_status",
            "is_breached",
            "breach_type",
            "breached_at",
            "next_escalation_due_at",
            "breach_metadata",
            "escalation_metadata",
            "policy_snapshot_metadata",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields


class SLACreateSerializer(serializers.ModelSerializer[SLA]):
    """Create the one current SLA row for a grievance."""

    class Meta:
        model = SLA
        fields = (
            "grievance",
            "response_due_at",
            "resolution_due_at",
            "next_escalation_due_at",
            "escalation_metadata",
            "policy_snapshot_metadata",
        )

    def create(self, validated_data: dict[str, Any]) -> SLA:
        return create_sla_for_grievance(**validated_data)


class SLADeadlineUpdateSerializer(serializers.ModelSerializer[SLA]):
    """Update SLA deadlines and policy/escalation hooks."""

    class Meta:
        model = SLA
        fields = (
            "response_due_at",
            "resolution_due_at",
            "next_escalation_due_at",
            "escalation_metadata",
            "policy_snapshot_metadata",
        )

    def update(self, instance: SLA, validated_data: dict[str, Any]) -> SLA:
        return update_sla_deadlines(sla=instance, **validated_data)


class SLABreachMarkSerializer(serializers.ModelSerializer[SLA]):
    """Explicit breach marking serializer."""

    class Meta:
        model = SLA
        fields = ("breach_type", "breach_metadata", "breached_at")

    def update(self, instance: SLA, validated_data: dict[str, Any]) -> SLA:
        return mark_sla_breached(sla=instance, **validated_data)
