"""DRF serializers for workflow history and transitions."""
from __future__ import annotations

from typing import Any

from rest_framework import serializers

from .models import WorkflowEvent, WorkflowTransitionType
from .services import record_workflow_comment, transition_grievance


class WorkflowEventSerializer(serializers.ModelSerializer[WorkflowEvent]):
    """Read representation for workflow history events."""

    class Meta:
        model = WorkflowEvent
        fields = (
            "id",
            "event_code",
            "grievance",
            "actor",
            "assignee",
            "transition_type",
            "previous_status",
            "new_status",
            "transition_reason",
            "remarks",
            "assignment_metadata",
            "escalation_metadata",
            "sla_metadata",
            "occurred_at",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields


class WorkflowTransitionSerializer(serializers.ModelSerializer[WorkflowEvent]):
    """Operational transition request coordinated with grievance status."""

    class Meta:
        model = WorkflowEvent
        fields = (
            "grievance",
            "assignee",
            "transition_type",
            "new_status",
            "transition_reason",
            "remarks",
            "assignment_metadata",
            "escalation_metadata",
            "sla_metadata",
        )

    def create(self, validated_data: dict[str, Any]) -> WorkflowEvent:
        return transition_grievance(**validated_data)

    def validate_transition_type(self, value: str) -> str:
        if value == WorkflowTransitionType.COMMENT:
            raise serializers.ValidationError("Use the workflow comment serializer for comments.")
        return value


class WorkflowCommentSerializer(serializers.ModelSerializer[WorkflowEvent]):
    """Operational history comment that leaves grievance status unchanged."""

    class Meta:
        model = WorkflowEvent
        fields = ("grievance", "remarks", "transition_reason", "sla_metadata")

    def create(self, validated_data: dict[str, Any]) -> WorkflowEvent:
        return record_workflow_comment(**validated_data)
