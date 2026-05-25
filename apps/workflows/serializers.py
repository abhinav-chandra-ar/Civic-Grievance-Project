"""DRF serializers for workflow history and transitions."""
from __future__ import annotations

from typing import Any

from rest_framework import serializers

from apps.grievances.models import GrievanceStatus

from .models import WorkflowEvent, WorkflowTransitionType
from .services import record_workflow_comment, return_grievance_to_intake, transition_grievance


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
        if str(validated_data.get("transition_type", "")) == WorkflowTransitionType.RETURN:
            return return_grievance_to_intake(
                grievance=validated_data["grievance"],
                actor=validated_data["actor"],
                transition_reason=validated_data.get("transition_reason", ""),
                remarks=validated_data.get("remarks", ""),
            )
        return transition_grievance(**validated_data)

    def validate_transition_type(self, value: str) -> str:
        if value == WorkflowTransitionType.COMMENT:
            raise serializers.ValidationError("Use the workflow comment serializer for comments.")
        return value

    def validate(self, data: dict[str, Any]) -> dict[str, Any]:
        if str(data.get("transition_type", "")) != WorkflowTransitionType.RETURN:
            return data

        # ── RETURN-specific rules ──────────────────────────────────────────────
        # Only dept_officer and municipal_admin may RETURN a complaint.
        request = self.context.get("request")
        if request is not None:
            role = getattr(request.user, "role", None)
            if role not in ("department_officer", "municipal_admin"):
                raise serializers.ValidationError(
                    {
                        "transition_type": (
                            "Only department officers and municipal admins may return "
                            "a complaint for re-routing. Ward officers should use a "
                            "workflow comment to flag incorrect routing."
                        )
                    }
                )

        # RETURN must target TRIAGED.
        if data.get("new_status") and data["new_status"] != GrievanceStatus.TRIAGED:
            raise serializers.ValidationError(
                {"new_status": "Return transitions must target TRIAGED status."}
            )
        data["new_status"] = GrievanceStatus.TRIAGED  # enforce regardless of caller

        # A routing reason is mandatory so municipal_admin knows why it was returned.
        if not data.get("transition_reason", "").strip():
            raise serializers.ValidationError(
                {
                    "transition_reason": (
                        "A routing return reason is required. "
                        "Explain why this complaint is wrongly routed."
                    )
                }
            )
        return data


class WorkflowCommentSerializer(serializers.ModelSerializer[WorkflowEvent]):
    """Operational history comment that leaves grievance status unchanged."""

    class Meta:
        model = WorkflowEvent
        fields = ("grievance", "remarks", "transition_reason", "sla_metadata")

    def create(self, validated_data: dict[str, Any]) -> WorkflowEvent:
        return record_workflow_comment(**validated_data)
