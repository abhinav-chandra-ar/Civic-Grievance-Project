"""DRF serializers for grievance submission and enrichment."""
from __future__ import annotations

from typing import Any

from rest_framework import serializers

from .models import Grievance, PRIORITY_SEVERITY_ORDER
from .services import create_grievance_with_foundation_records, update_grievance_enrichment

# Fields a ward_officer may never write during enrichment.
# These are routing fields — ward officers can only flag issues, not reroute.
_WARD_OFFICER_FORBIDDEN_FIELDS = frozenset({"department", "ward", "category_code"})

# Fields a department_officer may not set directly — they must use RETURN transition.
_DEPT_OFFICER_FORBIDDEN_ROUTING_FIELDS = frozenset({"department", "ward"})


class GrievanceSerializer(serializers.ModelSerializer[Grievance]):
    """Read representation for citizen and operational grievance views."""

    class Meta:
        model = Grievance
        fields = (
            "id",
            "tracking_code",
            "submitter",
            "raw_text",
            "landmark_mention",
            "citizen_location_text",
            "image_attachment_metadata",
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
            "status",
            "status_reason",
            "status_metadata",
            "submitted_at",
            "last_status_changed_at",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields


class GrievanceSubmitSerializer(serializers.ModelSerializer[Grievance]):
    """Citizen submission fields backed by tracking-code generation.

    ``id`` is included as a read-only response field so the frontend can
    immediately register attachments against the newly-created grievance
    without a follow-up lookup.
    """

    class Meta:
        model = Grievance
        fields = (
            "id",
            "raw_text",
            "landmark_mention",
            "citizen_location_text",
            "image_attachment_metadata",
        )
        read_only_fields = ("id",)

    def create(self, validated_data: dict[str, Any]) -> Grievance:
        return create_grievance_with_foundation_records(**validated_data)


class GrievanceEnrichmentSerializer(serializers.ModelSerializer[Grievance]):
    """Operational mapping and metadata fields before workflow ownership exists."""

    class Meta:
        model = Grievance
        fields = (
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
        )

    def validate(self, data: dict[str, Any]) -> dict[str, Any]:
        request = self.context.get("request")
        if request is None:
            # Service-to-service call (e.g. AI pipeline) — skip role enforcement.
            return data

        role = getattr(request.user, "role", None)

        if role == "ward_officer":
            self._validate_ward_officer_fields(data)
        elif role == "department_officer":
            self._validate_dept_officer_fields(data, request.user)
        # municipal_admin and super_admin: no enrichment restrictions.

        return data

    def _validate_ward_officer_fields(self, data: dict[str, Any]) -> None:
        """Enforce ward_officer enrichment boundary (Tasks 2 & 5)."""
        for field in _WARD_OFFICER_FORBIDDEN_FIELDS:
            if field in data:
                raise serializers.ValidationError(
                    {
                        field: (
                            "Ward officers cannot change routing fields. "
                            "Add a workflow comment to flag incorrect routing, "
                            "or escalate to municipal admin."
                        )
                    }
                )

        if "priority" in data and self.instance is not None:
            current_idx = PRIORITY_SEVERITY_ORDER.index(self.instance.priority)
            new_idx = PRIORITY_SEVERITY_ORDER.index(data["priority"])
            if new_idx < current_idx:
                raise serializers.ValidationError(
                    {"priority": "Ward officers may only raise priority, not lower it."}
                )

    def _validate_dept_officer_fields(
        self, data: dict[str, Any], user: Any
    ) -> None:
        """Enforce department_officer enrichment boundary (Tasks 3, 4 & 5)."""
        for field in _DEPT_OFFICER_FORBIDDEN_ROUTING_FIELDS:
            if field in data:
                raise serializers.ValidationError(
                    {
                        field: (
                            "Department officers cannot directly change routing assignments. "
                            "Use a RETURN workflow transition to send this complaint "
                            "back for re-routing by municipal admin."
                        )
                    }
                )

        # Category must belong to the officer's department (Task 3).
        if "category_code" in data and data["category_code"]:
            dept = getattr(user, "assigned_department", None)
            if dept is not None:
                handled: list[str] = dept.handled_categories or []
                if handled and data["category_code"] not in handled:
                    raise serializers.ValidationError(
                        {
                            "category_code": (
                                f"Category '{data['category_code']}' is not handled "
                                f"by your department. "
                                f"Handled categories: {', '.join(handled)}."
                            )
                        }
                    )

        # Priority lowering requires a documented reason (Task 5).
        if "priority" in data and self.instance is not None:
            current_idx = PRIORITY_SEVERITY_ORDER.index(self.instance.priority)
            new_idx = PRIORITY_SEVERITY_ORDER.index(data["priority"])
            if new_idx < current_idx:
                if not data.get("status_reason", "").strip():
                    raise serializers.ValidationError(
                        {
                            "status_reason": (
                                "A reason is required when lowering priority. "
                                "Populate status_reason with a justification."
                            )
                        }
                    )

    def update(self, instance: Grievance, validated_data: dict[str, Any]) -> Grievance:
        return update_grievance_enrichment(grievance=instance, values=validated_data)
