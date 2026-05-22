"""Django admin registration for grievances."""
from __future__ import annotations

from django.contrib import admin

from .models import Grievance
from .services import create_grievance_with_foundation_records


@admin.register(Grievance)
class GrievanceAdmin(admin.ModelAdmin):
    """Admin surface for grievance foundation records."""

    list_display = (
        "tracking_code",
        "submitter",
        "status",
        "priority",
        "department",
        "ward",
        "submitted_at",
    )
    list_filter = ("status", "priority", "submitted_at")
    search_fields = ("tracking_code", "raw_text", "normalized_summary", "category_code")
    readonly_fields = ("created_at", "updated_at")

    def save_model(self, request, obj: Grievance, form, change: bool) -> None:
        if change:
            super().save_model(request, obj, form, change)
            return

        values = {
            field: getattr(obj, field)
            for field in (
                "tracking_code",
                "submitted_at",
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
            )
        }
        created = create_grievance_with_foundation_records(
            submitter=obj.submitter,
            actor=request.user,
            **values,
        )
        obj.pk = created.pk
