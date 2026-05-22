"""Django admin registration for workflow history."""
from __future__ import annotations

from django.contrib import admin

from .models import WorkflowEvent


@admin.register(WorkflowEvent)
class WorkflowEventAdmin(admin.ModelAdmin):
    """Admin surface for grievance workflow history."""

    list_display = (
        "event_code",
        "grievance",
        "transition_type",
        "previous_status",
        "new_status",
        "actor",
        "assignee",
        "occurred_at",
    )
    list_filter = ("transition_type", "previous_status", "new_status", "occurred_at")
    search_fields = ("event_code", "grievance__tracking_code", "transition_reason", "remarks")
    readonly_fields = ("created_at", "updated_at")
