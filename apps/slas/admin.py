"""Django admin registration for SLAs."""
from __future__ import annotations

from django.contrib import admin

from .models import SLA


@admin.register(SLA)
class SLAAdmin(admin.ModelAdmin):
    """Admin surface for current grievance SLA state."""

    list_display = (
        "sla_code",
        "grievance",
        "sla_status",
        "breach_type",
        "is_breached",
        "response_due_at",
        "resolution_due_at",
    )
    list_filter = ("sla_status", "breach_type", "is_breached")
    search_fields = ("sla_code", "grievance__tracking_code")
    readonly_fields = ("created_at", "updated_at")
