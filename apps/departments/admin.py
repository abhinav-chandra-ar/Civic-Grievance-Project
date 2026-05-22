"""Django admin registration for departments."""
from __future__ import annotations

from django.contrib import admin

from .models import Department


@admin.register(Department)
class DepartmentAdmin(admin.ModelAdmin):
    """Admin surface for routing and metadata departments."""

    list_display = ("code", "name", "is_active", "updated_at")
    list_filter = ("is_active",)
    search_fields = ("code", "name")
    readonly_fields = ("created_at", "updated_at")
