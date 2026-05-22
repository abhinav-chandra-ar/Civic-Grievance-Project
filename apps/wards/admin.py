"""Django admin registration for wards."""
from __future__ import annotations

from django.contrib import admin
from django.contrib.gis.admin import GISModelAdmin

from .models import Ward


@admin.register(Ward)
class WardAdmin(GISModelAdmin):
    """Admin surface for ward boundaries and metadata hooks."""

    list_display = ("code", "name", "is_active", "updated_at")
    list_filter = ("is_active",)
    search_fields = ("code", "name")
    readonly_fields = ("created_at", "updated_at")
