"""Django admin registration for landmarks."""
from __future__ import annotations

from django.contrib import admin
from django.contrib.gis.admin import GISModelAdmin

from .models import Landmark


@admin.register(Landmark)
class LandmarkAdmin(GISModelAdmin):
    """Admin surface for civic landmark points and lookup metadata."""

    list_display = ("code", "primary_name", "landmark_type", "is_active", "updated_at")
    list_filter = ("landmark_type", "is_active")
    search_fields = ("code", "primary_name")
    readonly_fields = ("created_at", "updated_at")
