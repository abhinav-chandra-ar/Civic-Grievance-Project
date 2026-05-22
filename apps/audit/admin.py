"""Read-only Django admin registration for audit logs."""
from __future__ import annotations

from django.contrib import admin
from django.http import HttpRequest

from .models import AuditLog


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    """Investigation surface for append-oriented audit rows."""

    list_display = ("audit_code", "actor", "target_model", "target_object_id", "action_type", "created_at")
    list_filter = ("action_type", "created_at")
    search_fields = ("audit_code", "target_model", "target_object_id", "remarks")
    readonly_fields = tuple(field.name for field in AuditLog._meta.fields)

    def has_add_permission(self, request: HttpRequest) -> bool:
        return False

    def has_change_permission(self, request: HttpRequest, obj=None) -> bool:
        return False if obj is not None else super().has_change_permission(request, obj)

    def has_delete_permission(self, request: HttpRequest, obj=None) -> bool:
        return False
