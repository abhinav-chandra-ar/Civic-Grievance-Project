"""Django admin registration for attachments."""
from __future__ import annotations

from django.contrib import admin

from .models import Attachment


@admin.register(Attachment)
class AttachmentAdmin(admin.ModelAdmin):
    """Admin surface for grievance attachment records."""

    list_display = (
        "attachment_code",
        "grievance",
        "uploader",
        "content_type",
        "file_size_bytes",
        "is_active",
        "uploaded_at",
    )
    list_filter = ("content_type", "is_active", "uploaded_at")
    search_fields = ("attachment_code", "original_filename", "content_hash", "storage_reference")
    readonly_fields = ("created_at", "updated_at")
