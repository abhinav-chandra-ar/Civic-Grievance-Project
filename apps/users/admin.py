"""Django admin registration for users."""
from __future__ import annotations

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin

from .models import User


@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    """Admin surface for the project custom user model."""

    list_display = (
        "username",
        "email",
        "phone_number",
        "role",
        "preferred_language",
        "is_active",
        "is_staff",
    )
    list_filter = DjangoUserAdmin.list_filter + ("role", "preferred_language")
    search_fields = ("username", "first_name", "last_name", "email", "phone_number")
    readonly_fields = ("last_login", "date_joined")
    fieldsets = DjangoUserAdmin.fieldsets + (
        (
            "Grievance profile",
            {
                "fields": (
                    "role",
                    "phone_number",
                    "preferred_language",
                    "additional_translations",
                )
            },
        ),
    )
    add_fieldsets = DjangoUserAdmin.add_fieldsets + (
        (
            "Grievance profile",
            {
                "classes": ("wide",),
                "fields": ("role", "phone_number", "preferred_language"),
            },
        ),
    )
