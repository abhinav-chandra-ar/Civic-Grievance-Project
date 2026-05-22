"""App configuration for the audit app."""
from __future__ import annotations

from django.apps import AppConfig


class AuditConfig(AppConfig):
    name = "apps.audit"
    label = "audit"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Audit"

    def ready(self) -> None:
        from . import signals  # noqa: F401
