"""App configuration for the wards app."""
from __future__ import annotations

from django.apps import AppConfig


class WardsConfig(AppConfig):
    name = "apps.wards"
    label = "wards"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Wards"

    def ready(self) -> None:
        from . import signals  # noqa: F401
