"""App configuration for the slas app."""
from __future__ import annotations

from django.apps import AppConfig


class SlasConfig(AppConfig):
    name = "apps.slas"
    label = "slas"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Slas"

    def ready(self) -> None:
        from . import signals  # noqa: F401
