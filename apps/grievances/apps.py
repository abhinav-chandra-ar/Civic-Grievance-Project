"""App configuration for the grievances app."""
from __future__ import annotations

from django.apps import AppConfig


class GrievancesConfig(AppConfig):
    name = "apps.grievances"
    label = "grievances"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Grievances"

    def ready(self) -> None:
        from . import signals  # noqa: F401
