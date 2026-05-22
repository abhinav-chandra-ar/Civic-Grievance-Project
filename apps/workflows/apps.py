"""App configuration for the workflows app."""
from __future__ import annotations

from django.apps import AppConfig


class WorkflowsConfig(AppConfig):
    name = "apps.workflows"
    label = "workflows"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Workflows"

    def ready(self) -> None:
        from . import signals  # noqa: F401
