"""App configuration for the integrations app."""
from __future__ import annotations

from django.apps import AppConfig


class IntegrationsConfig(AppConfig):
    name = "apps.integrations"
    label = "integrations"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Integrations"

    def ready(self) -> None:
        from . import handlers  # noqa: F401  — registers signal receivers
        from . import signals  # noqa: F401   — ensures integration_call_completed is importable
