"""App configuration for the landmarks app."""
from __future__ import annotations

from django.apps import AppConfig


class LandmarksConfig(AppConfig):
    name = "apps.landmarks"
    label = "landmarks"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Landmarks"

    def ready(self) -> None:
        from . import signals  # noqa: F401
