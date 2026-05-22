"""App configuration for the attachments app."""
from __future__ import annotations

from django.apps import AppConfig


class AttachmentsConfig(AppConfig):
    name = "apps.attachments"
    label = "attachments"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Attachments"

    def ready(self) -> None:
        from . import signals  # noqa: F401
