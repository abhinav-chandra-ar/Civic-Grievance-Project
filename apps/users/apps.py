"""App configuration for the users app."""
from __future__ import annotations

from django.apps import AppConfig


class UsersConfig(AppConfig):
    name = "apps.users"
    label = "users"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Users"

    def ready(self) -> None:
        # Import domain signals during app startup for callers using autodiscovery.
        from . import signals  # noqa: F401
