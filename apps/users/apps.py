"""App configuration for the users app."""
from __future__ import annotations

from django.apps import AppConfig


class UsersConfig(AppConfig):
    name = "apps.users"
    label = "users"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Users"

    def ready(self) -> None:
        from . import signals  # noqa: F401
        self._gate_django_admin_to_super_admin()

    @staticmethod
    def _gate_django_admin_to_super_admin() -> None:
        """Restrict Django admin panel to super_admin role only.

        Replaces the default AdminSite.has_permission on the shared admin.site
        instance so that every admin view rejects any user whose role is not
        super_admin, regardless of is_staff or is_superuser flags.
        Module 7 will formalise this via Keycloak scopes; this guard is the
        interim enforcement layer.
        """
        from django.contrib import admin as django_admin

        def _super_admin_only(request) -> bool:  # type: ignore[override]
            return bool(
                getattr(request.user, "is_active", False)
                and getattr(request.user, "role", None) == "super_admin"
            )

        django_admin.site.has_permission = _super_admin_only
