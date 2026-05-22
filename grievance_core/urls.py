"""Root URL configuration for grievance-core."""
from __future__ import annotations

from django.contrib import admin
from django.urls import include, path

from grievance_core import health

urlpatterns = [
    # K8s probes
    path("health/live", health.liveness, name="liveness"),
    path("health/ready", health.readiness, name="readiness"),
    path("health/startup", health.startup, name="startup"),

    # Django admin (locked down to super_admin role in Module 7)
    path("admin/", admin.site.urls),

    # Public API consumed by frontends
    path("api/v1/", include("api.v1.urls")),

    # Service-to-service API consumed by other backend services
    path("api/internal/", include("api.internal.urls")),
]
