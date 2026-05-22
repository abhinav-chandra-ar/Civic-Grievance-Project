"""Public v1 API route aggregation."""
from __future__ import annotations

from django.urls import include, path

app_name = "v1"

urlpatterns = [
    path("users/", include("apps.users.urls")),
    path("departments/", include("apps.departments.urls")),
    path("wards/", include("apps.wards.urls")),
    path("landmarks/", include("apps.landmarks.urls")),
    path("grievances/", include("apps.grievances.urls")),
    path("attachments/", include("apps.attachments.urls")),
    path("workflows/", include("apps.workflows.urls")),
    path("slas/", include("apps.slas.urls")),
    path("audit/", include("apps.audit.urls")),
]
