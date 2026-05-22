"""Service-to-service internal API router.

These endpoints are reachable only from within the cluster network and require
a service-to-service OAuth2 client-credentials JWT (Module 6). They are
deliberately separated from /api/v1/ so they can be:

    * routed differently at the gateway
    * scoped to different Keycloak client scopes
    * load-balanced independently
    * excluded from the public OpenAPI bundle

The ingestion service uses these endpoints to persist provisional grievances
after the AI pipeline runs.
"""
from __future__ import annotations

from rest_framework.routers import DefaultRouter

app_name = "internal"

router = DefaultRouter()

urlpatterns = router.urls
