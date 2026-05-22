"""URL routes owned by the slas app."""
from __future__ import annotations

from rest_framework.routers import DefaultRouter

from .views import SLAViewSet

app_name = "slas"

router = DefaultRouter()
router.register("", SLAViewSet, basename="sla")

urlpatterns = router.urls
