"""URL routes owned by the wards app."""
from __future__ import annotations

from rest_framework.routers import DefaultRouter

from .views import WardViewSet

app_name = "wards"

router = DefaultRouter()
router.register("", WardViewSet, basename="ward")

urlpatterns = router.urls
