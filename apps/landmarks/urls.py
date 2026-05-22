"""URL routes owned by the landmarks app."""
from __future__ import annotations

from rest_framework.routers import DefaultRouter

from .views import LandmarkViewSet

app_name = "landmarks"

router = DefaultRouter()
router.register("", LandmarkViewSet, basename="landmark")

urlpatterns = router.urls
