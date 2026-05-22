"""URL routes owned by the grievances app."""
from __future__ import annotations

from rest_framework.routers import DefaultRouter

from .views import GrievanceViewSet

app_name = "grievances"

router = DefaultRouter()
router.register("", GrievanceViewSet, basename="grievance")

urlpatterns = router.urls
