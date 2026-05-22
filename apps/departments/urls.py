"""URL routes owned by the departments app."""
from __future__ import annotations

from rest_framework.routers import DefaultRouter

from .views import DepartmentViewSet

app_name = "departments"

router = DefaultRouter()
router.register("", DepartmentViewSet, basename="department")

urlpatterns = router.urls
