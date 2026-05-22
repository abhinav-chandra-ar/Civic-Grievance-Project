"""URL routes owned by the workflows app."""
from __future__ import annotations

from rest_framework.routers import DefaultRouter

from .views import WorkflowEventViewSet

app_name = "workflows"

router = DefaultRouter()
router.register("", WorkflowEventViewSet, basename="workflow-event")

urlpatterns = router.urls
