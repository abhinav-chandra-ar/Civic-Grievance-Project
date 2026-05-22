"""URL routes owned by the attachments app."""
from __future__ import annotations

from rest_framework.routers import DefaultRouter

from .views import AttachmentViewSet

app_name = "attachments"

router = DefaultRouter()
router.register("", AttachmentViewSet, basename="attachment")

urlpatterns = router.urls
