"""URL routes owned by the users app."""
from __future__ import annotations

from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import CurrentUserView, UserViewSet

app_name = "users"

router = DefaultRouter()
router.register("", UserViewSet, basename="user")

urlpatterns = [
    path("me/", CurrentUserView.as_view(), name="me"),
    *router.urls,
]
