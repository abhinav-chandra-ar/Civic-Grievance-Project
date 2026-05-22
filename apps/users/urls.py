"""URL routes owned by the users app."""
from __future__ import annotations

from django.urls import path
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from .views import CurrentUserView, RegisterView, UserViewSet

app_name = "users"

router = DefaultRouter()
router.register("", UserViewSet, basename="user")

urlpatterns = [
    # Auth endpoints — no authentication required.
    path("register/", RegisterView.as_view(), name="register"),
    path("token/", TokenObtainPairView.as_view(), name="token-obtain"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token-refresh"),
    # Authenticated profile endpoint.
    path("me/", CurrentUserView.as_view(), name="me"),
    # Administrative user directory (requires IsUserAdminRole).
    *router.urls,
]
