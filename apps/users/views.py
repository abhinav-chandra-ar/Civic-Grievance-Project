"""DRF views for users."""
from __future__ import annotations

from rest_framework import mixins, viewsets
from rest_framework.generics import RetrieveUpdateAPIView
from rest_framework.permissions import IsAuthenticated

from .permissions import IsSelfOrUserAdminRole, IsUserAdminRole
from .selectors import user_list
from .serializers import UserProfileUpdateSerializer, UserSerializer


class CurrentUserView(RetrieveUpdateAPIView):
    """Read or update the authenticated user's mutable profile."""

    permission_classes = (IsAuthenticated,)

    def get_object(self):
        return self.request.user

    def get_serializer_class(self):
        if self.request.method in {"PUT", "PATCH"}:
            return UserProfileUpdateSerializer
        return UserSerializer


class UserViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    """Administrative user directory endpoints."""

    serializer_class = UserSerializer
    permission_classes = (IsAuthenticated, IsUserAdminRole, IsSelfOrUserAdminRole)
    search_fields = ("username", "first_name", "last_name", "email", "phone_number")
    ordering_fields = ("id", "date_joined", "username", "role")

    def get_queryset(self):
        return user_list(active_only=False)
