"""DRF views for wards."""
from __future__ import annotations

from rest_framework import mixins, viewsets
from rest_framework.permissions import IsAuthenticated

from .permissions import IsWardManagerRole
from .selectors import ward_list
from .serializers import WardSerializer, WardWriteSerializer


class WardViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    viewsets.GenericViewSet,
):
    """Ward GeoJSON directory and boundary maintenance endpoints."""

    search_fields = ("code", "name")
    ordering_fields = ("code", "name", "created_at", "updated_at")

    def get_queryset(self):
        include_inactive = self.action in {"update", "partial_update"}
        return ward_list(active_only=not include_inactive)

    def get_permissions(self):
        if self.action in {"create", "update", "partial_update"}:
            return [IsAuthenticated(), IsWardManagerRole()]
        return [IsAuthenticated()]

    def get_serializer_class(self):
        if self.action in {"create", "update", "partial_update"}:
            return WardWriteSerializer
        return WardSerializer
