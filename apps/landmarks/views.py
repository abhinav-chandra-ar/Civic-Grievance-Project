"""DRF views for landmarks."""
from __future__ import annotations

from rest_framework import mixins, viewsets
from rest_framework.permissions import IsAuthenticated

from .permissions import IsLandmarkManagerRole
from .selectors import landmark_list
from .serializers import LandmarkSerializer, LandmarkWriteSerializer


class LandmarkViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    viewsets.GenericViewSet,
):
    """Landmark GeoJSON directory and catalog maintenance endpoints."""

    search_fields = ("code", "primary_name")
    ordering_fields = ("code", "primary_name", "landmark_type", "created_at", "updated_at")

    def get_queryset(self):
        include_inactive = self.action in {"update", "partial_update"}
        return landmark_list(active_only=not include_inactive)

    def get_permissions(self):
        if self.action in {"create", "update", "partial_update"}:
            return [IsAuthenticated(), IsLandmarkManagerRole()]
        return [IsAuthenticated()]

    def get_serializer_class(self):
        if self.action in {"create", "update", "partial_update"}:
            return LandmarkWriteSerializer
        return LandmarkSerializer
