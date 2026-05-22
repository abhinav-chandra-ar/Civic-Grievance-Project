"""DRF views for departments."""
from __future__ import annotations

from rest_framework import mixins, viewsets
from rest_framework.permissions import IsAuthenticated

from .permissions import IsDepartmentManagerRole
from .selectors import department_list
from .serializers import DepartmentSerializer, DepartmentWriteSerializer


class DepartmentViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    viewsets.GenericViewSet,
):
    """Department directory and manager maintenance endpoints."""

    search_fields = ("code", "name")
    ordering_fields = ("code", "name", "created_at", "updated_at")

    def get_queryset(self):
        include_inactive = self.action in {"update", "partial_update"}
        return department_list(active_only=not include_inactive)

    def get_permissions(self):
        if self.action in {"create", "update", "partial_update"}:
            return [IsAuthenticated(), IsDepartmentManagerRole()]
        return [IsAuthenticated()]

    def get_serializer_class(self):
        if self.action in {"create", "update", "partial_update"}:
            return DepartmentWriteSerializer
        return DepartmentSerializer
