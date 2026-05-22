"""DRF views for SLA state."""
from __future__ import annotations

from rest_framework import mixins, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .permissions import IsSLAOperatorRole, IsSLAVisible
from .selectors import sla_list_visible_to_user
from .serializers import SLABreachMarkSerializer, SLACreateSerializer, SLADeadlineUpdateSerializer, SLASerializer


class SLAViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    viewsets.GenericViewSet,
):
    """Current grievance SLA state endpoints."""

    search_fields = ("sla_code", "grievance__tracking_code")
    ordering_fields = ("response_due_at", "resolution_due_at", "updated_at", "breached_at")

    def get_queryset(self):
        return sla_list_visible_to_user(user=self.request.user)

    def get_permissions(self):
        if self.action in {"create", "update", "partial_update", "mark_breached"}:
            return [IsAuthenticated(), IsSLAOperatorRole()]
        if self.action == "retrieve":
            return [IsAuthenticated(), IsSLAVisible()]
        return [IsAuthenticated()]

    def get_serializer_class(self):
        if self.action == "create":
            return SLACreateSerializer
        if self.action in {"update", "partial_update"}:
            return SLADeadlineUpdateSerializer
        if self.action == "mark_breached":
            return SLABreachMarkSerializer
        return SLASerializer

    @action(detail=True, methods=["post"], url_path="mark-breached")
    def mark_breached(self, request, pk=None):
        sla = self.get_object()
        serializer = self.get_serializer(sla, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        updated_sla = serializer.save()
        return Response(SLASerializer(updated_sla).data)
