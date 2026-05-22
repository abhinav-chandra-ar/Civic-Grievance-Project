"""DRF views for grievances."""
from __future__ import annotations

from rest_framework import mixins, viewsets
from rest_framework.permissions import IsAuthenticated

from .permissions import IsGrievanceOperatorRole, IsSubmitterOrGrievanceOperatorRole
from .selectors import grievance_list_visible_to_user
from .serializers import GrievanceEnrichmentSerializer, GrievanceSerializer, GrievanceSubmitSerializer


class GrievanceViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    viewsets.GenericViewSet,
):
    """Submission, read, and pre-workflow enrichment endpoints."""

    search_fields = ("tracking_code", "raw_text", "normalized_summary", "category_code")
    ordering_fields = ("submitted_at", "updated_at", "priority", "status")

    def get_queryset(self):
        return grievance_list_visible_to_user(user=self.request.user)

    def get_permissions(self):
        if self.action in {"update", "partial_update"}:
            return [IsAuthenticated(), IsGrievanceOperatorRole()]
        if self.action == "retrieve":
            return [IsAuthenticated(), IsSubmitterOrGrievanceOperatorRole()]
        return [IsAuthenticated()]

    def get_serializer_class(self):
        if self.action == "create":
            return GrievanceSubmitSerializer
        if self.action in {"update", "partial_update"}:
            return GrievanceEnrichmentSerializer
        return GrievanceSerializer

    def perform_create(self, serializer):
        serializer.save(submitter=self.request.user)
