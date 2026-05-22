"""DRF views for workflow history."""
from __future__ import annotations

from rest_framework import mixins, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .permissions import IsWorkflowEventVisible, IsWorkflowOperatorRole
from .selectors import workflow_event_list_visible_to_user
from .serializers import WorkflowCommentSerializer, WorkflowEventSerializer, WorkflowTransitionSerializer


class WorkflowEventViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    viewsets.GenericViewSet,
):
    """Workflow transition history and operational event creation."""

    filterset_fields = ["grievance"]
    search_fields = ("event_code", "grievance__tracking_code", "transition_reason", "remarks")
    ordering_fields = ("occurred_at", "created_at", "transition_type", "new_status")

    def get_queryset(self):
        return workflow_event_list_visible_to_user(user=self.request.user)

    def get_permissions(self):
        if self.action in {"create", "comment"}:
            return [IsAuthenticated(), IsWorkflowOperatorRole()]
        if self.action == "retrieve":
            return [IsAuthenticated(), IsWorkflowEventVisible()]
        return [IsAuthenticated()]

    def get_serializer_class(self):
        if self.action == "create":
            return WorkflowTransitionSerializer
        if self.action == "comment":
            return WorkflowCommentSerializer
        return WorkflowEventSerializer

    def perform_create(self, serializer):
        serializer.save(actor=self.request.user)

    @action(detail=False, methods=["post"])
    def comment(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        event = serializer.save(actor=request.user)
        return Response(WorkflowEventSerializer(event).data)
