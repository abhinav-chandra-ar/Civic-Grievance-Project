"""DRF views for attachments."""
from __future__ import annotations

from rest_framework import mixins, viewsets
from rest_framework.permissions import IsAuthenticated

from .permissions import IsAttachmentOperatorRole, IsAttachmentOwnerOrOperatorRole
from .selectors import attachment_list_visible_to_user
from .serializers import AttachmentMetadataSerializer, AttachmentRegisterSerializer, AttachmentSerializer


class AttachmentViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    viewsets.GenericViewSet,
):
    """Attachment registration, reads, and validation metadata updates."""

    search_fields = ("attachment_code", "original_filename", "content_hash", "storage_reference")
    ordering_fields = ("uploaded_at", "updated_at", "content_type", "file_size_bytes")

    def get_queryset(self):
        return attachment_list_visible_to_user(user=self.request.user)

    def get_permissions(self):
        if self.action in {"update", "partial_update"}:
            return [IsAuthenticated(), IsAttachmentOperatorRole()]
        if self.action == "retrieve":
            return [IsAuthenticated(), IsAttachmentOwnerOrOperatorRole()]
        return [IsAuthenticated()]

    def get_serializer_class(self):
        if self.action == "create":
            return AttachmentRegisterSerializer
        if self.action in {"update", "partial_update"}:
            return AttachmentMetadataSerializer
        return AttachmentSerializer

    def perform_create(self, serializer):
        serializer.save(uploader=self.request.user)
