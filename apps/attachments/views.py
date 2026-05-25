"""DRF views for attachments."""
from __future__ import annotations

from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .permissions import IsAttachmentOperatorRole, IsAttachmentOwnerOrOperatorRole
from .selectors import attachment_list_visible_to_user
from .serializers import (
    AttachmentMetadataSerializer,
    AttachmentRegisterSerializer,
    AttachmentSerializer,
    AttachmentUploadSerializer,
)


class AttachmentViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    viewsets.GenericViewSet,
):
    """Attachment registration, reads, and validation metadata updates.

    Endpoints
    ---------
    POST   /api/v1/attachments/          — metadata-only registration (external storage)
    POST   /api/v1/attachments/upload/   — multipart binary upload (local FileStorage + CLIP)
    GET    /api/v1/attachments/          — list visible attachments
    GET    /api/v1/attachments/{id}/     — retrieve single attachment with full vision metadata
    PATCH  /api/v1/attachments/{id}/     — update validation/moderation metadata (operators only)
    """

    filterset_fields = ["grievance", "is_active"]
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
        if self.action == "upload":
            return AttachmentUploadSerializer
        if self.action in {"update", "partial_update"}:
            return AttachmentMetadataSerializer
        return AttachmentSerializer

    def perform_create(self, serializer):
        serializer.save(uploader=self.request.user)

    @action(
        detail=False,
        methods=["post"],
        url_path="upload",
        url_name="upload",
        parser_classes=[MultiPartParser, FormParser],
    )
    def upload(self, request, *args, **kwargs):
        """Accept a real multipart image upload and return the full AI analysis.

        Request (multipart/form-data)
        -----------------------------
        grievance       integer   PK of the parent grievance
        image_file      file      Raw image binary (JPEG, PNG, WEBP, GIF, …)
        attachment_metadata  JSON  Optional extra metadata (default {})

        Response (201 Created)
        ----------------------
        Full AttachmentSerializer payload including:
          image_validation_metadata       — Pillow quality/validity checks
          image_issue_classification_metadata — CLIP zero-shot class + scores
          image_text_consistency_metadata — supports | contradicts | uncertain
        The parent grievance's status_metadata.vision_analysis is also updated.
        """
        serializer = AttachmentUploadSerializer(
            data=request.data,
            context={"request": request},
        )
        serializer.is_valid(raise_exception=True)
        attachment = serializer.save(uploader=request.user)
        # Re-fetch from DB so the response reflects post-signal analysis results
        attachment.refresh_from_db()
        return Response(
            AttachmentSerializer(attachment, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )
