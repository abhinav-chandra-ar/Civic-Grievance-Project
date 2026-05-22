"""Read-only DRF views for audit logs."""
from __future__ import annotations

from rest_framework import mixins, viewsets
from rest_framework.permissions import IsAuthenticated

from .permissions import IsAuditReaderRole
from .selectors import audit_log_list
from .serializers import AuditLogSerializer


class AuditLogViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    """Read-only audit investigation endpoints."""

    serializer_class = AuditLogSerializer
    permission_classes = (IsAuthenticated, IsAuditReaderRole)
    search_fields = ("audit_code", "target_model", "target_object_id", "remarks")
    ordering_fields = ("created_at", "action_type")

    def get_queryset(self):
        return audit_log_list()
