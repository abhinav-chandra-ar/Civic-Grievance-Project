"""Permission tests for attachment metadata writes."""
from __future__ import annotations

from dataclasses import dataclass

from rest_framework.test import APIRequestFactory

from apps.attachments.permissions import IsAttachmentOperatorRole


@dataclass
class UserStub:
    role: str
    is_authenticated: bool = True


def test_attachment_operator_permission_accepts_field_verifier() -> None:
    request = APIRequestFactory().patch("/attachments/1/")
    request.user = UserStub(role="field_verifier")

    assert IsAttachmentOperatorRole().has_permission(request, view=None)


def test_attachment_operator_permission_rejects_citizen() -> None:
    request = APIRequestFactory().patch("/attachments/1/")
    request.user = UserStub(role="citizen")

    assert not IsAttachmentOperatorRole().has_permission(request, view=None)
