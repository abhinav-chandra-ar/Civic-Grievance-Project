"""Tests for the K8s health probe endpoints."""
from __future__ import annotations

from rest_framework import status
from rest_framework.test import APIClient


def test_liveness_returns_200(api_client: APIClient) -> None:
    response = api_client.get("/health/live")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "alive"}


def test_readiness_returns_200(api_client: APIClient) -> None:
    response = api_client.get("/health/ready")
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["status"] == "ready"
    assert "checks" in body


def test_startup_returns_200(api_client: APIClient) -> None:
    response = api_client.get("/health/startup")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "started"}


def test_liveness_only_allows_get(api_client: APIClient) -> None:
    response = api_client.post("/health/live")
    assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
