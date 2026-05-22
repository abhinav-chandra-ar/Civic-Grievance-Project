"""pytest fixtures shared across the test suite.

Module 1 ships the minimum: an API client fixture and a normal-user factory
hook. Module 8 (User management) adds factories per role; Module 9+ add
fixtures for grievances, wards, and departments.
"""
from __future__ import annotations

import pytest
from rest_framework.test import APIClient


@pytest.fixture
def api_client() -> APIClient:
    """Unauthenticated DRF test client."""
    return APIClient()
