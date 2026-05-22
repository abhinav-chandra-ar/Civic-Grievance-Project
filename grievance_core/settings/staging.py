"""Staging settings — production-like configuration for the staging cluster.

Module 1 keeps this thin. Module 3 (environment management) introduces:
    - secrets pulled from Vault via External Secrets Operator
    - per-region overrides
    - canary-friendly feature flags
"""
from __future__ import annotations

from .base import *  # noqa: F401, F403

DEBUG = False

# Security headers
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_HSTS_SECONDS = 60 * 60  # 1h in staging, raised to 1y in production
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = False
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
X_FRAME_OPTIONS = "DENY"
