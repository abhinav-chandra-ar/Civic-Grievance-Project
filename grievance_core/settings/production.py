"""Production settings — for the production cluster only."""
from __future__ import annotations

from .base import *  # noqa: F401, F403

DEBUG = False

# Strict security
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 60 * 60 * 24 * 365  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = True
X_FRAME_OPTIONS = "DENY"

# Database connection pooling via psycopg pool — tuned in Module 3
DATABASES["default"].setdefault("CONN_MAX_AGE", 300)
DATABASES["default"].setdefault("CONN_HEALTH_CHECKS", True)
