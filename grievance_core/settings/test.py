"""Test settings — used by pytest. Optimized for speed, never for production."""
from __future__ import annotations

from .base import *  # noqa: F401, F403

DEBUG = False
ALLOWED_HOSTS = ["*"]
SECRET_KEY = "test-secret-key-not-used-in-prod"  # noqa: S105

# Fast password hashing in tests
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]

# In-memory cache
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "grievance-core-test",
    },
}

# Run Celery tasks synchronously in tests
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

# Serve MEDIA_ROOT files via Django's dev server in tests so that
# the upload endpoint can be called and media paths verified.
SERVE_MEDIA_IN_TESTS = True
