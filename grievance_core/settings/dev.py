"""Development settings — for local developer machines only."""
from __future__ import annotations

from .base import *  # noqa: F401, F403
from .base import REST_FRAMEWORK

DEBUG = True
ALLOWED_HOSTS = ["*"]

# Browsable API is handy in dev
REST_FRAMEWORK["DEFAULT_RENDERER_CLASSES"] = [
    "rest_framework.renderers.JSONRenderer",
    "rest_framework.renderers.BrowsableAPIRenderer",
]

# Tighter feedback loop in dev: short cache
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "grievance-core-dev",
    },
}
