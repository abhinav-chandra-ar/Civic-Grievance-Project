"""ASGI config for grievance-core."""
from __future__ import annotations

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "grievance_core.settings.production")

application = get_asgi_application()
