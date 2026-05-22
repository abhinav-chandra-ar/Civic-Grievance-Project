"""Health probes for Kubernetes.

Three endpoints are exposed:

* /health/live    — liveness. Returns 200 as long as the process is responsive.
                    K8s restarts the pod on failure.
* /health/ready   — readiness. Returns 200 only when all critical dependencies
                    are reachable. K8s removes the pod from the Service when not
                    ready, but does not restart it.
* /health/startup — startup. Returns 200 once initial warm-up is complete.
                    K8s holds back liveness/readiness probes until this passes.

Module 1 returns static OKs for /ready and /startup. Module 20 (Monitoring
hooks) replaces these with real probes against PostgreSQL, Redis, Kafka, and
Qdrant, with appropriate timeouts and circuit-breaking.
"""
from __future__ import annotations

from typing import Any

from django.http import HttpRequest, JsonResponse
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_GET


@require_GET
@never_cache
def liveness(_request: HttpRequest) -> JsonResponse:
    """Process is alive and responding to HTTP."""
    return JsonResponse({"status": "alive"})


@require_GET
@never_cache
def readiness(_request: HttpRequest) -> JsonResponse:
    """Ready to serve traffic. Deep checks added in Module 20."""
    payload: dict[str, Any] = {
        "status": "ready",
        "checks": {
            "database": "skipped",
            "cache": "skipped",
            "broker": "skipped",
        },
    }
    return JsonResponse(payload)


@require_GET
@never_cache
def startup(_request: HttpRequest) -> JsonResponse:
    """Initial warm-up complete."""
    return JsonResponse({"status": "started"})
