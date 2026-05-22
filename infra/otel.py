"""OpenTelemetry instrumentation.

Module 1 stub. Module 20 (Monitoring hooks) wires:

    * trace exporter to the OTel Collector
    * Django auto-instrumentation
    * psycopg auto-instrumentation
    * Celery auto-instrumentation
    * custom spans on key service-layer operations
    * resource attributes (service.name, service.version, deployment.environment)
"""
from __future__ import annotations


def configure_telemetry() -> None:
    """Configure OpenTelemetry. No-op in Module 1."""
    return None
