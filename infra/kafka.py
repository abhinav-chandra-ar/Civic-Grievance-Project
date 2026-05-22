"""Kafka producer.

Module 1 stub. Module 18 (Event handlers) wires:

    * confluent-kafka Producer with idempotent delivery and acks=all
    * JSON schema validation against grievance-events Pydantic models
    * topic naming convention (event_type.vN)
    * dead-letter queue routing on publish failure
    * OpenTelemetry span around every produce()
"""
from __future__ import annotations


def publish(topic: str, payload: dict) -> None:  # type: ignore[type-arg]
    """Publish an event to Kafka. No-op in Module 1."""
    raise NotImplementedError("Implemented in Module 18 (Event handlers)")
