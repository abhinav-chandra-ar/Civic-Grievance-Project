# syntax=docker/dockerfile:1.7
#
# grievance-core — production container image
#
# Multi-stage:
#   1. builder   — installs Poetry, resolves and exports dependencies
#   2. runtime   — slim image, non-root user, only runtime deps
#
# GeoDjango requires GDAL, GEOS, and PROJ at runtime.

ARG PYTHON_VERSION=3.11.9

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /build

COPY pyproject.toml poetry.lock* ./

# Install only main dependencies into a virtualenv at /build/.venv
RUN poetry install --no-root --only main

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    DJANGO_SETTINGS_MODULE=grievance_core.settings.production

# Runtime libraries only (no -dev packages, no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        gdal-bin \
        libgdal32 \
        libgeos-c1v5 \
        libproj25 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --home-dir /app --shell /bin/bash app

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder --chown=app:app /build/.venv /app/.venv

# Copy application code
COPY --chown=app:app . /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live || exit 1

# gunicorn with uvicorn workers — ASGI for async views and channels readiness
CMD ["gunicorn", "grievance_core.asgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "60", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
