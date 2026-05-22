# grievance-core

Transactional core of the Civic Grievance Intelligence Platform.

This is the Django modular monolith from Phase 1 §10.1. It owns users,
wards, departments, grievances, attachments, workflows, SLAs, and the audit
write path. Compute-heavy concerns (NLP, vision, geo, dedup, routing) live in
separate FastAPI services and reach this service via the `/api/internal/`
namespace.

## Stack

| Layer            | Technology                                                  |
|------------------|-------------------------------------------------------------|
| Framework        | Django 5 + DRF + GeoDjango + DRF-GIS                        |
| Database         | PostgreSQL 16 + PostGIS 3.4                                 |
| Cache / broker   | Redis 7                                                     |
| Task queue       | Celery 5                                                    |
| Container server | gunicorn + uvicorn workers (ASGI)                           |
| Dependency mgmt  | Poetry 1.8                                                  |
| Tests            | pytest, pytest-django, factory_boy                          |
| Lint / format    | ruff                                                        |
| Type check       | mypy (strict)                                               |

## Local development

Prerequisites: Python 3.11, Poetry, PostgreSQL 16 with PostGIS, Redis 7.

```bash
cp .env.example .env
make install
make migrate
make run                           # http://localhost:8000
```

In a second terminal:

```bash
make worker                        # Celery worker
```

## Tests

```bash
make test                          # runs pytest with coverage
make lint                          # ruff
make typecheck                     # mypy
```

## Repository layout (this service)

```
grievance-core/
├── grievance_core/      Django project package
│   ├── settings/        base, dev, staging, production, test
│   ├── urls.py          root URL conf
│   ├── asgi.py / wsgi.py
│   ├── celery_app.py    Celery factory
│   └── health.py        K8s probes
├── apps/                Django apps (modular monolith)
│   ├── users/           custom User model
│   ├── wards/           ward and local body models
│   ├── departments/     department taxonomy
│   ├── landmarks/       POI references
│   ├── grievances/      grievance aggregate
│   ├── attachments/     image/file attachments
│   ├── workflows/       status transitions, escalations
│   ├── slas/            SLA tracking
│   ├── audit/           immutable audit log
│   └── integrations/    outbound clients to nlp/vision/geo/dedup
├── api/
│   ├── v1/              public API consumed by frontends
│   └── internal/        service-to-service API
├── infra/               cross-cutting infrastructure (otel, kafka, cache)
├── locale/              en + ml translation catalogs
└── tests/
```

## Module status

This service is being built one module at a time. Current status:

- [x] Module 1  — Core backend skeleton
- [ ] Module 2  — Configuration management
- [ ] Module 3  — Environment management
- [ ] Module 4  — Logging
- [ ] Module 5  — Error handling
- [ ] Module 6  — Auth (Keycloak OIDC)
- [ ] Module 7  — RBAC
- [ ] Module 8  — User management
- [ ] Module 9  — Complaint APIs
- [ ] Module 10 — Officer APIs
- [ ] Module 11 — Admin APIs
- [ ] Module 12 — Audit APIs
- [ ] Module 13 — AI pipeline APIs *(separate service: ingestion)*
- [ ] Module 14 — GIS APIs *(separate service: geo)*
- [ ] Module 15 — Duplicate detection APIs *(separate service: dedup)*
- [ ] Module 16 — Notifications
- [ ] Module 17 — Queue workers
- [ ] Module 18 — Event handlers
- [ ] Module 19 — Search indexing
- [ ] Module 20 — Monitoring hooks

See `ARCHITECTURE.md` at the repo root for the Phase 1 design.
