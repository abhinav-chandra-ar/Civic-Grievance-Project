# Project Architecture — TVMC Civic Grievance Platform

> **Stack:** Django 5 · PostgreSQL / PostGIS · React 18 + TypeScript + Vite  
> **Domain:** Thiruvananthapuram Municipal Corporation (TVMC), Kerala, India  
> **Scope:** Citizen grievance submission → AI enrichment → officer review → resolution

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Repository Structure](#repository-structure)
3. [Backend Django Applications](#backend-django-applications)
4. [AI/ML Pipeline](#aiml-pipeline)
5. [API Layer](#api-layer)
6. [Frontend Architecture](#frontend-architecture)
7. [Authentication & Roles](#authentication--roles)
8. [SLA System](#sla-system)
9. [Workflow System](#workflow-system)
10. [Database Schema](#database-schema)
11. [Management Commands](#management-commands)
12. [Configuration & Deployment](#configuration--deployment)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Citizens / Officers / Admins                │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTPS
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│   React SPA  (Vite · TypeScript)                                │
│   Role-aware routing · JWT auth · Axios                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │ REST/JSON
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│   Django REST Framework  (DRF)                                  │
│   JWT via SimpleJWT · CORS via django-cors-headers              │
│   Permissions: IsAuthenticated + role-gated views               │
└──────────┬──────────────────────────────────────────────────────┘
           │
    ┌──────▼──────┐     ┌──────────────────────────────────────┐
    │  PostgreSQL │     │  ML Inference Stack                  │
    │  + PostGIS  │     │  Tier 1: Transformer (MiniLM)        │
    │  (geospatial│     │  Tier 2: TF-IDF / LogisticRegression │
    │   ward maps)│     │  Tier 3: Rule engine (always on)     │
    └─────────────┘     └──────────────────────────────────────┘
```

---

## Repository Structure

```
grievance-core/                    # Django backend
├── grievance_core/
│   ├── settings/
│   │   ├── base.py                # Shared settings (all environments)
│   │   ├── dev.py                 # DEBUG=True, console email
│   │   ├── test.py                # Minimal test settings
│   │   ├── staging.py
│   │   └── production.py
│   ├── urls.py
│   └── wsgi.py / asgi.py
├── apps/
│   ├── users/                     # Custom AbstractUser + roles
│   ├── wards/                     # TVMC 101 wards (PostGIS geometry)
│   ├── departments/               # 7+ municipal departments
│   ├── landmarks/                 # 200+ TVMC landmark aliases
│   ├── grievances/                # Core grievance model + lifecycle
│   ├── attachments/               # File attachment metadata
│   ├── workflows/                 # WorkflowEvent history
│   ├── slas/                      # SLA state machine
│   ├── audit/                     # Immutable audit log
│   ├── integrations/              # External hooks / webhooks
│   └── ml/                        # AI/ML pipeline (see below)
└── tests/
    ├── ml/
    │   ├── test_ai_benchmark.py   # 60+ case accuracy benchmark
    │   └── ...
    └── ...

grievance-web/                     # React frontend
├── src/
│   ├── components/                # Shared components
│   │   ├── Sidebar.tsx/css
│   │   ├── StatusBadge.tsx/css
│   │   ├── AIExplainabilityPanel.tsx/css
│   │   └── Pagination.tsx/css
│   ├── pages/
│   │   ├── Dashboard/
│   │   ├── Grievance/             # Citizen submit + track
│   │   ├── Officer/               # Queue + review
│   │   ├── Admin/                 # Dashboard + analytics
│   │   └── Profile/
│   ├── lib/
│   │   ├── api.ts                 # Axios instance
│   │   ├── auth.ts                # JWT management
│   │   └── roles.ts               # isCitizenRole / isOperatorRole / isAdminRole
│   └── App.tsx                    # React Router + auth guard
└── vite.config.ts
```

---

## Backend Django Applications

### `apps/users`
Custom user model extending `AbstractUser`. Seven roles drive all permission logic.

| Role | Code | Description |
|---|---|---|
| Citizen | `citizen` | Submits and tracks grievances |
| Ward Officer | `ward_officer` | Reviews ward-scoped queue |
| Department Officer | `department_officer` | Reviews department-scoped queue |
| Municipal Admin | `municipal_admin` | Full analytics + governance |
| Super Admin | `super_admin` | Platform-level access |
| Field Verifier | `field_verifier` | On-site verification |
| System Operator | `system_operator` | Automation & bulk ops |

Fields: `role`, `phone_number` (E.164), `preferred_language` (en/ml), `assigned_ward` FK, `assigned_department` FK.

---

### `apps/wards`
101 TVMC administrative wards stored with PostGIS geometry. Seeded via `seed_tvmc_wards` command from TVMC GIS data.

---

### `apps/departments`
Municipal departments with:
- `code` — lowercase slug (e.g., `roads_and_drainage`)
- `name` + `translated_names` (JSON, keyed by language code)
- `handled_categories` — JSONB list of category codes for GIN-indexed routing queries
- `is_active` flag

---

### `apps/grievances`
Core domain model. Key fields:

| Field | Type | Notes |
|---|---|---|
| `tracking_code` | `GRV-YYYY-NNNNNN` | Unique citizen-facing identifier |
| `raw_text` | TextField | Original submission |
| `category_code` | CharField | ML-assigned category |
| `department` | FK → Department | AI-routed department |
| `ward` | FK → Ward | Detected ward |
| `priority` | choice | low / medium / high / urgent / critical |
| `status` | choice | 8-state lifecycle (see below) |
| `possible_duplicate_of` | self-FK | Duplicate detection link |

**Grievance status lifecycle:**
```
submitted → enrichment_pending → triaged → assigned → in_progress → resolved → closed
                                                                  ↘ rejected
```

---

### `apps/workflows`
Immutable `WorkflowEvent` records every status transition.

Transition types: `status_change`, `assignment`, `reassignment`, `escalation`, `resolution`, `rejection`, `closure`, `comment`

Each event records: actor, assignee, previous_status, new_status, transition_reason, remarks, timestamps.

---

### `apps/slas`
One `SLA` record per grievance.

| SLA Status | Meaning |
|---|---|
| `active` | Deadline running |
| `breached` | Deadline passed |
| `paused` | Clock held |
| `satisfied` | Resolved within deadline |
| `cancelled` | Grievance rejected/withdrawn |

Breach types: `none`, `response`, `resolution`, `both`

SLA deadlines by priority:

| Priority | Response | Resolution |
|---|---|---|
| `critical` | 4 hours | 12 hours |
| `urgent` | 24 hours | 1 day |
| `high` | 24 hours | 3 days |
| `medium` | 24 hours | 5 days |
| `low` | 24 hours | 7 days |

---

## AI/ML Pipeline

`analyze_complaint(text, recent_texts=None)` in `apps/ml/analyzer.py`

### 12-Phase Orchestration

```
Phase  1 — Language detection         (EN / ML / Manglish)
Phase  2 — Text normalization         (unicode NFD, whitespace, noise strip)
Phase  3 — Spam detection             (ML score, rule heuristics)
Phase  4 — Category classification    (ML + rule fusion via _fuse_category)
Phase  5 — Department routing         (_CATEGORY_TO_DEPT map, 10 entries)
Phase  6 — Landmark resolution        (200+ TVMC aliases → ward linkage)
Phase  7 — Priority inference         (signal phrases + category base priority)
Phase  8 — Duplicate detection        (semantic similarity vs recent_texts)
Phase  9 — Overall routing confidence (weighted: category 40%, ward 20%, language 15%, dept 15%, base 10%)
Phase 10 — Review flags               (spam_suspicion, no_category_detected, duplicate_risk_high, etc.)
Phase 11 — Image evidence analysis    (CLIP-based vision check if attachment present)
Phase 12 — Final decision engine      (reject / escalate / review_required / auto_route)
```

### Inference Tier Chain

```python
Tier 1 — transformer     paraphrase-multilingual-MiniLM-L12-v2
                         + per-task LogisticRegression heads
                         (best accuracy, semantic duplicate detection)

Tier 2 — tfidf           char + word n-gram TF-IDF pipelines
                         (joblib .joblib model files in apps/ml/models/)
                         (no neural backbone; good multilingual char coverage)

Tier 3 — rule            keyword + regex rules in _ISSUE_KEYWORDS
                         (always available, no training required)
```

Each function tries Tier 1 → Tier 2 → Tier 3 (raises `ModelUnavailable` only if all three fail).

### Category → Department Routing

```python
_CATEGORY_TO_DEPT = {
    "road_damage":           "roads_and_drainage",
    "waste_management":      "sanitation",
    "water_supply":          "water_authority",
    "street_light":          "street_lighting",
    "drainage":              "roads_and_drainage",
    "tree_fall":             "parks_and_environment",
    "illegal_construction":  "building_permit_office",
    "electrical_hazard":     "electrical_engineering",
    "sewage_issue":          "roads_and_drainage",
}
```

### ML-Rule Fusion

```python
_ML_PRIMARY_THRESHOLD = 0.55   # ML wins outright above this
_ML_BLEND_THRESHOLD   = 0.30   # blend zone: ML + rule averaged
                                # below 0.30: rule engine wins
```

### Decision Engine (Phase 12)

```python
# Escalate immediately
_LIFE_SAFETY_CATEGORIES = {"electrical_hazard", "tree_fall", "sewage_issue"}

# Hard-block to human review
_HARD_BLOCKING_REVIEW_FLAGS = {
    "spam_suspicion", "no_category_detected",
    "duplicate_risk_high", "image_contradicts_complaint", "image_invalid",
}

# Routing confidence weights
category: 40%  ward: 20%  language: 15%  dept: 15%  base: 10%
```

Decision outcomes: `reject` | `escalate` | `review_required` | `auto_route`

---

## API Layer

Base URL: `/api/v1/`

| Resource | Endpoint prefix | Auth | Notes |
|---|---|---|---|
| Auth (JWT) | `/api/token/` | Public | SimpleJWT access + refresh |
| Users / Profile | `/api/v1/users/` | IsAuthenticated | Own profile only for citizens |
| Grievances | `/api/v1/grievances/` | IsAuthenticated | Citizens see own; officers see scoped queue |
| Departments | `/api/v1/departments/` | IsAuthenticated | Read-only for citizens |
| Wards | `/api/v1/wards/` | IsAuthenticated | Read-only |
| SLAs | `/api/v1/slas/` | Officers + Admins | |
| Workflows | `/api/v1/workflows/` | Officers + Admins | |
| ML Analyze | `/api/v1/ml/analyze/` | Officers + Admins | Raw analyze_complaint proxy |

**Pagination:** 20 items per page (DRF `PageNumberPagination`).  
**Filtering:** `django-filters` on `status`, `priority`, `category_code`, `department`, `ward`, `submitted_at` range.

---

## Frontend Architecture

Single-page React application at `grievance-web/`.

### Role-Aware Routing

```typescript
// lib/roles.ts
isCitizenRole(role)   // citizen
isOperatorRole(role)  // ward_officer | department_officer | field_verifier | system_operator
isAdminRole(role)     // municipal_admin | super_admin
```

Routes are gated in `App.tsx`: citizens see submission/tracking flows; officers see the queue + review; admins see the analytics dashboard.

### Visual Role Identity

| Role | Sidebar BG | Welcome Banner | Active Nav Link |
|---|---|---|---|
| Citizen | `#1e3a5f` (civic navy) | Blue gradient | Civic blue |
| Officer | `#1e293b` (slate) | Slate gradient | Amber |
| Admin | `#0f172a` (deep slate) | Deep slate gradient | Red accent |

### Key Components

| Component | Purpose |
|---|---|
| `StatusBadge` | Unified 8-status chip (submitted → closed) |
| `AIExplainabilityPanel` | Renders ML decision, confidence bar, signals, flags |
| `Sidebar` | Role-aware nav with active link indicators |
| `Pagination` | Universal pagination with ellipsis |

### State Management

No external state library. Data flows via:
- React hooks (`useState`, `useEffect`)
- `lib/api.ts` (Axios with JWT interceptor)
- Context for auth state

---

## Authentication & Roles

JWT-based auth via `djangorestframework-simplejwt`.

```
POST /api/token/           → { access, refresh }
POST /api/token/refresh/   → { access }
```

Access token attached as `Authorization: Bearer <token>` on every request.

Role enforcement:
- **Django side:** `IsAuthenticated` + custom permission classes checking `request.user.role`
- **React side:** `isCitizenRole / isOperatorRole / isAdminRole` guards on route render

---

## Management Commands

| Command | App | Purpose |
|---|---|---|
| `seed_tvmc_wards` | `apps.wards` | Load 101 TVMC ward boundaries from GIS data |
| `import_tvmc_wards` | `apps.wards` | Import from shapefile / JSON |
| `seed_demo_data` | `apps.grievances` | Seed demo users, depts, grievances, SLAs |
| `check_sla_breaches` | `apps.slas` | Scan + mark breached SLAs; optional `--escalate` |
| `check_ml` | `apps.ml` | Verify transformer tier is live |
| `check_vision` | `apps.ml` | Verify CLIP image analysis |
| `audit_bias` | `apps.ml` | Check category + priority distribution |
| `train_ml_models` | `apps.ml` | Train TF-IDF models; `--no-transformer` flag |
| `project_health_check` | `apps.ml` | Full PASS/WARN/FAIL system health report |

---

## Configuration & Deployment

### Environment Variables (`.env`)

```ini
DJANGO_SECRET_KEY=<50-char random>
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=yourdomain.com
DATABASE_URL=postgis://user:pass@localhost:5432/grievance_db
CORS_ALLOWED_ORIGINS=https://yourdomain.com
EMAIL_HOST=smtp.example.com
EMAIL_PORT=587
EMAIL_HOST_USER=alerts@tvmc.gov.in
EMAIL_HOST_PASSWORD=<password>
```

### Settings Hierarchy

```
base.py     ← imported by all
├── dev.py          DJANGO_SETTINGS_MODULE=grievance_core.settings.dev
├── test.py         DJANGO_SETTINGS_MODULE=grievance_core.settings.test
├── staging.py
└── production.py
```

### PostGIS / GDAL (Windows)

```python
# base.py (Windows dev path)
GDAL_LIBRARY_PATH = r"C:\Program Files\PostgreSQL\17\bin\libgdal-35.dll"
GEOS_LIBRARY_PATH = r"C:\Program Files\PostgreSQL\17\bin\libgeos_c-1.dll"
```

### Quick Start

```bash
# Backend
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_tvmc_wards
python manage.py seed_demo_data
python manage.py runserver

# Frontend
cd ../grievance-web
npm install && npm run dev

# Health check
python manage.py project_health_check
```

---

## Sequence: Grievance Submission to Resolution

```
Citizen submits complaint (raw_text)
    │
    ▼
POST /api/v1/grievances/
    │  status = submitted
    │  tracking_code = GRV-2024-XXXXXX (generated)
    │
    ▼
AI enrichment task triggered (Celery or sync)
    │  analyze_complaint(raw_text)
    │  → category_code, department, ward, priority
    │  → review_flags, routing_confidence, decision
    │  status = enrichment_pending → triaged / review_required
    │
    ▼
Officer picks up from queue
    │  OfficerQueuePage — priority-tinted rows
    │  Filters: status, priority, breach status
    │
    ▼
Officer reviews on OfficerGrievanceReviewPage
    │  Left: grievance details, attachments, AI explainability panel
    │  Right: enrich form (update fields) + transition form (change status)
    │
    ▼
WorkflowEvent recorded (immutable)
SLA updated (satisfied or escalated)
    │
    ▼
Resolution / Closure
    └── Citizen notified (email)
```

---

*Document reflects codebase state as of 2026-05-24.*
