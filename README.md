# Civic Grievance Intelligence Platform

> **TVMC · Thiruvananthapuram Municipal Corporation, Kerala, India**  
> A full-stack civic grievance platform with a multilingual AI/ML pipeline, officer command-center, and citizen portal.

---

## What Is This?

Citizens of Thiruvananthapuram submit civic complaints (potholes, water supply failure, sewage overflow, illegal construction, etc.) via a web portal. The platform:

1. **Automatically classifies** complaints using a 12-phase ML pipeline
2. **Routes** them to the correct municipal department
3. **Detects** duplicates, spam, and life-safety escalations
4. **Tracks SLA deadlines** and auto-escalates breaches
5. **Provides officers** with a command-center review interface
6. **Gives admins** a real-time analytics dashboard

The ML pipeline supports **English, Malayalam Unicode, and Manglish** (romanized Malayalam).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 5 · Django REST Framework · GeoDjango |
| Database | PostgreSQL + PostGIS (spatial ward maps) |
| AI/ML | Sentence Transformers (MiniLM) · TF-IDF · Rule engine |
| Auth | JWT via `djangorestframework-simplejwt` |
| Frontend | React 18 · TypeScript · Vite |
| Styling | Per-component CSS (no CSS framework) |
| Tests | pytest · pytest-django |

---

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 17 with PostGIS
- Node.js 18+ (for frontend)

### Backend

```bash
# 1. Clone and create virtual environment
git clone <repo-url>
cd grievance-core
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env          # Windows
# cp .env.example .env          # Linux/macOS
# Edit .env — set DJANGO_SECRET_KEY and DATABASE_URL

# 4. Apply migrations
python manage.py migrate

# 5. Seed TVMC ward data (101 wards)
python manage.py seed_tvmc_wards

# 6. Seed demo data (users, departments, grievances)
python manage.py seed_demo_data

# 7. Run development server
python manage.py runserver
```

Backend available at: `http://localhost:8000`

### Frontend

```bash
cd ../grievance-web
npm install
npm run dev
```

Frontend available at: `http://localhost:5173`

---

## Environment Variables

Create `.env` in the `grievance-core/` directory:

```ini
DJANGO_SECRET_KEY=your-50-char-random-secret-key
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=postgis://postgres:password@localhost:5432/grievance_db

# Email (optional — leave blank to use console backend in dev)
EMAIL_HOST=smtp.example.com
EMAIL_PORT=587
EMAIL_HOST_USER=alerts@tvmc.gov.in
EMAIL_HOST_PASSWORD=your-smtp-password

# CORS (production only)
CORS_ALLOWED_ORIGINS=https://yourdomain.com
```

**Windows-specific:** The base settings hardcode GDAL/GEOS library paths for PostgreSQL 17. Adjust `GDAL_LIBRARY_PATH` and `GEOS_LIBRARY_PATH` in `grievance_core/settings/base.py` if your PostgreSQL is installed elsewhere.

---

## Demo Users

After running `python manage.py seed_demo_data`:

| Role | Username | Password | Description |
|---|---|---|---|
| Citizen | `demo_citizen_rajan` | `Demo@1234` | Malayalam-preferred citizen |
| Citizen | `demo_citizen_priya` | `Demo@1234` | English-preferred citizen |
| Citizen | `demo_citizen_ahmed` | `Demo@1234` | English-preferred citizen |
| Ward Officer | `demo_officer_vishnu` | `Demo@1234` | Reviews ward queue |
| Ward Officer | `demo_officer_deepa` | `Demo@1234` | Reviews ward queue |
| Dept Officer | `demo_dept_officer_suresh` | `Demo@1234` | Department-scoped queue |
| Municipal Admin | `demo_admin_anitha` | `Demo@1234` | Full analytics access |

---

## Project Structure

```
grievance-core/              Django backend (modular monolith)
├── apps/
│   ├── users/               Custom User model (7 roles)
│   ├── wards/               101 TVMC wards (PostGIS geometry)
│   ├── departments/         7 municipal departments
│   ├── landmarks/           200+ TVMC landmark aliases
│   ├── grievances/          Core grievance lifecycle model
│   ├── attachments/         Image/file metadata
│   ├── workflows/           Immutable WorkflowEvent history
│   ├── slas/                SLA state machine + breach tracking
│   ├── audit/               Immutable audit log
│   ├── integrations/        External webhooks / notifications
│   └── ml/                  AI/ML pipeline (12-phase orchestrator)
│       ├── analyzer.py      Main analyze_complaint() entry point
│       ├── decision_engine.py  Phase 12 — final routing decision
│       ├── ml_inference.py  Tier chain (transformer → TF-IDF → rule)
│       ├── transformer_inference.py  MiniLM sentence embeddings
│       ├── image_analyzer.py  CLIP-based image evidence check
│       └── models/          Trained .joblib model artifacts
├── grievance_core/
│   └── settings/            base / dev / test / staging / production
└── tests/
    └── ml/
        └── test_ai_benchmark.py  60+ case accuracy benchmark

grievance-web/               React frontend (Vite + TypeScript)
├── src/
│   ├── pages/
│   │   ├── Grievance/       Citizen submit + track
│   │   ├── Officer/         Queue + two-panel review
│   │   ├── Admin/           Analytics dashboard
│   │   └── Dashboard/       Role-aware landing page
│   ├── components/
│   │   ├── Sidebar.tsx      Role-aware navigation
│   │   ├── StatusBadge.tsx  Unified 8-status chip
│   │   └── AIExplainabilityPanel.tsx  ML decision display
│   └── lib/
│       ├── api.ts           Axios + JWT interceptor
│       └── roles.ts         Role guards
```

---

## AI/ML Pipeline

`analyze_complaint(text)` in `apps/ml/analyzer.py` runs 12 phases:

```
Phase  1  Language detection         English / Malayalam / Manglish
Phase  2  Text normalization
Phase  3  Spam detection
Phase  4  Category classification    ML + rule fusion (9 civic categories)
Phase  5  Department routing         Category → department mapping
Phase  6  Landmark resolution        200+ TVMC aliases → ward linkage
Phase  7  Priority inference         Signal phrases + category base
Phase  8  Duplicate detection        Semantic cosine similarity
Phase  9  Routing confidence         Weighted score (category 40%, ward 20%…)
Phase 10  Review flags               spam_suspicion, duplicate_risk_high, etc.
Phase 11  Image evidence             CLIP vision analysis (if attachment present)
Phase 12  Decision engine            reject / escalate / review_required / auto_route
```

**9 supported categories:** `road_damage`, `waste_management`, `water_supply`, `street_light`, `drainage`, `tree_fall`, `illegal_construction`, `electrical_hazard`, `sewage_issue`

**Inference tier chain:** Transformer (MiniLM) → TF-IDF pipelines → Keyword rules

---

## Management Commands

```bash
# System health — PASS/WARN/FAIL report
python manage.py project_health_check
python manage.py project_health_check --json

# Demo data
python manage.py seed_demo_data
python manage.py seed_tvmc_wards

# SLA breach monitoring
python manage.py check_sla_breaches
python manage.py check_sla_breaches --dry-run
python manage.py check_sla_breaches --escalate

# ML verification
python manage.py check_ml
python manage.py check_vision
python manage.py audit_bias
python manage.py train_ml_models
python manage.py train_ml_models --no-transformer
```

---

## Running Tests

```bash
# Full test suite
pytest

# ML accuracy benchmark (60+ civic complaint cases)
pytest tests/ml/test_ai_benchmark.py -v

# Print AI benchmark summary
pytest tests/ml/test_ai_benchmark.py::test_benchmark_print_summary -v -s

# SLA tests
pytest tests/slas/ -v

# With coverage
pytest --cov=apps --cov-report=term-missing
```

**Benchmark accuracy thresholds** (production minimums):

| Metric | Threshold |
|---|---|
| Category accuracy | ≥ 72% |
| Priority accuracy | ≥ 55% |
| Language detection | ≥ 80% |
| Spam precision | ≥ 55% |
| Spam recall | ≥ 50% |

See [`AI_BENCHMARK_REPORT.md`](AI_BENCHMARK_REPORT.md) for full details.

---

## API Overview

Base URL: `http://localhost:8000/api/v1/`

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/api/token/` | POST | Public | Obtain JWT access + refresh tokens |
| `/api/token/refresh/` | POST | Public | Refresh access token |
| `/api/v1/grievances/` | GET/POST | IsAuthenticated | List/submit grievances |
| `/api/v1/grievances/{id}/` | GET/PATCH | IsAuthenticated | Detail / officer enrichment |
| `/api/v1/departments/` | GET | IsAuthenticated | Department list |
| `/api/v1/wards/` | GET | IsAuthenticated | Ward list |
| `/api/v1/slas/` | GET | Officers+ | SLA records |
| `/api/v1/workflows/` | GET | Officers+ | Workflow event history |

---

## Architecture Documentation

See [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) for:
- Full system diagram
- ML pipeline details
- Database schema
- Role model
- SLA + workflow lifecycle
- Deployment configuration

---

## User Roles

| Role | Portal Access |
|---|---|
| `citizen` | Submit, track own grievances |
| `ward_officer` | View ward-scoped queue, review, transition |
| `department_officer` | View department-scoped queue |
| `municipal_admin` | Full analytics dashboard, all grievances |
| `super_admin` | Platform-level administration |
| `field_verifier` | On-site evidence verification |
| `system_operator` | Automation and bulk operations |

---

## Grievance Lifecycle

```
submitted → enrichment_pending → triaged → assigned → in_progress → resolved → closed
                                                                   ↘ rejected
```

Each transition creates an immutable `WorkflowEvent` record with actor, reason, and timestamps.

---

## License

Internal project — Thiruvananthapuram Municipal Corporation.
