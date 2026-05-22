.PHONY: help install run shell migrate makemigrations test lint format typecheck clean docker-build docker-run worker beat

SETTINGS ?= grievance_core.settings.dev

help:
	@echo "grievance-core — developer commands"
	@echo ""
	@echo "  make install         Install dependencies via Poetry"
	@echo "  make run             Run the dev server (uvicorn)"
	@echo "  make shell           Django shell"
	@echo "  make migrate         Apply DB migrations"
	@echo "  make makemigrations  Create new migrations"
	@echo "  make test            Run pytest with coverage"
	@echo "  make lint            Ruff lint"
	@echo "  make format          Ruff format"
	@echo "  make typecheck       mypy"
	@echo "  make worker          Run a Celery worker"
	@echo "  make beat            Run Celery beat scheduler"
	@echo "  make docker-build    Build the production image"
	@echo "  make docker-run      Run the production image locally"
	@echo "  make clean           Remove caches and build artifacts"

install:
	poetry install

run:
	DJANGO_SETTINGS_MODULE=$(SETTINGS) poetry run uvicorn grievance_core.asgi:application --reload --host 0.0.0.0 --port 8000

shell:
	DJANGO_SETTINGS_MODULE=$(SETTINGS) poetry run python manage.py shell

migrate:
	DJANGO_SETTINGS_MODULE=$(SETTINGS) poetry run python manage.py migrate

makemigrations:
	DJANGO_SETTINGS_MODULE=$(SETTINGS) poetry run python manage.py makemigrations

test:
	poetry run pytest

lint:
	poetry run ruff check .

format:
	poetry run ruff format .
	poetry run ruff check --fix .

typecheck:
	poetry run mypy .

worker:
	DJANGO_SETTINGS_MODULE=$(SETTINGS) poetry run celery -A grievance_core worker --loglevel=info

beat:
	DJANGO_SETTINGS_MODULE=$(SETTINGS) poetry run celery -A grievance_core beat --loglevel=info

docker-build:
	docker build -t grievance-core:dev .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env grievance-core:dev

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist *.egg-info
