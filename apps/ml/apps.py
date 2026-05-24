"""apps/ml/apps.py — AppConfig for the ML inference app."""
from django.apps import AppConfig


class MlConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.ml"
    label = "ml"
    verbose_name = "ML Inference"
