"""apps/ml/management/commands/train_ml_models.py

Django management command: train and save all ML models for the civic
grievance intelligence pipeline.

Training pipeline
-----------------
1. Transformer tier  (preferred — paraphrase-multilingual-MiniLM-L12-v2)
   Trains LogisticRegression heads on 384-dim sentence embeddings.
   Saves: apps/ml/models/transformer_heads.joblib
          apps/ml/models/landmark_embeddings.joblib

2. TF-IDF tier       (fallback — char+word n-gram LogisticRegression)
   Trains classical TF-IDF pipelines on augmented corpus_data_v1 seeds.
   Saves: apps/ml/models/{category,priority,department,spam,language}_pipeline.joblib
          apps/ml/models/duplicate_vectorizer.joblib
          apps/ml/models/label_encoders.joblib

Usage
-----
    python manage.py train_ml_models
    python manage.py train_ml_models --eval
    python manage.py train_ml_models --no-transformer    # TF-IDF only
    python manage.py train_ml_models --no-tfidf          # transformer only
    python manage.py train_ml_models --factor 6 --eval
"""
from __future__ import annotations

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = (
        "Train and save all civic grievance ML models to apps/ml/models/. "
        "Trains transformer-backbone heads first, then TF-IDF fallback models."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--eval",
            action="store_true",
            default=False,
            help="Run cross-validation and print evaluation metrics",
        )
        parser.add_argument(
            "--factor",
            type=int,
            default=4,
            help="TF-IDF corpus augmentation factor (default 4)",
        )
        parser.add_argument(
            "--target",
            type=int,
            default=6000,
            help="Transformer corpus target size (default 6000)",
        )
        parser.add_argument(
            "--no-transformer",
            action="store_true",
            default=False,
            help="Skip transformer tier training",
        )
        parser.add_argument(
            "--no-tfidf",
            action="store_true",
            default=False,
            help="Skip TF-IDF tier training",
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Starting ML model training …\n"))

        # ── Transformer tier ──────────────────────────────────────────────
        if not options["no_transformer"]:
            self.stdout.write(self.style.NOTICE(
                "[ 1/2 ] Training transformer-backbone heads …"
            ))
            try:
                from apps.ml.training.train_transformer import train_all as train_transformer  # noqa: PLC0415
                train_transformer(
                    target=options["target"],
                    evaluate=options["eval"],
                )
                self.stdout.write(self.style.SUCCESS(
                    "  ✓ Transformer heads saved to apps/ml/models/"
                ))
            except ImportError as exc:
                self.stderr.write(self.style.WARNING(
                    f"  ⚠ Transformer training skipped: {exc}\n"
                    "    Install sentence-transformers: pip install sentence-transformers"
                ))
            except Exception as exc:  # noqa: BLE001
                self.stderr.write(self.style.ERROR(
                    f"  ✗ Transformer training failed: {exc}"
                ))
        else:
            self.stdout.write("  (transformer tier skipped via --no-transformer)")

        # ── TF-IDF tier ───────────────────────────────────────────────────
        if not options["no_tfidf"]:
            self.stdout.write(self.style.NOTICE(
                "\n[ 2/2 ] Training TF-IDF fallback models …"
            ))
            try:
                from apps.ml.training.train_models import train_all as train_tfidf  # noqa: PLC0415
                train_tfidf(
                    evaluate=options["eval"],
                    augment_factor=options["factor"],
                    save=True,
                )
                self.stdout.write(self.style.SUCCESS(
                    "  ✓ TF-IDF models saved to apps/ml/models/"
                ))
            except ImportError as exc:
                self.stderr.write(self.style.ERROR(
                    f"  ✗ Cannot import TF-IDF training module: {exc}\n"
                    "    Make sure scikit-learn, scipy, joblib are installed."
                ))
            except Exception as exc:  # noqa: BLE001
                self.stderr.write(self.style.ERROR(
                    f"  ✗ TF-IDF training failed: {exc}"
                ))
        else:
            self.stdout.write("  (TF-IDF tier skipped via --no-tfidf)")

        self.stdout.write(self.style.SUCCESS(
            "\n✓ Training complete.  Models loaded lazily on next request."
        ))
