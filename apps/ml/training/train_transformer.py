"""apps/ml/training/train_transformer.py

Transformer-backbone training script for the TVMC civic grievance ML pipeline.

What this script does
---------------------
1. Expands corpus_data_v2.py seeds into 6 000+ samples via generate_corpus_v2.
2. Loads the frozen SentenceTransformer backbone
   (paraphrase-multilingual-MiniLM-L12-v2 -- 118 MB, CPU-friendly, 50+ languages).
3. Encodes ALL training samples in batches -> 384-dim embeddings.
4. Trains one LogisticRegression head per task on the frozen embeddings:
      category, priority, department, spam, language
5. Pre-encodes TVM ward/landmark names + compound aliases for location intelligence.
6. Saves two joblib artefacts:
      apps/ml/models/transformer_heads.joblib
      apps/ml/models/landmark_embeddings.joblib
7. (--eval flag) Cross-validates each head and prints accuracy / F1.

Usage
-----
# Full training + evaluation:
python -m apps.ml.training.train_transformer --eval

# Quick smoke test (no eval):
python -m apps.ml.training.train_transformer

# Custom target size:
python -m apps.ml.training.train_transformer --target 8000 --eval
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODELS_DIR   = PROJECT_ROOT / "apps" / "ml" / "models"

_HEADS_FILE    = MODELS_DIR / "transformer_heads.joblib"
_LANDMARK_FILE = MODELS_DIR / "landmark_embeddings.joblib"

_BACKBONE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def _check_deps() -> None:
    missing = []
    for pkg in ("sentence_transformers", "sklearn", "joblib", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {missing}")
        print("Install them with:")
        print("  pip install sentence-transformers scikit-learn joblib numpy")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_samples(target: int) -> list[tuple[str, str, str, str]]:
    """Return expanded training samples from corpus_data_v2 seeds."""
    # Ensure apps/ is importable
    sys.path.insert(0, str(PROJECT_ROOT))
    from apps.ml.training.generate_corpus_v2 import build_dataset  # noqa: PLC0415
    samples = build_dataset(target=target)
    print(f"  Loaded {len(samples)} training samples")
    return samples


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _encode_texts(
    model: Any,
    texts: list[str],
    batch_size: int = 64,
    desc: str = "Encoding",
) -> np.ndarray:
    """Encode a list of texts in batches, showing progress."""
    all_embs = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.append(embs)
        done = min(start + batch_size, n)
        print(f"  {desc}: {done}/{n}", end="\r", flush=True)
    print()
    return np.vstack(all_embs)


# ---------------------------------------------------------------------------
# Head training
# ---------------------------------------------------------------------------

def _train_head(
    X_train: np.ndarray,
    y_train: list[str],
    C: float = 5.0,
) -> tuple[Any, Any]:
    """Train LogisticRegression + LabelEncoder head.

    Returns (classifier, label_encoder).
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    clf = LogisticRegression(
        C=C,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, y_enc)
    return clf, le


def _evaluate_head(
    X: np.ndarray,
    y: list[str],
    clf: Any,
    le: Any,
    head_name: str,
    cv: int = 5,
) -> None:
    """Cross-validate and print accuracy + weighted F1."""
    from sklearn.model_selection import cross_validate  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

    le2 = LabelEncoder()
    y_enc = le2.fit_transform(y)
    cv_result = cross_validate(clf, X, y_enc, cv=cv,
                               scoring=["accuracy", "f1_weighted"],
                               n_jobs=-1)
    acc = cv_result["test_accuracy"].mean()
    f1  = cv_result["test_f1_weighted"].mean()
    print(f"  {head_name:<20}  acc={acc:.3f}  f1={f1:.3f}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_all(target: int = 6000, evaluate: bool = False) -> None:
    _check_deps()

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    import joblib  # noqa: PLC0415

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. Load corpus ----------------------------------------------------
    _banner("Step 1 / 5 -- Loading training corpus")
    samples = _load_samples(target)
    texts      = [s[0] for s in samples]
    categories = [s[1] for s in samples]
    priorities = [s[2] for s in samples]
    departments= [s[3] for s in samples]

    # Derive language labels from text heuristic (fallback for samples
    # that don't have an explicit language annotation in the corpus).
    # Simple heuristic: Unicode Malayalam block -> "ml"; Manglish keywords -> "manglish";
    # mixed detection; else "en".
    def _detect_lang(text: str) -> str:
        malayalam_chars = sum(1 for c in text if "ഀ" <= c <= "ൿ")
        if malayalam_chars > len(text) * 0.15:
            return "ml"
        manglish_tokens = {"aanu", "aayittu", "cheyyenam", "pottannu", "kittunnilla",
                           "vellam", "kuzhal", "maram", "oda", "mazha"}
        tokens = set(text.lower().split())
        if tokens & manglish_tokens:
            if any(c.isascii() for c in text):
                return "manglish"
        if any(not c.isascii() for c in text) and any(c.isascii() for c in text):
            return "mixed"
        return "en"

    languages = [_detect_lang(t) for t in texts]

    # Derive spam labels
    spam_labels = ["spam" if c == "spam" else "not_spam" for c in categories]

    # -- 2. Load backbone --------------------------------------------------
    _banner("Step 2 / 5 -- Loading transformer backbone")
    t0 = time.time()
    print(f"  Model : {_BACKBONE_MODEL}")
    backbone = SentenceTransformer(_BACKBONE_MODEL)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # -- 3. Encode corpus --------------------------------------------------
    _banner("Step 3 / 5 -- Encoding training corpus")
    t0 = time.time()
    X = _encode_texts(backbone, texts, batch_size=128, desc="Encoding corpus")
    print(f"  Encoded {len(texts)} samples -> shape {X.shape} in {time.time() - t0:.1f}s")

    # -- 4. Train classifier heads -----------------------------------------
    _banner("Step 4 / 5 -- Training classifier heads")

    tasks: list[tuple[str, list[str], float]] = [
        ("category",   categories,  5.0),
        ("priority",   priorities,  3.0),
        ("department", departments, 5.0),
        ("spam",       spam_labels, 3.0),
        ("language",   languages,   3.0),
    ]

    heads: dict[str, Any] = {}
    if evaluate:
        print(f"  {'Head':<20}  {'acc':>6}  {'f1':>6}")
        print(f"  {'-'*20}  {'-'*6}  {'-'*6}")

    for task_name, y, C in tasks:
        t0 = time.time()
        clf, le = _train_head(X, y, C=C)
        elapsed = time.time() - t0
        print(f"  Trained '{task_name}' head  ({len(set(y))} classes, {elapsed:.1f}s)")
        heads[f"{task_name}_head"] = clf
        heads[f"{task_name}_le"]   = le

        if evaluate:
            _evaluate_head(X, y, clf, le, task_name, cv=5)

    # -- 5a. Save transformer heads ----------------------------------------
    _banner("Step 5 / 5 -- Saving artefacts")
    joblib.dump(heads, _HEADS_FILE)
    print(f"  Saved transformer heads -> {_HEADS_FILE}")

    # -- 5b. Pre-encode landmark embeddings --------------------------------
    # Use TVM_LOCATIONS_EXTENDED (v2 single names + v3 compound aliases)
    # so find_ward_candidates() can match real complaint descriptions.
    from apps.ml.training.corpus_data_v3 import TVM_LOCATIONS_EXTENDED  # noqa: PLC0415
    print(f"  Encoding {len(TVM_LOCATIONS_EXTENDED)} TVM landmarks (incl. compound aliases)...")
    landmark_embs = backbone.encode(TVM_LOCATIONS_EXTENDED, normalize_embeddings=True)
    landmark_data = {
        "embeddings": landmark_embs,
        "names":      TVM_LOCATIONS_EXTENDED,
    }
    joblib.dump(landmark_data, _LANDMARK_FILE)
    print(f"  Saved landmark embeddings -> {_LANDMARK_FILE}")

    print("\n  Training complete.\n")
    print("  Artefact sizes:")
    for p in [_HEADS_FILE, _LANDMARK_FILE]:
        if p.exists():
            kb = p.stat().st_size / 1024
            print(f"    {p.name:<35}  {kb:,.0f} KB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train transformer-backbone classifier heads for TVMC grievance ML."
    )
    parser.add_argument("--target", type=int, default=6000,
                        help="Training corpus size target (default: 6000)")
    parser.add_argument("--eval", action="store_true",
                        help="Run 5-fold cross-validation and print metrics")
    args = parser.parse_args()
    train_all(target=args.target, evaluate=args.eval)


if __name__ == "__main__":
    _main()
