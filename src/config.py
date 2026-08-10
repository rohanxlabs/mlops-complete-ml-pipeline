"""
Central configuration — all paths derived from the project root.
Import this module instead of hard-coding paths anywhere in the pipeline.
"""

from pathlib import Path

# Project root = parent of this file's directory (src/)
ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
RAW_DATA_PATH   = DATA_DIR / "spam.csv"
CLEAN_DATA_PATH = DATA_DIR / "raw.csv"
PROCESSED_PATH  = DATA_DIR / "processed.csv"

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR      = ROOT / "models"
MODEL_PATH      = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT   = "spam-detection"

# ── Pre-processing hyper-parameters ───────────────────────────────────────────
TFIDF_MAX_FEATURES = 3000
TFIDF_STOP_WORDS   = "english"

# ── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
