"""
evaluate.py
-----------
Evaluates the trained model on a held-out test set.
Uses the same train/test split as train.py (same random_state)
to ensure evaluation is done on data the model has never seen.

Input:  data/processed.csv
        models/model.pkl
Output: prints classification report to stdout
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

from config import MODEL_PATH, PROCESSED_PATH, RANDOM_STATE, ROOT, TEST_SIZE

METRICS_PATH = ROOT / "metrics.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate() -> None:
    logger.info("Starting model evaluation")

    for path in (PROCESSED_PATH, MODEL_PATH):
        if not path.exists():
            logger.error("Required file not found: %s", path)
            sys.exit(1)

    try:
        df = pd.read_csv(PROCESSED_PATH)
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.error("Failed to load artifacts: %s", exc)
        sys.exit(1)

    X = df.drop("label", axis=1)
    y = df["label"]

    # Reproduce the same split used during training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Evaluating on held-out test set (%d samples)", len(X_test))

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted")

    logger.info("Accuracy (test set): %.4f", acc)
    logger.info("F1 score (weighted, test set): %.4f", f1)

    print("\n=== Classification Report (held-out test set) ===")
    print(
        classification_report(
            y_test,
            preds,
            target_names=["ham", "spam"],
        )
    )

    # Write metrics.json for DVC metrics tracking
    metrics = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics written to %s", METRICS_PATH)


if __name__ == "__main__":
    evaluate()
