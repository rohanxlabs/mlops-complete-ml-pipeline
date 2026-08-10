"""
train.py
--------
Loads processed.csv, trains a MultinomialNB classifier,
logs accuracy to MLflow, and saves model.pkl.

Input:  data/processed.csv
Output: models/model.pkl
        MLflow run logged to mlflow.db
"""

import logging
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    MODEL_PATH,
    PROCESSED_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train() -> None:
    logger.info("Starting model training")

    if not PROCESSED_PATH.exists():
        logger.error("processed.csv not found — run data_preprocessing.py first")
        sys.exit(1)

    try:
        df = pd.read_csv(PROCESSED_PATH)
    except Exception as exc:
        logger.error("Failed to read processed.csv: %s", exc)
        sys.exit(1)

    logger.info("Loaded feature matrix %s", df.shape)

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(
        "Train/test split — train: %d  test: %d  (test_size=%.0f%%)",
        len(X_train),
        len(X_test),
        TEST_SIZE * 100,
    )

    # MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        model = MultinomialNB()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model", "MultinomialNB")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("tfidf_max_features", X.shape[1] - 0)  # columns = features
        mlflow.log_metric("accuracy", acc)

        # Log model artifact
        mlflow.sklearn.log_model(model, name="model")
        logger.info("MLflow run complete — accuracy=%.4f", acc)

    # Save model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Training complete | Test accuracy: %.4f", acc)


if __name__ == "__main__":
    train()
