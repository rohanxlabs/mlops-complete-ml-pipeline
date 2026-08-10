"""
data_preprocessing.py
---------------------
Reads raw.csv, label-encodes the target, fits a TF-IDF vectorizer,
and writes processed.csv + models/vectorizer.pkl.

Input:  data/raw.csv
Output: data/processed.csv
        models/vectorizer.pkl
"""

import logging
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from config import (
    CLEAN_DATA_PATH,
    MODELS_DIR,
    PROCESSED_PATH,
    TFIDF_MAX_FEATURES,
    TFIDF_STOP_WORDS,
    VECTORIZER_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def preprocess() -> None:
    logger.info("Starting data preprocessing")

    if not CLEAN_DATA_PATH.exists():
        logger.error("raw.csv not found — run data_ingestion.py first")
        sys.exit(1)

    try:
        df = pd.read_csv(CLEAN_DATA_PATH)
    except Exception as exc:
        logger.error("Failed to read raw.csv: %s", exc)
        sys.exit(1)

    logger.info("Loaded %d rows from %s", len(df), CLEAN_DATA_PATH)

    # Label encode: ham=0, spam=1
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
    logger.info("Label encoding: %s", dict(zip(le.classes_, le.transform(le.classes_))))

    # TF-IDF vectorization
    logger.info(
        "Fitting TfidfVectorizer (max_features=%d, stop_words='%s')",
        TFIDF_MAX_FEATURES,
        TFIDF_STOP_WORDS,
    )
    vectorizer = TfidfVectorizer(
        stop_words=TFIDF_STOP_WORDS,
        max_features=TFIDF_MAX_FEATURES,
    )
    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    # Save vectorizer
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logger.info("Vectorizer saved to %s", VECTORIZER_PATH)

    # Save feature matrix with named columns (avoids sklearn feature-name warning)
    feature_names = vectorizer.get_feature_names_out()
    processed_df = pd.DataFrame(X.toarray(), columns=feature_names)
    processed_df["label"] = y.values

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_PATH, index=False)
    logger.info(
        "Preprocessing complete — feature matrix %s written to %s",
        processed_df.shape,
        PROCESSED_PATH,
    )


if __name__ == "__main__":
    preprocess()
