"""
test_pipeline.py
----------------
Tests for each pipeline stage: ingestion, validation,
preprocessing, training, and prediction consistency.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

# Ensure src/ is on the path for config imports
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    CLEAN_DATA_PATH,
    MODEL_PATH,
    PROCESSED_PATH,
    RAW_DATA_PATH,
    VECTORIZER_PATH,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Load raw.csv — requires data_ingestion to have run."""
    assert CLEAN_DATA_PATH.exists(), (
        f"raw.csv not found at {CLEAN_DATA_PATH}. Run data_ingestion.py first."
    )
    return pd.read_csv(CLEAN_DATA_PATH)


@pytest.fixture(scope="module")
def processed_df():
    """Load processed.csv — requires data_preprocessing to have run."""
    assert PROCESSED_PATH.exists(), (
        f"processed.csv not found at {PROCESSED_PATH}. Run data_preprocessing.py first."
    )
    return pd.read_csv(PROCESSED_PATH)


@pytest.fixture(scope="module")
def model():
    """Load model.pkl — requires train.py to have run."""
    assert MODEL_PATH.exists(), (
        f"model.pkl not found at {MODEL_PATH}. Run train.py first."
    )
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def vectorizer():
    """Load vectorizer.pkl — requires data_preprocessing.py to have run."""
    assert VECTORIZER_PATH.exists(), (
        f"vectorizer.pkl not found at {VECTORIZER_PATH}. Run data_preprocessing.py first."
    )
    return joblib.load(VECTORIZER_PATH)


# ── Ingestion tests ───────────────────────────────────────────────────────────

class TestIngestion:
    def test_source_file_exists(self):
        assert RAW_DATA_PATH.exists(), f"Source dataset missing: {RAW_DATA_PATH}"

    def test_raw_csv_has_correct_columns(self, raw_df):
        assert "label" in raw_df.columns
        assert "message" in raw_df.columns

    def test_raw_csv_has_rows(self, raw_df):
        assert len(raw_df) > 100, "raw.csv has too few rows"

    def test_raw_csv_no_nulls(self, raw_df):
        assert raw_df["label"].isnull().sum() == 0
        assert raw_df["message"].isnull().sum() == 0

    def test_raw_csv_label_values(self, raw_df):
        assert set(raw_df["label"].unique()) == {"ham", "spam"}


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_processed_has_label_column(self, processed_df):
        assert "label" in processed_df.columns

    def test_processed_has_feature_columns(self, processed_df):
        feature_cols = [c for c in processed_df.columns if c != "label"]
        assert len(feature_cols) > 0, "No feature columns found in processed.csv"

    def test_processed_label_is_binary(self, processed_df):
        assert set(processed_df["label"].unique()).issubset({0, 1})

    def test_vectorizer_artifact_exists(self):
        assert VECTORIZER_PATH.exists()

    def test_vectorizer_can_transform(self, vectorizer):
        result = vectorizer.transform(["Free prize winner click now"])
        assert result.shape[0] == 1


# ── Training / model tests ────────────────────────────────────────────────────

class TestModel:
    def test_model_artifact_exists(self):
        assert MODEL_PATH.exists()

    def test_model_can_predict(self, model, vectorizer):
        import pandas as pd
        vec    = vectorizer.transform(["Win a free iPhone now"])
        df     = pd.DataFrame(vec.toarray(), columns=vectorizer.get_feature_names_out())
        pred   = model.predict(df)
        assert pred.shape == (1,)
        assert pred[0] in {0, 1}

    def test_spam_prediction(self, model, vectorizer):
        """Obvious spam messages should be classified as spam (label=1)."""
        import pandas as pd
        spam_texts = [
            "WINNER!! Claim your FREE prize now. Call 0800-123-456",
            "Congratulations! You've been selected for a cash reward.",
        ]
        for text in spam_texts:
            vec  = vectorizer.transform([text])
            df   = pd.DataFrame(vec.toarray(), columns=vectorizer.get_feature_names_out())
            pred = int(model.predict(df)[0])
            assert pred == 1, f"Expected spam (1) for: {text!r}"

    def test_ham_prediction(self, model, vectorizer):
        """Ordinary messages should be classified as ham (label=0)."""
        import pandas as pd
        ham_texts = [
            "Hey, are we still meeting at 3pm today?",
            "Can you pick up some milk on your way home?",
        ]
        for text in ham_texts:
            vec  = vectorizer.transform([text])
            df   = pd.DataFrame(vec.toarray(), columns=vectorizer.get_feature_names_out())
            pred = int(model.predict(df)[0])
            assert pred == 0, f"Expected ham (0) for: {text!r}"

    def test_model_accuracy_above_threshold(self, model, processed_df):
        """Test-set accuracy must be above 95% (conservative threshold)."""
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from config import RANDOM_STATE, TEST_SIZE

        X = processed_df.drop("label", axis=1)
        y = processed_df["label"]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        assert acc >= 0.95, f"Model accuracy {acc:.4f} is below threshold 0.95"
