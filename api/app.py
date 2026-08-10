"""
app.py
------
FastAPI application for SMS spam detection.

Endpoints:
  GET  /health   — liveness check
  POST /predict  — classify a text message as spam or ham

Model artifacts are loaded once at startup from models/.
If the artifacts are missing the app will fail fast with a clear error.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the app works from any cwd
# ---------------------------------------------------------------------------
ROOT            = Path(__file__).resolve().parent.parent
MODEL_PATH      = ROOT / "models" / "model.pkl"
VECTORIZER_PATH = ROOT / "models" / "vectorizer.pkl"

# ---------------------------------------------------------------------------
# Model state (populated during startup)
# ---------------------------------------------------------------------------
_model      = None
_vectorizer = None


# ---------------------------------------------------------------------------
# Lifespan — load artifacts once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _vectorizer

    for path in (MODEL_PATH, VECTORIZER_PATH):
        if not path.exists():
            logger.error("Required artifact not found: %s", path)
            sys.exit(1)

    try:
        _model      = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
        logger.info("Vectorizer loaded from %s", VECTORIZER_PATH)
    except Exception as exc:
        logger.error("Failed to load model artifacts: %s", exc)
        sys.exit(1)

    yield  # application runs here

    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SMS Spam Detector",
    description="Classify SMS messages as spam or ham using a MultinomialNB model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="The SMS message text to classify.",
        examples=["Congratulations! You've won a FREE prize. Call now!"],
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace only")
        return v


class PredictResponse(BaseModel):
    spam: bool = Field(..., description="True if the message is classified as spam.")
    label: str = Field(..., description="Human-readable label: 'spam' or 'ham'.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> Any:
    """Liveness check — confirms the API is running and model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None and _vectorizer is not None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> Any:
    """
    Classify a text message as spam or ham.

    Send a JSON body: `{"text": "your message here"}`
    """
    if _model is None or _vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import pandas as pd
        vec       = _vectorizer.transform([request.text])
        feat_cols = _vectorizer.get_feature_names_out()
        vec_df    = pd.DataFrame(vec.toarray(), columns=feat_cols)
        pred      = int(_model.predict(vec_df)[0])
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    label = "spam" if pred == 1 else "ham"
    logger.info("Prediction — label=%s  input_length=%d", label, len(request.text))
    return PredictResponse(spam=bool(pred), label=label)
