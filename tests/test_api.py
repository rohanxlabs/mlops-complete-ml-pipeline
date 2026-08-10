"""
test_api.py
-----------
Tests for the FastAPI spam detection endpoint.
Uses TestClient so no running server is needed.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure src/ is importable (needed for api/app.py → config.py)
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from api.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Return a synchronous TestClient for the app."""
    with TestClient(app) as c:
        yield c


# ── Health ─────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


# ── Valid predictions ──────────────────────────────────────────────────────────

class TestPredictValid:
    def test_spam_message(self, client):
        resp = client.post("/predict", json={"text": "WINNER! Claim your FREE prize now. Call 0800123456"})
        assert resp.status_code == 200
        body = resp.json()
        assert "spam"  in body
        assert "label" in body
        assert body["spam"]  is True
        assert body["label"] == "spam"

    def test_ham_message(self, client):
        resp = client.post("/predict", json={"text": "Hey, are we still meeting at 3pm today?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["spam"]  is False
        assert body["label"] == "ham"

    def test_response_schema(self, client):
        resp = client.post("/predict", json={"text": "Hello there"})
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["spam"],  bool)
        assert isinstance(body["label"], str)
        assert body["label"] in {"spam", "ham"}


# ── Invalid inputs ─────────────────────────────────────────────────────────────

class TestPredictInvalidInput:
    def test_missing_text_field(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_empty_string(self, client):
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_whitespace_only(self, client):
        resp = client.post("/predict", json={"text": "   "})
        assert resp.status_code == 422

    def test_wrong_type(self, client):
        resp = client.post("/predict", json={"text": 12345})
        # FastAPI coerces int to str for string fields — prediction should still work
        assert resp.status_code in {200, 422}

    def test_malformed_json(self, client):
        resp = client.post(
            "/predict",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_text_too_long(self, client):
        resp = client.post("/predict", json={"text": "x" * 10_001})
        assert resp.status_code == 422
