"""FastAPI contract tests (Phase 5): auth, health, tickers, predict error shapes."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Ensure project root on path (matches api/main.py)
os.environ.setdefault("API_KEY", "test-api-key-for-pytest")

from api.main import app  # noqa: E402


@pytest.fixture
def client():
    """Use a stub bundle when the real joblib model cannot be loaded (e.g. no xgboost)."""
    with TestClient(app) as c:
        if app.state.model_bundle is None:
            app.state.model_bundle = {"stub": True}
        yield c


def test_health_ok(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "v1"


def test_tickers_list(client: TestClient):
    r = client.get("/api/tickers")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert "AAPL" in body


def test_predict_unauthorized_without_key(client: TestClient):
    r = client.post("/api/predict", json={"ticker": "AAPL"})
    assert r.status_code == 401


def test_predict_unauthorized_wrong_key(client: TestClient):
    r = client.post(
        "/api/predict",
        json={"ticker": "AAPL"},
        headers={"x-api-key": "wrong"},
    )
    assert r.status_code == 401


def test_predict_404_unknown_ticker(client: TestClient):
    r = client.post(
        "/api/predict",
        json={"ticker": "ZZZZ"},
        headers={"x-api-key": os.environ["API_KEY"]},
    )
    assert r.status_code == 404
    assert r.json() == {"error": "ticker_not_supported"}


def test_predict_422_insufficient_history(monkeypatch, client: TestClient):
    if app.state.model_bundle is None:
        pytest.skip("Model bundle not loaded")

    from src.errors import InsufficientHistoryError

    def _raise(*_a, **_k):
        raise InsufficientHistoryError("test")

    monkeypatch.setattr("src.predict.predict_for_ticker", _raise)
    r = client.post(
        "/api/predict",
        json={"ticker": "AAPL"},
        headers={"x-api-key": os.environ["API_KEY"]},
    )
    assert r.status_code == 422
    assert r.json() == {"error": "insufficient_history"}


def test_predict_200_mocked(monkeypatch, client: TestClient):
    if app.state.model_bundle is None:
        pytest.skip("Model bundle not loaded")

    def _fake_predict(ticker: str, **kwargs):
        return {
            "ticker": ticker,
            "prediction": "BEAT",
            "confidence": 0.71,
            "probabilities": {"BEAT": 0.71, "IN_LINE": 0.19, "MISS": 0.10},
            "top_features": [
                {"feature": "beat_rate_8q", "value": 0.875, "direction": "positive"},
                {"feature": "sentiment_score", "value": 0.63, "direction": "positive"},
                {"feature": "momentum_60d", "value": 0.04, "direction": "positive"},
            ],
            "last_quarters": [
                {
                    "quarter": "2024-Q3",
                    "estimate": 1.43,
                    "actual": 1.51,
                    "surprise_pct": 5.6,
                    "label": "BEAT",
                }
            ],
            "price_history": [
                {"date": "2024-09-01", "close": 220.0},
                {"date": "2024-09-03", "close": 222.5},
            ],
            "upcoming_fiscal_quarter": "2024-Q4",
            "earnings_anchor_date": "2024-09-30",
        }

    monkeypatch.setattr("src.predict.predict_for_ticker", _fake_predict)
    r = client.post(
        "/api/predict",
        json={"ticker": "AAPL"},
        headers={"x-api-key": os.environ["API_KEY"]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ticker"] == "AAPL"
    assert data["prediction"] == "BEAT"
    assert data["confidence"] == pytest.approx(0.71)
    assert set(data["probabilities"].keys()) == {"BEAT", "IN_LINE", "MISS"}
    assert len(data["top_features"]) == 3
    assert data["last_quarters"][0]["quarter"] == "2024-Q3"
