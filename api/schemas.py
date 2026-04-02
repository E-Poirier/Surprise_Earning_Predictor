"""Pydantic request/response models shared by the FastAPI layer (Phase 5).

These mirror the JSON shape produced by :mod:`src.predict_core` so responses validate
without ad-hoc dict assembly in route handlers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service liveness and configured model label (from ``config.api.model_version``)."""

    status: Literal["ok"] = "ok"
    model_version: str


class PredictRequest(BaseModel):
    """Body for ``POST /api/predict``."""

    ticker: str = Field(..., min_length=1, description="NYSE/Nasdaq symbol from the supported universe")


class TopFeatureItem(BaseModel):
    """One row in the importance summary (value + coarse direction for UI)."""

    feature: str
    value: float
    direction: Literal["positive", "negative", "neutral"]


class ShapRowItem(BaseModel):
    """Single feature contribution toward the explained margin output."""

    feature: str
    value: float
    shap: float


class ShapExplanation(BaseModel):
    """TreeSHAP contributions for the predicted class (margin / logit before softmax)."""

    explained_class: str
    base_value: float
    model_output: float
    rows: list[ShapRowItem]


class LastQuarterItem(BaseModel):
    """Completed quarter shown for context (estimate, actual, derived surprise, label)."""

    quarter: str
    estimate: float | None = None
    actual: float | None = None
    surprise_pct: float | None = None
    label: str | None = None


class PricePoint(BaseModel):
    """Daily close from processed Yahoo price history (UTC-stripped dates)."""

    date: str
    close: float


class PredictResponse(BaseModel):
    """Successful prediction payload returned to the client."""

    ticker: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    top_features: list[TopFeatureItem]
    last_quarters: list[LastQuarterItem]
    upcoming_fiscal_quarter: str = Field(
        "",
        description="Fiscal quarter label for the upcoming EPS row (estimate vs actual not yet reported)",
    )
    earnings_anchor_date: str | None = Field(
        None,
        description="Fiscal period-end date (D) used as the earnings anchor for PIT features, ISO YYYY-MM-DD",
    )
    price_history: list[PricePoint] = Field(
        default_factory=list,
        description="Recent daily closes (up to ~90 calendar days) for charting",
    )
    shap_explanation: ShapExplanation | None = Field(
        None,
        description="Per-instance SHAP for the predicted class (margin space); omitted if unavailable",
    )


class ErrorBody(BaseModel):
    """Shape of JSON error bodies (e.g. insufficient history); not all routes use this model."""

    error: str
    reason: str | None = Field(None, description="Machine-readable code when error is insufficient_history")
    detail: str | None = Field(None, description="Human-readable explanation")


class PredictableTickersResponse(BaseModel):
    """Symbols that can build a full inference row from processed data (see ``predictability_for_ticker``)."""

    tickers: list[str] = Field(..., description="Subset of the configured universe that is currently predictable")
    ineligible: dict[str, str] = Field(
        default_factory=dict,
        description="Ticker → reason code for symbols that are not predictable",
    )
