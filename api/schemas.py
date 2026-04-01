"""Pydantic request/response models (Phase 5)."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_version: str


class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="NYSE/Nasdaq symbol from the supported universe")


class TopFeatureItem(BaseModel):
    feature: str
    value: float
    direction: Literal["positive", "negative", "neutral"]


class ShapRowItem(BaseModel):
    feature: str
    value: float
    shap: float


class ShapExplanation(BaseModel):
    """TreeSHAP contributions for the predicted class (margin / logit before softmax)."""

    explained_class: str
    base_value: float
    model_output: float
    rows: List[ShapRowItem]


class LastQuarterItem(BaseModel):
    quarter: str
    estimate: Optional[float] = None
    actual: Optional[float] = None
    surprise_pct: Optional[float] = None
    label: Optional[str] = None


class PricePoint(BaseModel):
    """Daily close from processed Yahoo price history (UTC-stripped dates)."""

    date: str
    close: float


class PredictResponse(BaseModel):
    ticker: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    top_features: List[TopFeatureItem]
    last_quarters: List[LastQuarterItem]
    upcoming_fiscal_quarter: str = Field(
        "",
        description="Fiscal quarter label for the upcoming EPS row (estimate vs actual not yet reported)",
    )
    earnings_anchor_date: Optional[str] = Field(
        None,
        description="Fiscal period-end date (D) used as the earnings anchor for PIT features, ISO YYYY-MM-DD",
    )
    price_history: List[PricePoint] = Field(
        default_factory=list,
        description="Recent daily closes (up to ~90 calendar days) for charting",
    )
    shap_explanation: Optional[ShapExplanation] = Field(
        None,
        description="Per-instance SHAP for the predicted class (margin space); omitted if unavailable",
    )


class ErrorBody(BaseModel):
    error: str
    reason: Optional[str] = Field(None, description="Machine-readable code when error is insufficient_history")
    detail: Optional[str] = Field(None, description="Human-readable explanation")


class PredictableTickersResponse(BaseModel):
    """Symbols that can build a full inference row from processed data (see ``predictability_for_ticker``)."""

    tickers: List[str] = Field(..., description="Subset of the configured universe that is currently predictable")
    ineligible: Dict[str, str] = Field(
        default_factory=dict,
        description="Ticker → reason code for symbols that are not predictable",
    )
