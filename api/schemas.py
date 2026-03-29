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


class ErrorBody(BaseModel):
    error: str
