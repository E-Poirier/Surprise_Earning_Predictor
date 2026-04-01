"""Lightweight exceptions shared by API and inference (no heavy imports)."""

from typing import Optional

# Stable codes for API responses and logs (see ``build_upcoming_inference_row``).
REASON_MISSING_PROCESSED_DATA = "missing_processed_data"
REASON_SHORT_HISTORY = "short_history"
REASON_NO_UPCOMING_QUARTER = "no_upcoming_quarter"
REASON_PRIOR_QUARTER_LABELS = "prior_quarter_labels"
REASON_SURPRISE_MAGNITUDE = "surprise_magnitude"
REASON_PRICE_FEATURES = "price_features"

# Short messages for API ``detail`` / logs (keys = reason codes above).
REASON_DETAIL_MESSAGES: dict[str, str] = {
    REASON_MISSING_PROCESSED_DATA: "Missing processed earnings, prices, or metadata on disk; run ingestion.",
    REASON_SHORT_HISTORY: "Fewer than nine fiscal quarters after merge; refresh ingestion or add supplemental EPS history.",
    REASON_NO_UPCOMING_QUARTER: "No upcoming quarter with an EPS estimate and eight complete prior quarters; check Yahoo calendar merge.",
    REASON_PRIOR_QUARTER_LABELS: "Prior quarters lack usable actual/estimate pairs for beat-rate features.",
    REASON_SURPRISE_MAGNITUDE: "Could not compute surprise magnitude features from the prior four quarters.",
    REASON_PRICE_FEATURES: "Insufficient price history for momentum or 30d volatility before the earnings anchor.",
}


class InsufficientHistoryError(Exception):
    """Raised when processed data does not support an upcoming-quarter prediction."""

    def __init__(self, message: str, *, reason_code: Optional[str] = None):
        super().__init__(message)
        self.reason_code = reason_code
