"""Lightweight exceptions shared by API and inference (no heavy imports)."""


class InsufficientHistoryError(Exception):
    """Raised when processed data does not support an upcoming-quarter prediction."""
