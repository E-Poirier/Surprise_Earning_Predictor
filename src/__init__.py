"""Earnings surprise pipeline (Phases 1–5).

Typical flow: :mod:`src.ingestion` → :mod:`src.features` → :mod:`src.train` →
:mod:`src.predict` / :mod:`api.main`. Sentiment lives in :mod:`src.sentiment`;
shared error types in :mod:`src.errors`.
"""
