"""HTTP API package (FastAPI).

The ASGI app lives in :mod:`api.main`. Request/response types are in :mod:`api.schemas`.

Run locally (from project root, with ``API_KEY`` and model artifacts configured)::

    uvicorn api.main:app --reload
"""
