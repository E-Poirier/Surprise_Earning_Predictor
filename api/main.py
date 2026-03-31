"""FastAPI app entrypoint (Phase 5)."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_config
from config.tickers import TICKERS
from src.errors import InsufficientHistoryError
from src.model_io import load_model_bundle
from src.shap_explain import make_tree_explainer

from api.schemas import HealthResponse, PredictRequest, PredictResponse

_TICKER_SET = {t.upper() for t in TICKERS}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(_PROJECT_ROOT / ".env")
    cfg = load_config()
    app.state.config = cfg
    try:
        bundle = load_model_bundle(config=cfg)
        bundle["shap_explainer"] = make_tree_explainer(bundle["model"])
        app.state.model_bundle = bundle
        app.state.model_load_error = None
    except Exception as e:
        # Missing files, unpickle errors (e.g. xgboost not installed), etc.
        app.state.model_bundle = None
        app.state.model_load_error = str(e)
    yield


app = FastAPI(title="Earnings Surprise Predictor", lifespan=lifespan)


def _require_api_key(x_api_key: Annotated[Optional[str], Header(alias="x-api-key")] = None) -> bool:
    expected = os.environ.get("API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY is not set")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg: Dict[str, Any] = app.state.config
    ver = str(cfg.get("api", {}).get("model_version", "v1"))
    return HealthResponse(status="ok", model_version=ver)


@app.get("/api/tickers", response_model=List[str])
def list_tickers() -> List[str]:
    return list(TICKERS)


@app.post("/api/predict", response_model=None)
def predict(
    body: PredictRequest,
    _auth: Annotated[bool, Depends(_require_api_key)],
) -> Union[PredictResponse, JSONResponse]:
    t = body.ticker.strip().upper()
    if t not in _TICKER_SET:
        return JSONResponse(status_code=404, content={"error": "ticker_not_supported"})
    if app.state.model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {app.state.model_load_error}",
        )

    from src.predict import predict_for_ticker

    try:
        raw = predict_for_ticker(t, config=app.state.config, bundle=app.state.model_bundle)
    except InsufficientHistoryError:
        return JSONResponse(status_code=422, content={"error": "insufficient_history"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return PredictResponse.model_validate(raw)
