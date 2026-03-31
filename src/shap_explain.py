"""TreeSHAP helpers for multiclass XGBoost (margin space)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def make_tree_explainer(model: Any) -> Any | None:
    """Return ``shap.TreeExplainer(model)`` or ``None`` if SHAP cannot be used."""
    try:
        import shap

        return shap.TreeExplainer(model)
    except Exception as e:
        logger.warning("SHAP TreeExplainer not available: %s", e)
        return None


def build_shap_explanation(
    explainer: Any | None,
    model: Any,
    X: np.ndarray,
    feature_columns: list[str],
    label_order: list[str],
    pred_idx: int,
    *,
    top_n: int = 12,
) -> dict[str, Any] | None:
    """Per-instance SHAP for the predicted class; top-|shap| rows plus optional *Other* bucket."""
    if explainer is None:
        return None
    try:
        sv = explainer.shap_values(X)
        ev = explainer.expected_value
    except Exception as e:
        logger.warning("SHAP values failed: %s", e)
        return None

    n_class = len(label_order)
    if isinstance(sv, list):
        if len(sv) != n_class or pred_idx >= len(sv):
            logger.warning("Unexpected SHAP list length: %s vs %s classes", len(sv), n_class)
            return None
        per = np.asarray(sv[pred_idx], dtype=np.float64).reshape(-1)
    else:
        arr = np.asarray(sv, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[-1] == n_class:
            # XGBoost multiclass: (n_samples, n_features, n_class)
            per = arr[0, :, pred_idx]
        elif arr.ndim == 2:
            per = arr[0]
        else:
            logger.warning("Unexpected SHAP array shape: %s", getattr(arr, "shape", None))
            return None

    ev_arr = np.asarray(ev, dtype=np.float64).reshape(-1)
    if pred_idx >= len(ev_arr):
        return None
    base = float(ev_arr[pred_idx])

    if len(per) != len(feature_columns):
        logger.warning("SHAP length %s != feature_columns %s", len(per), len(feature_columns))
        return None

    try:
        margin = model.predict(X, output_margin=True)
        m = np.asarray(margin, dtype=np.float64).reshape(-1)
        if len(m) == n_class:
            model_output = float(m[pred_idx])
        else:
            model_output = float(base + float(np.sum(per)))
    except Exception:
        model_output = float(base + float(np.sum(per)))

    pairs = [(feature_columns[i], float(per[i]), float(X[0, i])) for i in range(len(feature_columns))]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)

    cap = max(1, int(top_n))
    head = pairs[:cap]
    tail = pairs[cap:]
    other_shap = float(sum(t[1] for t in tail))

    rows: list[dict[str, Any]] = [
        {"feature": name, "value": val, "shap": sh}
        for name, sh, val in head
    ]
    if tail and abs(other_shap) > 1e-12:
        rows.append({"feature": "Other (remaining)", "value": 0.0, "shap": other_shap})

    explained = label_order[pred_idx]
    return {
        "explained_class": explained,
        "base_value": base,
        "model_output": model_output,
        "rows": rows,
    }
