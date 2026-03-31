"""Unit tests for SHAP explanation helper (no Finnhub)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shap_explain import build_shap_explanation, make_tree_explainer


def test_make_tree_explainer_unsupported_type_returns_none():
    assert make_tree_explainer("not_a_model") is None


def test_build_shap_explanation_none_explainer():
    X = np.zeros((1, 3))
    out = build_shap_explanation(
        None,
        MagicMock(),
        X,
        ["a", "b", "c"],
        ["BEAT", "IN_LINE", "MISS"],
        0,
    )
    assert out is None


def test_build_shap_explanation_ndarray_multiclass():
    """Matches SHAP (1, n_features, n_class) layout from TreeExplainer."""
    explainer = MagicMock()
    explainer.shap_values = MagicMock(
        return_value=np.zeros((1, 4, 3))
    )
    explainer.expected_value = np.array([0.1, 0.2, 0.3])

    X = np.array([[1.0, 2.0, 3.0, 4.0]])
    sv = np.zeros((1, 4, 3))
    sv[0, :, 0] = [0.05, -0.02, 0.01, 0.0]
    explainer.shap_values = MagicMock(return_value=sv)

    model = MagicMock()
    # base 0.1 + sum(shap) = 0.14 for class 0
    model.predict = MagicMock(return_value=np.array([[0.14, 0.0, 0.0]]))

    out = build_shap_explanation(
        explainer,
        model,
        X,
        ["f0", "f1", "f2", "f3"],
        ["BEAT", "IN_LINE", "MISS"],
        0,
        top_n=10,
    )
    assert out is not None
    assert out["explained_class"] == "BEAT"
    assert out["base_value"] == pytest.approx(0.1)
    assert len(out["rows"]) >= 1
    assert all("shap" in r for r in out["rows"])
