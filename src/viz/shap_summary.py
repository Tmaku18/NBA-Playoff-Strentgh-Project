"""SHAP summary for Model B (TreeExplainer)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def shap_summary(
    model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    *,
    out_path: str | Path | None = None,
    max_display: int = 10,
) -> Any:
    """shap.TreeExplainer(model), explainer(X). Plot summary. Returns Explanation."""
    if not _HAS_SHAP:
        raise ImportError("shap is required")
    try:
        explainer = shap.TreeExplainer(model, data=X, feature_perturbation="interventional")
    except Exception:
        explainer = shap.TreeExplainer(model, data=X)
    sv = explainer.shap_values(X)
    if sv is None:
        raise ValueError("TreeExplainer.shap_values returned None")
    if isinstance(sv, list):
        sv = sv[0] if len(sv) else np.zeros_like(X)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        plt.figure()
        shap.summary_plot(sv, X, feature_names=feature_names, max_display=max_display, show=False)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    return sv
