"""Helpers shared by every fitted model pipeline."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def log_top_features(pipeline: Any, model_name: str, top_k: int = 5) -> None:
    """Log the most influential features of a fitted pipeline, when the model exposes them.

    Uses ``|coef_|`` for linear models and ``feature_importances_`` for trees;
    other models (calibrated SVM, MLP) are skipped.
    """
    model = pipeline.named_steps["model"]
    names = pipeline[:-1].get_feature_names_out()
    if hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        order = np.argsort(np.abs(coefs))[::-1][:top_k]
        lines = [
            f"    {'+' if coefs[i] > 0 else '-'} {names[i]:<25} coef = {coefs[i]:+.4f}"
            for i in order
        ]
        logger.info("[%s] Top %d features by |coef|:\n%s", model_name, top_k, "\n".join(lines))
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1][:top_k]
        lines = [f"    {names[i]:<25} importance = {importances[i]:.4f}" for i in order]
        logger.info("[%s] Top %d features by importance:\n%s", model_name, top_k, "\n".join(lines))
