"""Baseline sklearn classifier (scaled logistic regression)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 2000,
    class_weight: str | dict[int, float] | None = "balanced",
) -> tuple[Pipeline, dict[str, float]]:
    """Fit a ``StandardScaler + LogisticRegression`` pipeline.

    Args:
        X: Shape ``(n_samples, n_features)``.
        y: Shape ``(n_samples,)`` with class indices (e.g. 0=LEFT, 1=RIGHT).

    Returns:
        Fitted pipeline and a small metrics dict (train accuracy).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features)")
    pipe: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight=class_weight,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    y_hat = pipe.predict(X)
    metrics = {"train_accuracy": float(accuracy_score(y, y_hat))}
    return pipe, metrics


def save_sklearn_pipeline(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_sklearn_pipeline(path: str | Path) -> Any:
    return joblib.load(path)
