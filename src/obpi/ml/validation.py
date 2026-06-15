"""Validation helpers for discriminating high and low OBPI profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

METRIC_COLUMNS = [f"M{i}" for i in range(1, 10)]


@dataclass(frozen=True)
class ValidationResult:
    """Compact summary of a model validation run."""

    model_name: str
    accuracy: float
    roc_auc: float
    best_params: dict[str, Any]


def create_labels(obpi_scores: pd.Series) -> pd.Series:
    """Label top quartile as 1, bottom quartile as 0, and discard the middle."""

    if obpi_scores.empty:
        raise ValueError("obpi_scores must not be empty")

    q25 = float(obpi_scores.quantile(0.25))
    q75 = float(obpi_scores.quantile(0.75))
    labels = pd.Series(np.nan, index=obpi_scores.index, dtype="float")
    labels.loc[obpi_scores <= q25] = 0
    labels.loc[obpi_scores >= q75] = 1
    return labels.dropna().astype(int)


def train_svm(
    x: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> GridSearchCV:
    """Train an RBF SVM with standard scaling and grid search."""

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
        ],
    )
    param_grid = {
        "classifier__C": [0.1, 1.0, 10.0, 100.0],
        "classifier__gamma": ["scale", "auto", 0.001, 0.01],
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc")
    grid.fit(x, y)
    return grid


def train_logistic(
    x: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> GridSearchCV:
    """Train a scaled logistic-regression baseline."""

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2_000, random_state=42)),
        ],
    )
    param_grid = {"classifier__C": [0.1, 1.0, 10.0]}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc")
    grid.fit(x, y)
    return grid


def validate(
    metrics_df: pd.DataFrame,
    score_column: str = "obpi",
    metric_columns: list[str] | None = None,
    cv_splits: int = 5,
) -> dict[str, Any]:
    """Run no-leakage label construction and baseline model validation."""

    metric_columns = metric_columns or METRIC_COLUMNS
    required_columns = set(metric_columns + [score_column])
    missing = required_columns - set(metrics_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"metrics_df is missing required columns: {missing_text}")

    labels = create_labels(metrics_df[score_column])
    x = metrics_df.loc[labels.index, metric_columns]
    y = labels
    if y.nunique() != 2:
        raise ValueError("validation requires both high and low classes")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    models = {
        "svm": train_svm(x, y, cv_splits=cv_splits),
        "logistic": train_logistic(x, y, cv_splits=cv_splits),
    }

    results: dict[str, Any] = {"n_samples": int(len(y)), "models": {}}
    for model_name, grid in models.items():
        proba = cross_val_predict(
            grid.best_estimator_,
            x,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        predictions = (proba >= 0.5).astype(int)
        result = ValidationResult(
            model_name=model_name,
            accuracy=float(accuracy_score(y, predictions)),
            roc_auc=float(roc_auc_score(y, proba)),
            best_params=dict(grid.best_params_),
        )
        results["models"][model_name] = result.__dict__
    return results

