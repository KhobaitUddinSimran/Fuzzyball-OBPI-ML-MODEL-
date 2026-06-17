"""Validation helpers for discriminating high and low OBPI profiles."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

METRIC_COLUMNS = [f"M{i}" for i in range(1, 10)]


@dataclass(frozen=True)
class ValidationResult:
    """Cross-validation summary for one model."""

    model_name: str
    accuracy_mean: float
    accuracy_std: float
    roc_auc_mean: float
    roc_auc_std: float
    recall_class_1_mean: float
    recall_class_1_std: float
    n_samples: int
    n_splits: int
    best_params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class TrainingPreparationResult:
    """Prepared extreme-quartile dataset plus reusable metadata."""

    prepared_df: pd.DataFrame
    metadata: dict[str, Any]


def create_labels(
    obpi_scores: pd.Series,
    high_quantile: float = 0.75,
    low_quantile: float = 0.25,
) -> pd.Series:
    """Label exact top/bottom quantile slices and discard the middle."""
    if not 0.0 < low_quantile < high_quantile < 1.0:
        raise ValueError("expected 0 < low_quantile < high_quantile < 1")

    scores = pd.Series(obpi_scores).dropna().astype(float)
    if scores.empty:
        raise ValueError("obpi_scores must contain at least one non-null score")

    n_rows = len(scores)
    n_low = max(1, int(np.floor(n_rows * low_quantile)))
    n_high = max(1, int(np.floor(n_rows * (1.0 - high_quantile))))
    if n_low + n_high > n_rows:
        raise ValueError("quantile selection leaves no middle rows")

    sorted_scores = scores.sort_values(kind="mergesort")
    low_indices = sorted_scores.head(n_low).index
    high_indices = sorted_scores.tail(n_high).index

    overlap = set(low_indices) & set(high_indices)
    if overlap:
        raise ValueError("quantile selection produced overlapping label sets")

    labels = pd.Series(np.nan, index=scores.index, dtype="float64")
    labels.loc[low_indices] = 0
    labels.loc[high_indices] = 1
    return labels.dropna().astype(int).sort_index()


def prepare_labeled_data(
    metrics_df: pd.DataFrame,
    metric_columns: list[str] | None = None,
    score_column: str = "obpi",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X/y after creating OBPI extreme-quartile labels."""
    metric_columns = metric_columns or METRIC_COLUMNS
    required_columns = set(metric_columns + [score_column])
    missing = required_columns - set(metrics_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"metrics_df is missing required columns: {missing_text}")

    metric_values = metrics_df.loc[:, metric_columns].astype(float)
    if metric_values.isna().any().any():
        raise ValueError("metric columns cannot contain missing values")
    if ((metric_values < 0.0) | (metric_values > 1.0)).any().any():
        raise ValueError("metric columns must be normalized to [0, 1]")

    labels = create_labels(metrics_df[score_column])
    x = metric_values.loc[labels.index]
    y = labels
    _validate_class_balance(y)
    return x, y


def prepare_training_frame(
    metrics_df: pd.DataFrame,
    metric_columns: list[str] | None = None,
    score_column: str = "obpi",
    cv_splits: int = 5,
    id_columns: list[str] | None = None,
) -> TrainingPreparationResult:
    """Create an ML-ready extreme-quartile dataset with scaled features and CV folds."""
    metric_columns = metric_columns or METRIC_COLUMNS
    id_columns = id_columns or [
        column
        for column in metrics_df.columns
        if column not in set(metric_columns + [score_column])
    ]

    x, y = prepare_labeled_data(
        metrics_df,
        metric_columns=metric_columns,
        score_column=score_column,
    )
    cv = _make_cv(y, cv_splits)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    scaled_columns = [f"{column}_scaled" for column in metric_columns]
    prepared_df = metrics_df.loc[y.index, id_columns + metric_columns + [score_column]].copy()
    prepared_df["label"] = y.to_numpy()
    for idx, _column in enumerate(metric_columns):
        prepared_df[scaled_columns[idx]] = x_scaled[:, idx]

    folds = []
    for fold_idx, (_, test_idx) in enumerate(cv.split(x, y), start=1):
        fold_indices = y.iloc[test_idx].index.tolist()
        folds.append({"fold": fold_idx, "test_indices": fold_indices})

    metadata = {
        "metric_columns": metric_columns,
        "scaled_metric_columns": scaled_columns,
        "score_column": score_column,
        "class_counts": {
            str(label): int(count)
            for label, count in y.value_counts().sort_index().items()
        },
        "n_rows_input": int(len(metrics_df)),
        "n_rows_prepared": int(len(prepared_df)),
        "n_splits": cv.get_n_splits(),
        "folds": folds,
        "scaler": {
            "mean": {
                column: float(scaler.mean_[idx])
                for idx, column in enumerate(metric_columns)
            },
            "scale": {
                column: float(scaler.scale_[idx])
                for idx, column in enumerate(metric_columns)
            },
        },
    }
    return TrainingPreparationResult(
        prepared_df=prepared_df.reset_index(drop=False),
        metadata=metadata,
    )


def save_training_preparation(
    result: TrainingPreparationResult,
    output_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Persist the prepared training frame and metadata to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.prepared_df.to_parquet(output, index=False)

    metadata_output = Path(metadata_path)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")


def extract_prepared_xy(
    prepared_df: pd.DataFrame,
    metric_columns: list[str] | None = None,
    label_column: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and labels from a training-prepared parquet frame."""
    metric_columns = metric_columns or METRIC_COLUMNS
    required = set(metric_columns + [label_column])
    missing = required - set(prepared_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"prepared_df is missing required columns: {missing_text}")

    x = prepared_df.loc[:, metric_columns].astype(float)
    y = prepared_df.loc[:, label_column].astype(int)
    _validate_class_balance(y)
    return x, y


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
    cv = _make_cv(y, cv_splits)
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
    cv = _make_cv(y, cv_splits)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc")
    grid.fit(x, y)
    return grid


def train_xgboost(
    x: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> GridSearchCV:
    """Train XGBoost when the optional dependency is installed."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ImportError(
            "xgboost is required for train_xgboost(); install xgboost to run "
            "Week 6 XGBoost validation"
        ) from exc

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
    )
    param_grid = {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }
    cv = _make_cv(y, cv_splits)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="roc_auc")
    grid.fit(x, y)
    return grid


def evaluate_estimator(
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cv_splits: int = 5,
    best_params: dict[str, Any] | None = None,
) -> ValidationResult:
    """Evaluate an estimator with stratified cross-validation."""
    cv = _make_cv(y, cv_splits)
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "recall_class_1": "recall",
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        scores = cross_validate(estimator, x, y, cv=cv, scoring=scoring)

    return ValidationResult(
        model_name=model_name,
        accuracy_mean=float(np.mean(scores["test_accuracy"])),
        accuracy_std=float(np.std(scores["test_accuracy"])),
        roc_auc_mean=float(np.mean(scores["test_roc_auc"])),
        roc_auc_std=float(np.std(scores["test_roc_auc"])),
        recall_class_1_mean=float(np.mean(scores["test_recall_class_1"])),
        recall_class_1_std=float(np.std(scores["test_recall_class_1"])),
        n_samples=int(len(y)),
        n_splits=cv.get_n_splits(),
        best_params=best_params,
    )


def evaluate_holdout_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Evaluate already-generated predictions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall_class_1": float(recall_score(y_true, y_pred)),
    }


def validate(
    metrics_df: pd.DataFrame,
    score_column: str = "obpi",
    metric_columns: list[str] | None = None,
    cv_splits: int = 5,
    include_xgboost: bool = False,
) -> dict[str, Any]:
    """Run no-leakage label construction and baseline model validation."""
    x, y = prepare_labeled_data(metrics_df, metric_columns, score_column)
    report: dict[str, Any] = {
        "n_samples": int(len(y)),
        "n_rows": int(len(metrics_df)),
        "class_counts": {
            str(label): int(count)
            for label, count in y.value_counts().sort_index().items()
        },
        "models": {},
        "notes": [],
    }

    models = {
        "logistic": train_logistic(x, y, cv_splits=cv_splits),
        "svm": train_svm(x, y, cv_splits=cv_splits),
    }
    for model_name, grid in models.items():
        report["models"][model_name] = evaluate_estimator(
            grid.best_estimator_,
            x,
            y,
            model_name,
            cv_splits=cv_splits,
            best_params=dict(grid.best_params_),
        ).to_dict()

    if include_xgboost:
        try:
            xgb_grid = train_xgboost(x, y, cv_splits=cv_splits)
        except ImportError as exc:
            report["notes"].append(str(exc))
        else:
            report["models"]["xgboost"] = evaluate_estimator(
                xgb_grid.best_estimator_,
                x,
                y,
                "xgboost",
                cv_splits=cv_splits,
                best_params=dict(xgb_grid.best_params_),
            ).to_dict()

    return report


def validate_prepared_data(
    prepared_df: pd.DataFrame,
    metric_columns: list[str] | None = None,
    label_column: str = "label",
    cv_splits: int = 5,
    include_xgboost: bool = False,
) -> dict[str, Any]:
    """Train and evaluate validation models from a pre-labeled prepared frame."""
    x, y = extract_prepared_xy(
        prepared_df,
        metric_columns=metric_columns,
        label_column=label_column,
    )
    report: dict[str, Any] = {
        "n_samples": int(len(y)),
        "n_rows": int(len(prepared_df)),
        "class_counts": {
            str(label): int(count)
            for label, count in y.value_counts().sort_index().items()
        },
        "models": {},
        "notes": [],
    }

    models = {
        "logistic": train_logistic(x, y, cv_splits=cv_splits),
        "svm": train_svm(x, y, cv_splits=cv_splits),
    }
    for model_name, grid in models.items():
        report["models"][model_name] = evaluate_estimator(
            grid.best_estimator_,
            x,
            y,
            model_name,
            cv_splits=cv_splits,
            best_params=dict(grid.best_params_),
        ).to_dict()

    if include_xgboost:
        try:
            xgb_grid = train_xgboost(x, y, cv_splits=cv_splits)
        except ImportError as exc:
            report["notes"].append(str(exc))
        else:
            report["models"]["xgboost"] = evaluate_estimator(
                xgb_grid.best_estimator_,
                x,
                y,
                "xgboost",
                cv_splits=cv_splits,
                best_params=dict(xgb_grid.best_params_),
            ).to_dict()

    return report


def save_validation_results(
    report: dict[str, Any],
    output_path: str | Path,
    markdown_path: str | Path | None = None,
) -> None:
    """Persist model-validation results as JSON and optional Markdown."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if markdown_path is None:
        return

    markdown_lines = [
        "# Validation Report",
        "",
        f"- Samples: {report['n_samples']}",
        f"- Input rows: {report['n_rows']}",
        f"- Class counts: {report['class_counts']}",
        "",
        "## Models",
        "",
    ]
    for model_name, metrics in report["models"].items():
        markdown_lines.extend(
            [
                f"### {model_name}",
                "",
                f"- Accuracy mean: {metrics['accuracy_mean']:.4f}",
                f"- ROC-AUC mean: {metrics['roc_auc_mean']:.4f}",
                f"- Recall@Class1 mean: {metrics['recall_class_1_mean']:.4f}",
                f"- Best params: {metrics.get('best_params')}",
                "",
            ]
        )
    if report.get("notes"):
        markdown_lines.extend(["## Notes", ""])
        markdown_lines.extend([f"- {note}" for note in report["notes"]])

    markdown_output = Path(markdown_path)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")


def _make_cv(y: pd.Series, cv_splits: int) -> StratifiedKFold:
    _validate_class_balance(y)
    min_class_count = int(y.value_counts().min())
    splits = min(cv_splits, min_class_count)
    if splits < 2:
        raise ValueError("each class must contain at least two samples")
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)


def _validate_class_balance(y: pd.Series) -> None:
    class_counts = y.value_counts()
    if set(class_counts.index) != {0, 1}:
        raise ValueError("labels must contain both class 0 and class 1")
    if class_counts.min() < 2:
        raise ValueError("each class must contain at least two samples")
