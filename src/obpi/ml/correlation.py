"""Correlation and benchmarking helpers for OBPI validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def spearman_correlation(
    obpi_scores: pd.Series | np.ndarray,
    comparison_scores: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute Spearman correlation and p-value."""
    aligned = pd.concat(
        [
            pd.Series(obpi_scores, name="obpi", dtype=float),
            pd.Series(comparison_scores, name="comparison", dtype=float),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < 3:
        raise ValueError("at least three paired scores are required")

    statistic, p_value = spearmanr(aligned["obpi"], aligned["comparison"])
    return {
        "spearman_rho": float(statistic),
        "p_value": float(p_value),
        "n": int(len(aligned)),
    }


def compare_benchmarks(
    obpi_scores: pd.Series,
    benchmarks: pd.DataFrame,
) -> pd.DataFrame:
    """Compare OBPI against each benchmark column with Spearman rho."""
    rows = []
    for column in benchmarks.columns:
        result = spearman_correlation(obpi_scores, benchmarks[column])
        rows.append({"benchmark": column, **result})
    return pd.DataFrame(rows).sort_values(
        "spearman_rho",
        ascending=False,
        ignore_index=True,
    )


def orthogonal_variance_test(
    scores_df: pd.DataFrame,
    n_components: int | None = None,
) -> dict[str, object]:
    """Run PCA on OBPI plus benchmark scores."""
    clean = scores_df.dropna().astype(float)
    if clean.shape[0] < 3 or clean.shape[1] < 2:
        raise ValueError("PCA requires at least three rows and two score columns")

    components = n_components or min(clean.shape)
    scaled = StandardScaler().fit_transform(clean)
    pca = PCA(n_components=components)
    pca.fit(scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=clean.columns,
        columns=[f"PC{i + 1}" for i in range(components)],
    )
    return {
        "explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
        "loadings": loadings,
        "n": int(len(clean)),
    }


def expert_correlation(
    obpi_scores: pd.Series | np.ndarray,
    expert_median: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute convergent validity against expert median ratings."""
    return spearman_correlation(obpi_scores, expert_median)


def cronbach_alpha(expert_ratings_df: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for expert rating columns."""
    ratings = expert_ratings_df.dropna().astype(float)
    if ratings.shape[1] < 2:
        raise ValueError("Cronbach alpha requires at least two raters")
    if ratings.shape[0] < 2:
        raise ValueError("Cronbach alpha requires at least two rated items")

    item_variances = ratings.var(axis=0, ddof=1)
    total_variance = ratings.sum(axis=1).var(ddof=1)
    if total_variance == 0.0:
        raise ValueError("total rating variance must be positive")

    n_items = ratings.shape[1]
    alpha = (n_items / (n_items - 1)) * (
        1 - float(item_variances.sum() / total_variance)
    )
    return float(alpha)
