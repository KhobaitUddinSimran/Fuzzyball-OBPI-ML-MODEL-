"""Generate paper-ready OBPI validation figures."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))


def main() -> int:
    """Generate validation figures from current result artifacts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_parquet("data/processed/player_obpi_scores.parquet")
    cv_report = json.loads(Path("results/cv_results.json").read_text())
    weights = json.loads(Path("results/metric_weights.json").read_text())
    permutation = pd.read_csv("results/permutation_importance.csv")
    ablation = pd.read_csv("results/ablation_results.csv")

    _plot_obpi_distribution(scored, figures_dir / "obpi_distribution.png", plt)
    _plot_metric_distributions(scored, figures_dir / "metric_distributions.png", plt)
    _plot_model_accuracy(cv_report, figures_dir / "model_accuracy.png", plt)
    _plot_metric_weights(weights, figures_dir / "metric_weights.png", plt)
    _plot_permutation(permutation, figures_dir / "permutation_importance.png", plt)
    _plot_ablation(ablation, figures_dir / "ablation_plot.png", plt)

    shap_path = Path("results/shap_values.csv")
    if shap_path.exists():
        shap_values = pd.read_csv(shap_path)
        _plot_shap(shap_values, figures_dir / "shap_summary.png", plt)

    sensitivity_path = Path("results/frame_cap_sensitivity.csv")
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
        _plot_sensitivity(sensitivity, figures_dir / "frame_cap_sensitivity.png", plt)

    aggregate_path = Path("results/match_vs_aggregate_validation.json")
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text())
        _plot_match_vs_aggregate(
            aggregate,
            figures_dir / "match_vs_aggregate_accuracy.png",
            plt,
        )

    benchmark_path = Path("results/benchmark_correlations.csv")
    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path)
        _plot_benchmark(benchmark, figures_dir / "benchmark_correlations.png", plt)

    print(f"figures_dir: {figures_dir}")
    for path in sorted(figures_dir.glob("*.png")):
        print(path)
    return 0


def _plot_obpi_distribution(df, path: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.hist(df["obpi"], bins=24, color="#2563eb", edgecolor="white")
    ax.set_title("OBPI Score Distribution")
    ax.set_xlabel("OBPI")
    ax.set_ylabel("Player-match rows")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_metric_distributions(df, path: Path, plt) -> None:
    metrics = [f"M{i}" for i in range(1, 10)]
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), dpi=180)
    flat_axes = axes.ravel()
    for idx, metric in enumerate(metrics):
        ax = flat_axes[idx]
        ax.hist(df[metric], bins=18, color="#0f766e", edgecolor="white")
        ax.set_title(metric)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Normalized Metric Distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_model_accuracy(report: dict, path: Path, plt) -> None:
    models = list(report["models"])
    accuracy = [report["models"][model]["accuracy_mean"] for model in models]
    errors = [report["models"][model]["accuracy_std"] for model in models]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.bar(models, accuracy, yerr=errors, capsize=4, color="#7c3aed")
    ax.set_ylim(0.9, 1.01)
    ax.set_title("Cross-Validated Model Accuracy")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_metric_weights(weights: dict, path: Path, plt) -> None:
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.bar(labels, values, color="#ea580c")
    ax.set_title("Normalized Explainability Weights")
    ax.set_ylabel("Weight")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_permutation(df, path: Path, plt) -> None:
    score_col = "importance_mean" if "importance_mean" in df else df.columns[-1]
    ordered = df.sort_values(score_col, ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.barh(ordered["metric"], ordered[score_col], color="#0891b2")
    ax.set_title("Permutation Importance")
    ax.set_xlabel(score_col)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_ablation(df, path: Path, plt) -> None:
    ordered = df.sort_values("delta_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.barh(ordered["removed_metric"], ordered["delta_accuracy"], color="#16a34a")
    ax.set_title("Leave-One-Metric-Out Accuracy Drop")
    ax.set_xlabel("Full accuracy - ablated accuracy")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_shap(df, path: Path, plt) -> None:
    mean_abs = df.abs().mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.barh(mean_abs.index, mean_abs.values, color="#dc2626")
    ax.set_title("Mean Absolute SHAP Values")
    ax.set_xlabel("Mean |SHAP|")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_sensitivity(df, path: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    for column, label in [
        ("xgboost_accuracy_mean", "XGBoost"),
        ("svm_accuracy_mean", "SVM"),
        ("logistic_accuracy_mean", "Logistic"),
    ]:
        if column in df:
            ax.plot(df["frame_cap"], df[column], marker="o", label=label)
    ax.set_title("Frame-Cap Sensitivity")
    ax.set_xlabel("Per-match 360 frame cap")
    ax.set_ylabel("CV accuracy")
    ax.set_ylim(0.9, 1.01)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_benchmark(df, path: Path, plt) -> None:
    ordered = df.sort_values("spearman_rho", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.barh(ordered["benchmark"], ordered["spearman_rho"], color="#4f46e5")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Benchmark Spearman Correlations")
    ax.set_xlabel("Spearman rho")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_match_vs_aggregate(report: dict, path: Path, plt) -> None:
    levels = [
        ("Player-match", report["match_level"]),
        ("Aggregate player", report["aggregate_player_level"]),
    ]
    labels = []
    values = []
    for label, level in levels:
        best_model = level["best_accuracy_model"]
        labels.append(f"{label}\n{best_model}")
        values.append(level["models"][best_model]["accuracy_mean"])

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.bar(labels, values, color="#9333ea")
    ax.set_ylim(0.85, 1.01)
    ax.set_title("Match-Level vs Aggregate Validation")
    ax.set_ylabel("Best CV accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
