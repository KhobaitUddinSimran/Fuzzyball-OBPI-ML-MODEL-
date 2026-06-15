import numpy as np
import pandas as pd

from obpi.ml.validation import create_labels, validate


def test_create_labels_keeps_top_and_bottom_quartiles() -> None:
    scores = pd.Series(np.arange(100) / 100)
    labels = create_labels(scores)

    assert len(labels) == 50
    assert labels.value_counts().to_dict() == {0: 25, 1: 25}
    assert labels.loc[0] == 0
    assert labels.loc[99] == 1


def test_validate_returns_model_metrics() -> None:
    rng = np.random.default_rng(42)
    low = rng.normal(0.2, 0.03, size=(30, 9))
    high = rng.normal(0.8, 0.03, size=(30, 9))
    x = np.clip(np.vstack([low, high]), 0.0, 1.0)
    df = pd.DataFrame(x, columns=[f"M{i}" for i in range(1, 10)])
    df["obpi"] = df.mean(axis=1)

    result = validate(df, cv_splits=3)

    assert result["n_samples"] == 30
    assert set(result["models"]) == {"svm", "logistic"}
    assert result["models"]["svm"]["accuracy"] >= 0.9
