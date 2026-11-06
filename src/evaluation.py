"""Evaluation metrics, aggregation, and result persistence for Phase 5.

The supervised experiments report two test-set metrics:

* **Accuracy** — fraction of correctly classified test samples.
* **Macro F1** — unweighted mean of per-class F1 scores. Macro averaging gives
  every class the same influence, which matters when class frequencies differ.

Metrics are always computed on the held-out test set, never on training rows.
Training time is recorded by the experiment runner around ``model.fit`` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

SUPPORTED_METRICS: tuple[str, ...] = ("accuracy", "macro_f1")
METRIC_COLUMNS: tuple[str, ...] = ("accuracy", "macro_f1")


def _validate_vectors(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.ndim != 1 or prediction.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if truth.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if truth.size != prediction.size:
        raise ValueError(
            f"y_true has {truth.size} labels but y_pred has {prediction.size}."
        )
    if truth.dtype == object and any(value is None for value in truth):
        raise ValueError("y_true contains missing labels.")
    if prediction.dtype == object and any(value is None for value in prediction):
        raise ValueError("y_pred contains missing labels.")
    return truth, prediction


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute accuracy and macro F1 for test-set predictions."""
    truth, prediction = _validate_vectors(y_true, y_pred)
    requested = list(metric_names or SUPPORTED_METRICS)
    unknown = [name for name in requested if name not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(
            f"Unsupported metric(s) {unknown}; choose from {SUPPORTED_METRICS}."
        )

    metrics: dict[str, float] = {}
    if "accuracy" in requested:
        metrics["accuracy"] = float(accuracy_score(truth, prediction))
    if "macro_f1" in requested:
        metrics["macro_f1"] = float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        )
    return metrics


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Validate test predictions and delegate to metric computation."""
    metrics = compute_classification_metrics(y_true, y_pred, metric_names)
    for name, value in metrics.items():
        if not np.isfinite(value):
            raise RuntimeError(f"Metric {name} is not finite: {value!r}.")
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Metric {name} is outside [0, 1]: {value!r}.")
    return metrics


def compute_acceptance_rate(n_pseudo_labels: int, n_unlabelled: int) -> float:
    """Fraction of the original unlabelled pool accepted as pseudo-labelled.

    The acceptance rate is a training-time statistic only; it is never used to
    select thresholds or to decide when to stop.
    """
    pseudo = int(n_pseudo_labels)
    pool = int(n_unlabelled)
    if pseudo < 0 or pool < 0:
        raise ValueError("Pseudo-label and unlabelled counts must be non-negative.")
    if pseudo > pool:
        raise ValueError(
            f"Pseudo-label count {pseudo} exceeds the unlabelled pool size {pool}."
        )
    if pool == 0:
        return 0.0
    return pseudo / pool


def aggregate_results(
    results: "pd.DataFrame",
    group_by: Sequence[str] | None = None,
) -> "pd.DataFrame":
    """Aggregate metrics with mean and standard deviation over repeats.

    ``labelled_percentage`` and ``feature_representation`` are preserved as
    grouping keys so the report can compare experimental factors.
    """
    if results.empty:
        raise ValueError("Cannot aggregate an empty results table.")
    keys = list(group_by or ("dataset", "feature_representation", "classifier"))
    missing = [key for key in keys if key not in results.columns]
    if missing:
        raise ValueError(f"Missing grouping columns: {missing}.")
    aggregated = (
        results.groupby(keys, dropna=False)[list(METRIC_COLUMNS)]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        "_".join(str(part) for part in column if part).strip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in aggregated.columns
    ]
    return aggregated


def save_metrics(
    results: "pd.DataFrame",
    output_path: str | Path = "results/classifier_evaluation/tables/classifier_results.csv",
) -> Path:
    """Persist the experiment results table to CSV and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return path


def to_json_records(frame: "pd.DataFrame") -> list[dict[str, Any]]:
    """Convert a results frame to JSON-safe Python records."""
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        cleaned: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (np.integer,)):
                cleaned[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                cleaned[key] = float(value) if np.isfinite(value) else None
            elif value is None or isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        records.append(cleaned)
    return records
