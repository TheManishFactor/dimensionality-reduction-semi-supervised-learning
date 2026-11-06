"""Verify the Phase 6 semi-supervised self-training experiments.

Run from the project root:

    python scripts/verify_self_training.py

The verifier reads the saved Phase 6 artifacts and independently checks:

* exactly 81 experiment conditions are present, with every factor level;
* completed rows have finite metrics in [0, 1], non-negative times and counts,
  and a valid stopping reason consistent with the iteration limit;
* stored accuracy and macro F1 reproduce from the saved test predictions;
* pseudo-label audits contain unique positions drawn from the original
  unlabelled training pool, never from the labelled subset or the test set,
  and never contain unlabelled ground-truth labels;
* iteration logs are internally consistent and reconcile with the result row
  and the pseudo-label audit;
* baseline/PCA/LDA widths follow the Phase 4 rules, including the
  ``n_classes - 1`` LDA cap, and PCA/LDA fit scopes match the Phase 4 metadata;
* the Phase 5/Phase 6 comparison matches all 81 conditions and computes
  improvements as self-training minus supervised;
* no Phase 1-5 output file was added, removed, or modified, using the
  immutability manifest written before the Phase 6 run.

The verifier recomputes the Phase 3 splits and Phase 4 transformations only to
reconstruct the expected fit arrays. It never refits a Phase 6 model, never
uses unlabelled ground-truth labels, and never downloads or modifies data.

Exit code 0 means every check passed; 1 means at least one check failed.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import inspect
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from src.classifiers import CLASSIFIER_NAMES
from src.dimensionality_reduction import run_dimensionality_reduction
from src.experiment_runner import (
    DATASET_ORDER,
    LABELLED_PERCENTAGES,
    REPRESENTATION_ORDER,
    fingerprint_features,
    fingerprint_labels,
    get_self_training_views,
    load_config,
)
from src.preprocessing import CheckResult, preprocess_dataset
from src.self_training import SelfTrainingConfig, run_self_training

SEPARATOR = "=" * 78
OUTPUT_ROOT = PROJECT_ROOT / "results" / "self_training"
TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_DIR = OUTPUT_ROOT / "reports"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
LOG_DIR = OUTPUT_ROOT / "logs"
ITERATION_DIR = LOG_DIR / "iterations"
AUDIT_DIR = LOG_DIR / "pseudo_labels"
FIGURE_DIR = OUTPUT_ROOT / "figures"

CSV_PATH = TABLE_DIR / "self_training_results.csv"
JSON_PATH = TABLE_DIR / "self_training_results.json"
COMPARISON_CSV_PATH = TABLE_DIR / "supervised_vs_self_training.csv"
COMPARISON_JSON_PATH = TABLE_DIR / "supervised_vs_self_training.json"
REPORT_PATH = REPORT_DIR / "phase6_self_training_report.txt"
VERIFICATION_REPORT_PATH = REPORT_DIR / "phase6_verification_report.txt"
MANIFEST_PATH = LOG_DIR / "phase1_phase5_artifact_manifest.json"

EXPECTED_EXPERIMENTS = 81
METRIC_TOLERANCE = 1e-9
COUNT_TOLERANCE = 0

REQUIRED_COLUMNS: tuple[str, ...] = (
    "dataset",
    "labelled_percentage",
    "classifier",
    "feature_representation",
    "original_feature_count",
    "transformed_feature_count",
    "accuracy",
    "macro_f1",
    "training_time_seconds",
    "prediction_time_seconds",
    "pseudo_label_count",
    "remaining_unlabelled_count",
    "iterations_completed",
    "pseudo_label_acceptance_rate",
    "stopping_reason",
    "status",
)

ITERATION_COLUMNS: tuple[str, ...] = (
    "iteration",
    "pseudo_labels_added",
    "cumulative_pseudo_labels",
    "remaining_unlabelled",
    "mean_accepted_confidence",
    "min_accepted_confidence",
    "max_accepted_confidence",
    "training_set_size",
    "unlabelled_considered",
)

AUDIT_COLUMNS: tuple[str, ...] = (
    "unlabelled_position",
    "train_index",
    "dataset_index",
    "pseudo_label",
    "confidence",
    "iteration",
    "pseudo_label_source",
)

ALLOWED_STOPPING_REASONS: tuple[str, ...] = (
    "no_samples_meeting_confidence_threshold",
    "minimum_new_pseudo_labels_not_met",
    "max_iterations_reached",
    "no_unlabelled_samples_remaining",
)

EXPECTED_REPORT_SECTIONS = 18
EXPECTED_FIGURES = 18
REQUIRED_FIGURE_NAMES: tuple[str, ...] = (
    "supervised_vs_self_training_accuracy.png",
    "supervised_vs_self_training_macro_f1.png",
    "macro_f1_improvement_by_dataset.png",
    "self_training_performance_by_labelled_percentage.png",
    "pseudo_label_acceptance_counts.png",
    "pseudo_label_acceptance_rate_heatmap.png",
    "pseudo_label_growth.png",
    "training_time_comparison.png",
    "representation_comparison_self_training.png",
)

PHASE15_ROOTS: tuple[str, ...] = (
    "results/metrics",
    "results/figures",
    "results/reports",
    "results/dimensionality_reduction",
    "results/classifier_evaluation",
    "data/processed",
)


# ---------------------------------------------------------------------------
# Check bookkeeping
# ---------------------------------------------------------------------------
def _check(
    checks: list[CheckResult], name: str, passed: bool, detail: str
) -> None:
    checks.append(CheckResult(name=name, passed=bool(passed), detail=detail))


def _summarise(
    checks: list[CheckResult],
    name: str,
    failures: Sequence[str],
    total: int,
) -> None:
    if failures:
        shown = " | ".join(failures[:6])
        if len(failures) > 6:
            shown += f" | (+{len(failures) - 6} more)"
        detail = f"{len(failures)}/{total} failed: {shown}"
    else:
        detail = f"{total}/{total} passed"
    _check(checks, name, not failures, detail)


def _normalise_labels(values: Sequence[Any]) -> list[str]:
    return [str(value) for value in np.asarray(values).tolist()]


def _as_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def _resolve(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _enabled_datasets(config: Mapping[str, Any]) -> list[str]:
    return [
        name
        for name in DATASET_ORDER
        if bool(
            (config["data"]["datasets"].get(name, {}) or {}).get("enabled", True)
        )
    ]


def _configured_percentages(config: Mapping[str, Any]) -> list[int]:
    return [
        int(round(float(value) * 100))
        for value in config["data"]["labelled_proportions"]
    ]


def _expected_keys(
    datasets: Sequence[str], percentages: Sequence[int]
) -> set[tuple[str, int, str, str]]:
    return {
        (dataset, percentage, representation, classifier)
        for dataset in datasets
        for percentage in percentages
        for representation in REPRESENTATION_ORDER
        for classifier in CLASSIFIER_NAMES
    }


def _frame_key(row: Any) -> tuple[str, int, str, str]:
    return (
        str(row.dataset),
        int(round(float(row.labelled_percentage))),
        str(row.feature_representation),
        str(row.classifier),
    )


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def _load_artifacts(
    checks: list[CheckResult],
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    frame: pd.DataFrame | None = None
    payload: dict[str, Any] | None = None

    if CSV_PATH.is_file():
        frame = pd.read_csv(CSV_PATH)
        _check(checks, "Results CSV exists", True, str(CSV_PATH))
    else:
        _check(checks, "Results CSV exists", False, f"Missing {CSV_PATH}")

    if JSON_PATH.is_file():
        try:
            with JSON_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle, parse_constant=_reject_json_constant)
            _check(
                checks,
                "Results JSON is strict and parseable",
                True,
                str(JSON_PATH),
            )
        except Exception as exc:
            _check(
                checks,
                "Results JSON is strict and parseable",
                False,
                f"{type(exc).__name__}: {exc}",
            )
    else:
        _check(
            checks,
            "Results JSON is strict and parseable",
            False,
            f"Missing {JSON_PATH}",
        )
    return frame, payload


# ---------------------------------------------------------------------------
# Global result-table checks
# ---------------------------------------------------------------------------
def _check_coverage(
    frame: pd.DataFrame,
    datasets: Sequence[str],
    percentages: Sequence[int],
    checks: list[CheckResult],
) -> None:
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in frame.columns
    ]
    _check(
        checks,
        "All required result columns are present",
        not missing_columns,
        (
            f"{len(REQUIRED_COLUMNS)} required columns present"
            if not missing_columns
            else f"Missing: {missing_columns}"
        ),
    )
    if missing_columns:
        return

    _check(
        checks,
        "Experiment row count is 81",
        len(frame) == EXPECTED_EXPERIMENTS,
        f"rows={len(frame)} (expected {EXPECTED_EXPERIMENTS})",
    )

    key_columns = [
        "dataset",
        "labelled_percentage",
        "classifier",
        "feature_representation",
    ]
    duplicates = int(
        frame.duplicated(subset=key_columns, keep=False).sum()
    )
    _check(
        checks,
        "No duplicate experiment conditions",
        duplicates == 0,
        f"duplicate rows={duplicates}",
    )

    actual = {_frame_key(row) for row in frame.itertuples(index=False)}
    expected = _expected_keys(datasets, percentages)
    missing = sorted(expected.difference(actual))
    unexpected = sorted(actual.difference(expected))
    _check(
        checks,
        "All 81 conditions are covered exactly",
        not missing and not unexpected,
        (
            f"expected={len(expected)}, found={len(actual)}"
            if not missing and not unexpected
            else f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        ),
    )

    _check(
        checks,
        "All factor levels are present",
        (
            set(frame["dataset"]) == set(datasets)
            and set(frame["labelled_percentage"].astype(int))
            == set(percentages)
            and set(frame["feature_representation"])
            == set(REPRESENTATION_ORDER)
            and set(frame["classifier"]) == set(CLASSIFIER_NAMES)
        ),
        (
            f"datasets={sorted(set(frame['dataset']))}, "
            f"percentages={sorted(set(frame['labelled_percentage'].astype(int)))}, "
            f"representations={sorted(set(frame['feature_representation']))}, "
            f"classifiers={sorted(set(frame['classifier']))}"
        ),
    )

    statuses = set(frame["status"].astype(str))
    _check(
        checks,
        "Status values are completed or failed",
        statuses.issubset({"completed", "failed"}),
        f"status values={sorted(statuses)}",
    )

    completed = frame[frame["status"] == "completed"]
    failed = frame[frame["status"] == "failed"]
    _check(
        checks,
        "Completed and failed counts are reported explicitly",
        True,
        f"completed={len(completed)}, failed={len(failed)}",
    )


def _check_metrics_and_counts(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    checks: list[CheckResult],
) -> None:
    completed = frame[frame["status"] == "completed"].copy()
    if completed.empty:
        _check(checks, "Completed rows exist", False, "No completed rows.")
        return

    bad_metrics: list[str] = []
    for row in completed.itertuples(index=False):
        for column in ("accuracy", "macro_f1"):
            value = float(getattr(row, column))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                bad_metrics.append(f"{row.experiment_id}:{column}={value!r}")
    _summarise(
        checks,
        "Completed metrics are finite and inside [0, 1]",
        bad_metrics,
        len(completed) * 2,
    )

    bad_times: list[str] = []
    for row in completed.itertuples(index=False):
        for column in ("training_time_seconds", "prediction_time_seconds"):
            value = float(getattr(row, column))
            if not np.isfinite(value) or value < 0.0:
                bad_times.append(f"{row.experiment_id}:{column}={value!r}")
    _summarise(
        checks,
        "Training and prediction times are finite and non-negative",
        bad_times,
        len(completed) * 2,
    )

    bad_counts: list[str] = []
    for row in completed.itertuples(index=False):
        for column in (
            "pseudo_label_count",
            "remaining_unlabelled_count",
            "n_labelled_train",
            "n_unlabelled_train",
            "iterations_completed",
        ):
            value = getattr(row, column)
            if pd.isna(value) or float(value) < 0.0:
                bad_counts.append(f"{row.experiment_id}:{column}={value!r}")
    _summarise(
        checks,
        "Pseudo-label, unlabelled, and iteration counts are non-negative",
        bad_counts,
        len(completed) * 5,
    )

    bad_accounting: list[str] = []
    for row in completed.itertuples(index=False):
        pseudo = int(row.pseudo_label_count)
        remaining = int(row.remaining_unlabelled_count)
        total = int(row.n_unlabelled_train)
        if pseudo + remaining != total:
            bad_accounting.append(
                f"{row.experiment_id}:{pseudo}+{remaining}!={total}"
            )
        if pseudo > total or remaining > total:
            bad_accounting.append(
                f"{row.experiment_id}: count exceeds pool ({pseudo},{remaining}>{total})"
            )
        expected_rate = 0.0 if total == 0 else pseudo / total
        if not np.isclose(
            float(row.pseudo_label_acceptance_rate),
            expected_rate,
            atol=1e-12,
            rtol=0.0,
        ):
            bad_accounting.append(
                f"{row.experiment_id}: acceptance "
                f"{row.pseudo_label_acceptance_rate!r}!={expected_rate!r}"
            )
    _summarise(
        checks,
        "Pseudo-label accounting and acceptance rates are consistent",
        bad_accounting,
        len(completed),
    )

    max_iterations = int(
        SelfTrainingConfig.from_mapping(
            config.get("self_training", {})
        ).max_iterations
    )
    threshold = float(
        SelfTrainingConfig.from_mapping(
            config.get("self_training", {})
        ).confidence_threshold
    )
    bad_iterations = [
        f"{row.experiment_id}:iterations={row.iterations_completed}"
        for row in completed.itertuples(index=False)
        if int(row.iterations_completed) > max_iterations
    ]
    _summarise(
        checks,
        "Iteration counts do not exceed the configured maximum",
        bad_iterations,
        len(completed),
    )
    _check(
        checks,
        "Configured confidence threshold and iteration limit are valid",
        0.0 < threshold <= 1.0 and max_iterations >= 1,
        f"threshold={threshold}, max_iterations={max_iterations}",
    )

    unknown_stops = sorted(
        set(completed["stopping_reason"].astype(str))
        - set(ALLOWED_STOPPING_REASONS)
    )
    _check(
        checks,
        "Stopping reasons are within the documented set",
        not unknown_stops,
        f"unknown={unknown_stops}" if unknown_stops else "all reasons valid",
    )

    bad_stops: list[str] = []
    for row in completed.itertuples(index=False):
        reason = str(row.stopping_reason)
        iterations = int(row.iterations_completed)
        remaining = int(row.remaining_unlabelled_count)
        if reason == "max_iterations_reached":
            if iterations != max_iterations or remaining == 0:
                bad_stops.append(
                    f"{row.experiment_id}:{reason},iters={iterations},"
                    f"remaining={remaining}"
                )
        elif reason == "no_unlabelled_samples_remaining":
            if remaining != 0:
                bad_stops.append(
                    f"{row.experiment_id}:{reason},remaining={remaining}"
                )
        elif reason == "no_samples_meeting_confidence_threshold":
            if iterations > max_iterations:
                bad_stops.append(
                    f"{row.experiment_id}:{reason},iters={iterations}"
                )
        elif reason == "minimum_new_pseudo_labels_not_met":
            if iterations >= max_iterations and remaining > 0:
                bad_stops.append(
                    f"{row.experiment_id}:{reason},iters={iterations}"
                )
        else:
            bad_stops.append(f"{row.experiment_id}:unknown={reason}")
    _summarise(
        checks,
        "Stopping reasons are consistent with counts and iteration limits",
        bad_stops,
        len(completed),
    )

    bad_methods: list[str] = []
    for row in completed.itertuples(index=False):
        method = str(row.confidence_method).lower()
        if row.classifier in {"logistic_regression", "knn"}:
            if "predict_proba" not in method:
                bad_methods.append(f"{row.experiment_id}:{row.confidence_method!r}")
        elif row.classifier == "svm":
            if "calibrat" not in method or "predict_proba" not in method:
                bad_methods.append(f"{row.experiment_id}:{row.confidence_method!r}")
    _summarise(
        checks,
        "Confidence methods match the classifier type",
        bad_methods,
        len(completed),
    )

    method_sets = (
        completed.groupby("classifier")["confidence_method"]
        .nunique(dropna=True)
        .to_dict()
    )
    _check(
        checks,
        "Each classifier uses one documented confidence method",
        all(int(value) == 1 for value in method_sets.values()),
        f"{method_sets}",
    )


def _check_failed_rows(
    frame: pd.DataFrame, checks: list[CheckResult]
) -> None:
    failed = frame[frame["status"] == "failed"]
    bad: list[str] = []
    for row in failed.itertuples(index=False):
        if not isinstance(row.error_message, str) or not row.error_message.strip():
            bad.append(f"{row.experiment_id}: missing error message")
        for column in (
            "accuracy",
            "macro_f1",
            "training_time_seconds",
            "prediction_time_seconds",
            "pseudo_label_count",
            "remaining_unlabelled_count",
            "iterations_completed",
            "pseudo_label_acceptance_rate",
        ):
            if not pd.isna(getattr(row, column)):
                bad.append(f"{row.experiment_id}:{column} is not missing")
    _summarise(
        checks,
        "Failed rows are explicit and do not carry fabricated metrics",
        bad,
        max(len(failed), 1),
    )


def _check_degenerate_flags(
    frame: pd.DataFrame, checks: list[CheckResult]
) -> None:
    if "numerically_degenerate_fit" not in frame.columns:
        _check(
            checks,
            "Numerical degeneracy is reported explicitly",
            False,
            "Missing numerically_degenerate_fit column.",
        )
        return

    flags = frame["numerically_degenerate_fit"].map(_as_bool)
    flagged = frame[flags]
    bad: list[str] = []
    for row in flagged.itertuples(index=False):
        if str(row.status) != "completed":
            bad.append(f"{row.experiment_id}: flagged but status={row.status}")
        diagnostic = getattr(row, "fit_diagnostic", None)
        if not isinstance(diagnostic, str) or not diagnostic.strip():
            bad.append(f"{row.experiment_id}: missing fit diagnostic")
        warning = getattr(row, "warning_message", None)
        if not isinstance(warning, str) or "NumericalStabilityWarning" not in warning:
            bad.append(f"{row.experiment_id}: missing numerical warning")
        for column in ("accuracy", "macro_f1"):
            if not np.isfinite(float(getattr(row, column))):
                bad.append(f"{row.experiment_id}:{column} is not finite")
    _summarise(
        checks,
        "Numerically degenerate fits are flagged with diagnostics",
        bad,
        max(len(flagged), 1),
    )

    absent_diagnostics = [
        str(row.experiment_id)
        for row in frame[frame["status"] == "completed"].itertuples(index=False)
        if not isinstance(getattr(row, "fit_diagnostic", None), str)
        or not str(getattr(row, "fit_diagnostic")).strip()
    ]
    _summarise(
        checks,
        "Every completed fit has a diagnostic record",
        absent_diagnostics,
        max(int((frame["status"] == "completed").sum()), 1),
    )

    summary = (
        "; ".join(
            f"{row.experiment_id} (accuracy={float(row.accuracy):.4f})"
            for row in flagged.itertuples(index=False)
        )
        if len(flagged)
        else "none"
    )
    _check(
        checks,
        "Degenerate-fit count",
        True,
        f"{len(flagged)} flagged: {summary}",
    )


def _check_artifact_paths(
    frame: pd.DataFrame,
    checks: list[CheckResult],
) -> None:
    completed = frame[frame["status"] == "completed"]
    for column, directory, suffix in (
        ("prediction_file", PREDICTION_DIR, ".csv"),
        ("iteration_log_file", ITERATION_DIR, ".csv"),
        ("pseudo_label_file", AUDIT_DIR, ".csv"),
    ):
        missing: list[str] = []
        for row in completed.itertuples(index=False):
            path = _resolve(getattr(row, column))
            if path is None or not path.is_file():
                missing.append(f"{row.experiment_id}:{getattr(row, column)!r}")
        _summarise(
            checks,
            f"{column.replace('_', ' ').capitalize()} files exist",
            missing,
            max(len(completed), 1),
        )
        if directory.is_dir():
            actual = sorted(path.name for path in directory.glob(f"*{suffix}"))
            expected = sorted(
                f"{row.experiment_id}{suffix}"
                for row in completed.itertuples(index=False)
            )
            extra = sorted(set(actual) - set(expected))
            _check(
                checks,
                f"No stale {column.replace('_', ' ')} files",
                not extra,
                f"{len(actual)} files, extra={extra[:5]}"
                if extra
                else f"{len(actual)} files",
            )

    bad_paths: list[str] = []
    for row in completed.itertuples(index=False):
        for column in (
            "prediction_file",
            "iteration_log_file",
            "pseudo_label_file",
        ):
            path = _resolve(getattr(row, column))
            if path is None:
                bad_paths.append(f"{row.experiment_id}:{column} is empty")
            else:
                try:
                    path.relative_to(OUTPUT_ROOT)
                except ValueError:
                    bad_paths.append(
                        f"{row.experiment_id}:{column} escapes {OUTPUT_ROOT}"
                    )
    _summarise(
        checks,
        "Artifact paths stay inside the Phase 6 output directory",
        bad_paths,
        max(len(completed) * 3, 1),
    )


# ---------------------------------------------------------------------------
# Iteration logs and pseudo-label audits
# ---------------------------------------------------------------------------
def _check_iteration_log(
    row: Any,
    log: pd.DataFrame,
    threshold: float,
    checks: list[CheckResult],
) -> list[str]:
    failures: list[str] = []
    identifier = str(row.experiment_id)
    missing = [column for column in ITERATION_COLUMNS if column not in log.columns]
    if missing:
        return [f"{identifier}: missing columns {missing}"]
    if log.empty:
        return [f"{identifier}: empty iteration log"]

    rows = len(log)
    if int(row.iterations_completed) != rows:
        failures.append(
            f"{identifier}: iterations_completed={row.iterations_completed} != rows={rows}"
        )
    if list(log["iteration"].astype(int)) != list(range(1, rows + 1)):
        failures.append(f"{identifier}: iteration numbers are not 1..{rows}")

    added = log["pseudo_labels_added"].astype(int).to_numpy()
    cumulative = log["cumulative_pseudo_labels"].astype(int).to_numpy()
    remaining = log["remaining_unlabelled"].astype(int).to_numpy()
    training_size = log["training_set_size"].astype(int).to_numpy()
    considered = log["unlabelled_considered"].astype(int).to_numpy()

    if np.any(added <= 0):
        failures.append(f"{identifier}: non-positive pseudo-label additions")
    if not np.array_equal(cumulative, np.cumsum(added)):
        failures.append(f"{identifier}: cumulative counts are not a running sum")
    if int(cumulative[-1]) != int(row.pseudo_label_count):
        failures.append(
            f"{identifier}: log total={cumulative[-1]} != result={row.pseudo_label_count}"
        )
    if int(remaining[-1]) != int(row.remaining_unlabelled_count):
        failures.append(
            f"{identifier}: final remaining={remaining[-1]} != result="
            f"{row.remaining_unlabelled_count}"
        )
    if int(considered[0]) != int(row.n_unlabelled_train):
        failures.append(
            f"{identifier}: first considered={considered[0]} != pool="
            f"{row.n_unlabelled_train}"
        )
    if rows > 1 and not np.array_equal(considered[1:], remaining[:-1]):
        failures.append(f"{identifier}: considered pool does not follow removals")
    if not np.array_equal(
        training_size, int(row.n_labelled_train) + cumulative
    ):
        failures.append(f"{identifier}: training_set_size is inconsistent")
    if not np.array_equal(
        cumulative + remaining, np.full(rows, int(row.n_unlabelled_train))
    ):
        failures.append(f"{identifier}: cumulative + remaining != pool size")

    for matrix_name, matrix in (
        ("mean", log["mean_accepted_confidence"].to_numpy(dtype=float)),
        ("min", log["min_accepted_confidence"].to_numpy(dtype=float)),
        ("max", log["max_accepted_confidence"].to_numpy(dtype=float)),
    ):
        if not np.isfinite(matrix).all():
            failures.append(f"{identifier}: non-finite {matrix_name} confidence")
    minimums = log["min_accepted_confidence"].to_numpy(dtype=float)
    maximums = log["max_accepted_confidence"].to_numpy(dtype=float)
    means = log["mean_accepted_confidence"].to_numpy(dtype=float)
    if np.any(minimums < threshold - 1e-12):
        failures.append(f"{identifier}: accepted confidence below threshold")
    if np.any(maximums > 1.0 + 1e-12):
        failures.append(f"{identifier}: accepted confidence above 1")
    if np.any(minimums > means + 1e-12) or np.any(means > maximums + 1e-12):
        failures.append(f"{identifier}: confidence min/mean/max are not ordered")

    return failures


def _check_pseudo_audit(
    row: Any,
    audit: pd.DataFrame,
    split: Any,
    preprocessed: Any,
    checks: list[CheckResult],
) -> list[str]:
    failures: list[str] = []
    identifier = str(row.experiment_id)
    if list(audit.columns) != list(AUDIT_COLUMNS):
        failures.append(f"{identifier}: audit columns={list(audit.columns)}")
        return failures
    if len(audit) != int(row.pseudo_label_count):
        failures.append(
            f"{identifier}: audit rows={len(audit)} != pseudo count="
            f"{row.pseudo_label_count}"
        )

    forbidden = [
        column
        for column in audit.columns
        if "true" in column.lower() or "actual" in column.lower()
    ]
    if forbidden:
        failures.append(f"{identifier}: truth columns present {forbidden}")
    sources = set(audit["pseudo_label_source"].astype(str))
    if sources != {"classifier_prediction"}:
        failures.append(f"{identifier}: pseudo_label_source={sorted(sources)}")

    positions = audit["unlabelled_position"].to_numpy(dtype=np.int64)
    train_indices = audit["train_index"].to_numpy(dtype=np.int64)
    dataset_indices = audit["dataset_index"].to_numpy(dtype=np.int64)
    iterations = audit["iteration"].to_numpy(dtype=np.int64)
    confidences = audit["confidence"].to_numpy(dtype=float)
    labels = audit["pseudo_label"].tolist()

    if np.unique(positions).size != positions.size:
        failures.append(f"{identifier}: duplicate unlabelled positions")
    if np.unique(train_indices).size != train_indices.size:
        failures.append(f"{identifier}: duplicate train indices")

    n_unlabelled = int(split.n_unlabelled)
    if positions.size and (
        positions.min() < 0 or positions.max() >= n_unlabelled
    ):
        failures.append(f"{identifier}: position outside [0, {n_unlabelled})")

    unlabelled_indices = np.asarray(split.unlabelled_indices, dtype=np.int64)
    if positions.size and not np.array_equal(
        train_indices, unlabelled_indices[positions]
    ):
        failures.append(f"{identifier}: train_index != unlabelled_indices[position]")

    labelled = set(np.asarray(split.labelled_indices, dtype=np.int64).tolist())
    if any(int(value) in labelled for value in train_indices):
        failures.append(f"{identifier}: a labelled row was pseudo-labelled")

    train_partition = np.asarray(preprocessed.train_indices, dtype=np.int64)
    if dataset_indices.size:
        expected_dataset = train_partition[train_indices]
        if not np.array_equal(dataset_indices, expected_dataset):
            failures.append(
                f"{identifier}: dataset_index does not match train_indices"
            )
        if not bool(preprocessed.metadata.get("official_test", False)):
            test_set = set(
                np.asarray(preprocessed.test_indices, dtype=np.int64).tolist()
            )
            if any(int(value) in test_set for value in dataset_indices):
                failures.append(
                    f"{identifier}: a test row was pseudo-labelled"
                )

    if iterations.size and (
        iterations.min() < 1 or iterations.max() > int(row.iterations_completed)
    ):
        failures.append(f"{identifier}: pseudo-label iteration outside the log")
    if not np.isfinite(confidences).all():
        failures.append(f"{identifier}: non-finite pseudo confidences")

    known = set(_normalise_labels(np.unique(split.y_labelled)))
    unknown = [str(label) for label in labels if str(label) not in known]
    if unknown:
        failures.append(f"{identifier}: unknown pseudo classes {sorted(set(unknown))[:3]}")

    return failures


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def _check_predictions(
    row: Any,
    prediction: pd.DataFrame,
    preprocessed: Any,
    checks: list[CheckResult],
) -> list[str]:
    failures: list[str] = []
    identifier = str(row.experiment_id)
    required = ["test_position", "test_index", "y_true", "y_pred"]
    if list(prediction.columns) != required:
        failures.append(f"{identifier}: columns={list(prediction.columns)}")
        return failures
    if len(prediction) != int(preprocessed.n_test):
        failures.append(
            f"{identifier}: rows={len(prediction)} != n_test={preprocessed.n_test}"
        )
        return failures

    positions = prediction["test_position"].to_numpy(dtype=np.int64)
    if not np.array_equal(positions, np.arange(preprocessed.n_test)):
        failures.append(f"{identifier}: test_position is not 0..n_test-1")
    indices = prediction["test_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        indices, np.asarray(preprocessed.test_indices, dtype=np.int64)
    ):
        failures.append(f"{identifier}: test_index differs from the fixed split")

    expected_true = _normalise_labels(preprocessed.y_test)
    stored_true = _normalise_labels(prediction["y_true"])
    if stored_true != expected_true:
        failures.append(f"{identifier}: y_true differs from the fixed test labels")

    stored_pred = _normalise_labels(prediction["y_pred"])
    known = set(_normalise_labels(np.unique(preprocessed.y_train)))
    unknown = sorted(set(stored_pred) - known)
    if unknown:
        failures.append(f"{identifier}: unknown predicted classes {unknown[:3]}")

    accuracy = float(accuracy_score(expected_true, stored_pred))
    macro_f1 = float(
        f1_score(expected_true, stored_pred, average="macro", zero_division=0)
    )
    if not np.isclose(
        accuracy, float(row.accuracy), atol=METRIC_TOLERANCE, rtol=0.0
    ):
        failures.append(
            f"{identifier}: accuracy {row.accuracy!r} != recomputed {accuracy!r}"
        )
    if not np.isclose(
        macro_f1, float(row.macro_f1), atol=METRIC_TOLERANCE, rtol=0.0
    ):
        failures.append(
            f"{identifier}: macro_f1 {row.macro_f1!r} != recomputed {macro_f1!r}"
        )

    return failures


# ---------------------------------------------------------------------------
# Dataset-level checks: dimensions, provenance, leakage
# ---------------------------------------------------------------------------
def _phase4_metadata() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    configurations = PROJECT_ROOT / (
        "results/dimensionality_reduction/metadata/reduction_configurations.csv"
    )
    lda = PROJECT_ROOT / (
        "results/dimensionality_reduction/metadata/lda_details.csv"
    )
    return (
        pd.read_csv(configurations) if configurations.is_file() else None,
        pd.read_csv(lda) if lda.is_file() else None,
    )


def _check_phase4_rules(
    dataset: str,
    preprocessed: Any,
    configurations: pd.DataFrame | None,
    lda_details: pd.DataFrame | None,
    checks: list[CheckResult],
) -> None:
    if configurations is None or lda_details is None:
        _check(
            checks,
            "Phase 4 metadata is available for fit-scope checks",
            False,
            "Missing reduction_configurations.csv or lda_details.csv",
        )
        return

    config_rows = configurations[configurations["dataset"] == dataset]
    lda_rows = lda_details[lda_details["dataset"] == dataset]
    n_classes = len(np.unique(preprocessed.y_train))
    n_features = int(preprocessed.n_features)

    failures: list[str] = []
    for proportion, split in sorted(preprocessed.semi_supervised.items()):
        percentage = int(round(proportion * 100))
        pca_row = config_rows[
            (config_rows["method"] == "pca")
            & (np.isclose(config_rows["labelled_proportion"], proportion))
        ]
        lda_row = config_rows[
            (config_rows["method"] == "lda")
            & (np.isclose(config_rows["labelled_proportion"], proportion))
        ]
        lda_detail = lda_rows[np.isclose(lda_rows["labelled_proportion"], proportion)]
        if len(pca_row) != 1 or len(lda_row) != 1 or len(lda_detail) != 1:
            failures.append(f"{percentage}%: missing Phase 4 metadata row")
            continue
        pca_row = pca_row.iloc[0]
        lda_row = lda_row.iloc[0]
        lda_detail = lda_detail.iloc[0]

        if pca_row["fitted_on"] != "complete_training_pool":
            failures.append(f"{percentage}%: PCA fitted_on={pca_row['fitted_on']}")
        if int(pca_row["n_samples_fit"]) != int(preprocessed.n_train):
            failures.append(
                f"{percentage}%: PCA fit rows={pca_row['n_samples_fit']} "
                f"!= n_train={preprocessed.n_train}"
            )
        if not _as_bool(pca_row["projection_reused_across_proportions"]):
            failures.append(f"{percentage}%: PCA projection not reused")

        expected_fitted_on = f"labelled_training_subset_{proportion:.2f}"
        if str(lda_row["fitted_on"]) != expected_fitted_on:
            failures.append(
                f"{percentage}%: LDA fitted_on={lda_row['fitted_on']} "
                f"!= {expected_fitted_on}"
            )
        if int(lda_row["n_samples_fit"]) != int(split.n_labelled):
            failures.append(
                f"{percentage}%: LDA fit rows={lda_row['n_samples_fit']} "
                f"!= n_labelled={split.n_labelled}"
            )
        cap = min(n_features, n_classes - 1)
        if int(lda_detail["components"]) != cap:
            failures.append(
                f"{percentage}%: LDA components={lda_detail['components']} != cap={cap}"
            )
        if int(lda_detail["max_components"]) != n_classes - 1:
            failures.append(
                f"{percentage}%: LDA max_components={lda_detail['max_components']} "
                f"!= n_classes-1={n_classes - 1}"
            )
        if int(lda_detail["n_labelled_fit_rows"]) != int(split.n_labelled):
            failures.append(
                f"{percentage}%: LDA n_labelled_fit_rows="
                f"{lda_detail['n_labelled_fit_rows']} != {split.n_labelled}"
            )

    _summarise(
        checks,
        "PCA is fitted on the training pool and LDA on labelled rows only",
        failures,
        len(preprocessed.semi_supervised),
    )


def _check_dimensions(
    dataset_frame: pd.DataFrame,
    preprocessed: Any,
    suite: Any,
    checks: list[CheckResult],
) -> None:
    n_features = int(preprocessed.n_features)
    n_classes = len(np.unique(preprocessed.y_train))
    expected = {
        "baseline": (n_features, None),
        "pca": (int(suite.pca.n_components), None),
        "lda": (None, min(n_features, n_classes - 1)),
    }
    failures: list[str] = []
    for row in dataset_frame.itertuples(index=False):
        original = int(row.original_feature_count)
        transformed = int(row.transformed_feature_count)
        representation = str(row.feature_representation)
        if original != n_features:
            failures.append(f"{row.experiment_id}: original={original} != {n_features}")
        if representation == "baseline":
            if transformed != n_features:
                failures.append(f"{row.experiment_id}: baseline width={transformed}")
        elif representation == "pca":
            if transformed != int(suite.pca.n_components):
                failures.append(
                    f"{row.experiment_id}: PCA width={transformed} != "
                    f"{suite.pca.n_components}"
                )
        elif representation == "lda":
            result = suite.lda.get(float(round(row.labelled_percentage / 100.0, 2)))
            if result is None:
                failures.append(f"{row.experiment_id}: missing LDA result")
                continue
            cap = min(n_features, n_classes - 1)
            if transformed != int(result.n_components) or transformed > cap:
                failures.append(
                    f"{row.experiment_id}: LDA width={transformed}, "
                    f"result={result.n_components}, cap={cap}"
                )
        else:
            failures.append(f"{row.experiment_id}: unknown representation")
    _summarise(
        checks,
        "Feature widths follow baseline/PCA/LDA rules and the LDA cap",
        failures,
        len(dataset_frame),
    )

    pca_widths = set(
        dataset_frame.loc[
            dataset_frame["feature_representation"] == "pca",
            "transformed_feature_count",
        ].astype(int)
    )
    _check(
        checks,
        "PCA width is shared by every proportion and classifier",
        len(pca_widths) == 1,
        f"widths={sorted(pca_widths)}",
    )


def _compare_views(
    dataset_frame: pd.DataFrame,
    preprocessed: Any,
    suite: Any,
) -> tuple[list[str], set[str], dict[str, list[str]]]:
    """Compare stored views with a reconstructed suite.

    Returns structural failures (always fatal), the experiment ids whose
    feature fingerprints match, and per-row fingerprint mismatches.
    """
    failures: list[str] = []
    matched: set[str] = set()
    mismatched: dict[str, list[str]] = {}
    for row in dataset_frame.itertuples(index=False):
        proportion = float(round(row.labelled_percentage / 100.0, 2))
        split = preprocessed.semi_supervised[proportion]
        try:
            (
                X_labelled,
                X_unlabelled,
                X_test,
                _,
                transformed,
            ) = get_self_training_views(
                preprocessed,
                suite,
                split,
                str(row.feature_representation),
                proportion,
            )
        except Exception as exc:
            failures.append(
                f"{row.experiment_id}: view reconstruction failed: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if X_labelled.shape[0] != int(row.n_labelled_train):
            failures.append(
                f"{row.experiment_id}: labelled rows={X_labelled.shape[0]} "
                f"!= {row.n_labelled_train}"
            )
        if X_unlabelled.shape[0] != int(row.n_unlabelled_train):
            failures.append(
                f"{row.experiment_id}: unlabelled rows={X_unlabelled.shape[0]} "
                f"!= {row.n_unlabelled_train}"
            )
        if int(transformed) != int(row.transformed_feature_count):
            failures.append(
                f"{row.experiment_id}: width={transformed} != "
                f"{row.transformed_feature_count}"
            )
        if X_test.shape[0] != int(preprocessed.n_test):
            failures.append(
                f"{row.experiment_id}: test rows={X_test.shape[0]} "
                f"!= {preprocessed.n_test}"
            )

        row_mismatch = []
        if fingerprint_features(X_labelled) != str(row.fit_feature_sha256):
            row_mismatch.append("labelled feature fingerprint")
        if fingerprint_labels(split.y_labelled) != str(row.fit_label_sha256):
            row_mismatch.append("labelled target fingerprint")
        if fingerprint_features(X_unlabelled) != str(
            row.unlabelled_feature_sha256
        ):
            row_mismatch.append("unlabelled feature fingerprint")
        if row_mismatch:
            mismatched[str(row.experiment_id)] = row_mismatch
        else:
            matched.add(str(row.experiment_id))
    return failures, matched, mismatched


def _check_views_and_fingerprints(
    dataset_frame: pd.DataFrame,
    preprocessed: Any,
    suite: Any,
    config: Mapping[str, Any],
    checks: list[CheckResult],
) -> None:
    failures, matched_default, mismatched = _compare_views(
        dataset_frame, preprocessed, suite
    )
    matched_alt: set[str] = set()
    alt_note = ""
    if mismatched:
        # Randomized PCA and the LDA SVD solver produce last-bit differences
        # when BLAS partitions work differently. Phase 6 ran with both the
        # default pool and a pinned two-thread pool, so a stored fingerprint
        # is accepted if it is bit-identical under either canonical setting.
        pending = dataset_frame[
            dataset_frame["experiment_id"].astype(str).isin(mismatched)
        ]
        try:
            with threadpool_limits(limits=2):
                alt_preprocessed = preprocess_dataset(preprocessed.name, config)
                alt_suite = run_dimensionality_reduction(
                    alt_preprocessed, config
                )
            _, matched_alt, alt_mismatched = _compare_views(
                pending, alt_preprocessed, alt_suite
            )
            del alt_suite
            del alt_preprocessed
            gc.collect()
        except Exception as exc:
            alt_mismatched = dict(mismatched)
            alt_note = f"; two-thread recomputation failed: {type(exc).__name__}: {exc}"

        for identifier in mismatched:
            if identifier not in matched_alt:
                reasons = alt_mismatched.get(identifier, mismatched[identifier])
                failures.append(f"{identifier}: {', '.join(reasons)}")

    failed_ids = {failure.split(":", 1)[0] for failure in failures}
    passed = len(dataset_frame) - len(failed_ids)
    _check(
        checks,
        "Stored fit arrays match the fixed labelled and unlabelled views",
        not failures,
        f"{passed}/{len(dataset_frame)} passed "
        f"(default BLAS threads: {len(matched_default)}, "
        f"pinned two-thread setting: {len(matched_alt)}){alt_note}",
    )


def _check_split_leakage(
    dataset_frame: pd.DataFrame,
    preprocessed: Any,
    checks: list[CheckResult],
) -> None:
    failures: list[str] = []
    for row in dataset_frame.itertuples(index=False):
        proportion = float(round(row.labelled_percentage / 100.0, 2))
        split = preprocessed.semi_supervised[proportion]
        if set(np.asarray(split.labelled_indices).tolist()) & set(
            np.asarray(split.unlabelled_indices).tolist()
        ):
            failures.append(f"{row.experiment_id}: labelled/unlabelled overlap")
        if split.n_labelled + split.n_unlabelled != preprocessed.n_train:
            failures.append(f"{row.experiment_id}: split does not cover training")
        if int(row.n_labelled_train) + int(row.n_unlabelled_train) != int(
            preprocessed.n_train
        ):
            failures.append(f"{row.experiment_id}: counts do not cover training")
        # HAR train and test come from separate source files, so their index
        # arrays describe two independent row spaces. Only holdout datasets
        # index into one shared source table where disjointness is meaningful;
        # HAR separation is verified through the subject-wise check below.
        if not bool(preprocessed.metadata.get("official_test", False)):
            test_indices = set(
                np.asarray(preprocessed.test_indices).tolist()
            )
            train_indices = set(
                np.asarray(preprocessed.train_indices).tolist()
            )
            if test_indices & train_indices:
                failures.append(
                    f"{row.experiment_id}: train/test index overlap"
                )

    _summarise(
        checks,
        "Fixed splits keep labelled, unlabelled, and test rows disjoint",
        failures,
        len(dataset_frame),
    )

    if dataset_frame is not None and len(dataset_frame):
        if preprocessed.train_subjects is not None:
            train_subjects = set(
                np.asarray(preprocessed.train_subjects).tolist()
            )
            test_subjects = set(np.asarray(preprocessed.test_subjects).tolist())
            _check(
                checks,
                "HAR subject-wise train/test separation is preserved",
                not (train_subjects & test_subjects),
                f"train_subjects={len(train_subjects)}, "
                f"test_subjects={len(test_subjects)}, "
                f"overlap={sorted(train_subjects & test_subjects)}",
            )
        elif preprocessed.name == "har":
            _check(
                checks,
                "HAR subject-wise train/test separation is preserved",
                False,
                "HAR has no subject arrays.",
            )


def _check_dataset_specific_rules(
    preprocessed: Any, checks: list[CheckResult]
) -> None:
    if preprocessed.name == "mnist":
        metadata = preprocessed.metadata or {}
        n_original = int(metadata.get("n_original", -1))
        test_indices = np.asarray(preprocessed.test_indices, dtype=np.int64)
        _check(
            checks,
            "MNIST uses only the 42,000 labelled competition-training rows",
            n_original == 42000
            and int(preprocessed.n_test) == 8400
            and int(test_indices.max()) < 42000,
            f"n_original={n_original}, n_test={preprocessed.n_test}, "
            f"max_test_index={int(test_indices.max())}",
        )
        _check(
            checks,
            "MNIST competition test.csv is ignored for evaluation",
            bool(metadata.get("competition_test_ignored", False))
            and bool(metadata.get("competition_test_present", False)),
            f"ignored={metadata.get('competition_test_ignored')}, "
            f"present={metadata.get('competition_test_present')}",
        )


def _verify_dataset(
    dataset: str,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    configurations: pd.DataFrame | None,
    lda_details: pd.DataFrame | None,
    checks: list[CheckResult],
) -> None:
    dataset_frame = frame[frame["dataset"] == dataset].copy()
    _check(
        checks,
        "Dataset contributes 27 experiments",
        len(dataset_frame) == 27,
        f"rows={len(dataset_frame)}",
    )
    if dataset_frame.empty:
        return

    try:
        preprocessed = preprocess_dataset(dataset, config)
        suite = run_dimensionality_reduction(preprocessed, config)
        if suite.pca is None or not suite.lda:
            raise RuntimeError("Phase 4 suite is missing PCA or LDA results.")
    except Exception as exc:
        _check(
            checks,
            "Dataset reconstruction",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        return

    try:
        _check_dimensions(dataset_frame, preprocessed, suite, checks)
        _check_phase4_rules(
            dataset, preprocessed, configurations, lda_details, checks
        )
        _check_views_and_fingerprints(
            dataset_frame, preprocessed, suite, config, checks
        )
        _check_split_leakage(dataset_frame, preprocessed, checks)
        _check_dataset_specific_rules(preprocessed, checks)

        threshold = float(
            SelfTrainingConfig.from_mapping(
                config.get("self_training", {})
            ).confidence_threshold
        )
        completed = dataset_frame[
            dataset_frame["status"] == "completed"
        ]
        iteration_failures: list[str] = []
        audit_failures: list[str] = []
        prediction_failures: list[str] = []
        for row in completed.itertuples(index=False):
            iteration_path = _resolve(row.iteration_log_file)
            audit_path = _resolve(row.pseudo_label_file)
            prediction_path = _resolve(row.prediction_file)
            proportion = float(round(row.labelled_percentage / 100.0, 2))
            split = preprocessed.semi_supervised[proportion]

            if iteration_path and iteration_path.is_file():
                log = pd.read_csv(iteration_path)
                iteration_failures.extend(
                    _check_iteration_log(row, log, threshold, checks)
                )
            if audit_path and audit_path.is_file():
                audit = pd.read_csv(audit_path)
                audit_failures.extend(
                    _check_pseudo_audit(
                        row, audit, split, preprocessed, checks
                    )
                )
            if prediction_path and prediction_path.is_file():
                prediction = pd.read_csv(prediction_path)
                prediction_failures.extend(
                    _check_predictions(row, prediction, preprocessed, checks)
                )

        _summarise(
            checks,
            "Iteration logs are internally consistent",
            iteration_failures,
            max(len(completed), 1),
        )
        _summarise(
            checks,
            "Pseudo-label audits use unique unlabelled positions only",
            audit_failures,
            max(len(completed), 1),
        )
        _summarise(
            checks,
            "Stored metrics reproduce from saved test predictions",
            prediction_failures,
            max(len(completed), 1),
        )
        _check(
            checks,
            "No test row appears in any pseudo-label audit",
            not any(
                "test row" in failure or "test" in failure.lower()
                for failure in audit_failures
            ),
            "holdout datasets checked against preprocessed.test_indices; "
            "HAR pseudo-labels index the training file only, and its "
            "subject-wise separation is checked separately",
        )
    finally:
        del suite
        del preprocessed
        gc.collect()


# ---------------------------------------------------------------------------
# Comparison checks
# ---------------------------------------------------------------------------
def _check_comparison(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    checks: list[CheckResult],
) -> None:
    if not COMPARISON_CSV_PATH.is_file():
        _check(
            checks,
            "Comparison CSV exists",
            False,
            f"Missing {COMPARISON_CSV_PATH}",
        )
        return
    comparison = pd.read_csv(COMPARISON_CSV_PATH)
    _check(
        checks,
        "Comparison CSV exists",
        True,
        str(COMPARISON_CSV_PATH),
    )
    _check(
        checks,
        "Comparison has 81 matched rows",
        len(comparison) == EXPECTED_EXPERIMENTS,
        f"rows={len(comparison)}",
    )

    key_columns = [
        "dataset",
        "labelled_percentage",
        "classifier",
        "feature_representation",
    ]
    _check(
        checks,
        "Comparison has no duplicate conditions",
        int(comparison.duplicated(subset=key_columns).sum()) == 0,
        f"duplicates={int(comparison.duplicated(subset=key_columns).sum())}",
    )

    comparison_keys = {_frame_key(row) for row in comparison.itertuples(index=False)}
    frame_keys = {_frame_key(row) for row in frame.itertuples(index=False)}
    _check(
        checks,
        "Comparison matches every Phase 6 condition",
        comparison_keys == frame_keys,
        (
            f"comparison={len(comparison_keys)}, phase6={len(frame_keys)}, "
            f"missing={sorted(frame_keys - comparison_keys)[:3]}, "
            f"extra={sorted(comparison_keys - frame_keys)[:3]}"
        ),
    )

    by_key = {
        _frame_key(row): row for row in comparison.itertuples(index=False)
    }
    failures: list[str] = []
    for row in frame.itertuples(index=False):
        match = by_key.get(_frame_key(row))
        if match is None:
            continue
        if str(match.self_training_status) != str(row.status):
            failures.append(f"{row.experiment_id}: status mismatch")
        if _as_bool(getattr(match, "numerically_degenerate_fit", False)) != _as_bool(
            getattr(row, "numerically_degenerate_fit", False)
        ):
            failures.append(f"{row.experiment_id}: degenerate flag mismatch")
        both_completed = (
            str(match.supervised_status) == "completed"
            and str(match.self_training_status) == "completed"
        )
        if both_completed:
            expected_accuracy = float(match.self_training_accuracy) - float(
                match.supervised_accuracy
            )
            expected_f1 = float(match.self_training_macro_f1) - float(
                match.supervised_macro_f1
            )
            expected_time = float(
                match.self_training_training_time_seconds
            ) - float(match.supervised_training_time_seconds)
            for column, expected in (
                ("accuracy_improvement", expected_accuracy),
                ("macro_f1_improvement", expected_f1),
                ("training_time_difference_seconds", expected_time),
            ):
                value = float(getattr(match, column))
                if not np.isclose(
                    value, expected, atol=1e-9, rtol=0.0
                ):
                    failures.append(
                        f"{row.experiment_id}: {column}={value!r} != {expected!r}"
                    )
        else:
            for column in (
                "accuracy_improvement",
                "macro_f1_improvement",
                "training_time_difference_seconds",
            ):
                if not pd.isna(getattr(match, column)):
                    failures.append(
                        f"{row.experiment_id}: {column} should be missing"
                    )
    _summarise(
        checks,
        "Comparison improvements equal self-training minus supervised",
        failures,
        len(frame),
    )

    if not COMPARISON_JSON_PATH.is_file():
        _check(
            checks,
            "Comparison JSON is strict and parseable",
            False,
            f"Missing {COMPARISON_JSON_PATH}",
        )
        return
    try:
        with COMPARISON_JSON_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=_reject_json_constant)
        summary = payload.get("summary", {})
        _check(
            checks,
            "Comparison JSON is strict and parseable",
            True,
            str(COMPARISON_JSON_PATH),
        )
        _check(
            checks,
            "Comparison JSON summary matches the CSV",
            int(summary.get("matched_experiments", -1)) == len(comparison)
            and int(summary.get("comparable_experiments", -1))
            == int(
                (
                    (comparison["supervised_status"] == "completed")
                    & (comparison["self_training_status"] == "completed")
                ).sum()
            ),
            f"summary={summary}",
        )
    except Exception as exc:
        _check(
            checks,
            "Comparison JSON is strict and parseable",
            False,
            f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# JSON, report, and figure checks
# ---------------------------------------------------------------------------
def _check_json_payload(
    frame: pd.DataFrame,
    payload: dict[str, Any],
    checks: list[CheckResult],
) -> None:
    metadata = payload.get("metadata", {})
    experiments = payload.get("experiments", [])
    completed = int((frame["status"] == "completed").sum())
    failed = int((frame["status"] == "failed").sum())
    _check(
        checks,
        "Results JSON metadata counts match the CSV",
        int(metadata.get("recorded_experiments", -1)) == len(frame)
        and int(metadata.get("completed_experiments", -1)) == completed
        and int(metadata.get("failed_experiments", -1)) == failed,
        f"recorded={metadata.get('recorded_experiments')}, "
        f"completed={metadata.get('completed_experiments')}, "
        f"failed={metadata.get('failed_experiments')}",
    )
    _check(
        checks,
        "Results JSON contains all 81 experiment records",
        len(experiments) == EXPECTED_EXPERIMENTS,
        f"records={len(experiments)}",
    )
    _check(
        checks,
        "Results JSON records required fields",
        all(
            all(column in record for column in REQUIRED_COLUMNS)
            and "numerically_degenerate_fit" in record
            for record in experiments
        ),
        "required columns present in every JSON record",
    )


def _check_report_and_figures(checks: list[CheckResult]) -> None:
    if REPORT_PATH.is_file():
        text = REPORT_PATH.read_text(encoding="utf-8")
        sections = re.findall(r"(?m)^(\d{1,2})\. ", text)
        _check(
            checks,
            "Phase 6 report contains 18 numbered sections",
            len(sections) == EXPECTED_REPORT_SECTIONS,
            f"sections={len(sections)}",
        )
        _check(
            checks,
            "Phase 6 report covers leakage prevention",
            "Leakage prevention" in text,
            "section present" if "Leakage prevention" in text else "missing",
        )
    else:
        _check(
            checks,
            "Phase 6 report contains 18 numbered sections",
            False,
            f"Missing {REPORT_PATH}",
        )

    figure_files = (
        sorted(path for path in FIGURE_DIR.glob("*.png") if path.is_file())
        if FIGURE_DIR.is_dir()
        else []
    )
    _check(
        checks,
        "All 18 Phase 6 figures exist",
        len(figure_files) == EXPECTED_FIGURES,
        f"figures={len(figure_files)}",
    )
    names = {path.name for path in figure_files}
    missing = [name for name in REQUIRED_FIGURE_NAMES if name not in names]
    _check(
        checks,
        "All eight required figure types are present",
        not missing,
        f"missing={missing}" if missing else "all required names present",
    )
    empty = [path.name for path in figure_files if path.stat().st_size == 0]
    _check(
        checks,
        "Figure files are non-empty",
        not empty,
        f"empty={empty}" if empty else f"{len(figure_files)} non-empty files",
    )


# ---------------------------------------------------------------------------
# Phase 1-5 immutability
# ---------------------------------------------------------------------------
def _check_immutability(checks: list[CheckResult]) -> None:
    if not MANIFEST_PATH.is_file():
        _check(
            checks,
            "Phase 1-5 immutability manifest exists",
            False,
            f"Missing {MANIFEST_PATH}",
        )
        return
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        _check(
            checks,
            "Phase 1-5 immutability manifest is parseable",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        return

    _check(
        checks,
        "Phase 1-5 immutability manifest exists",
        True,
        f"{manifest.get('file_count')} files recorded at "
        f"{manifest.get('created_utc')}",
    )

    failures: list[str] = []
    recorded: set[str] = set()
    for entry in manifest.get("files", []):
        relative = str(entry.get("relative_path"))
        recorded.add(relative)
        path = PROJECT_ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if int(path.stat().st_size) != int(entry.get("size_bytes", -1)):
            failures.append(f"size changed: {relative}")
            continue
        if _sha256_file(path) != str(entry.get("sha256")):
            failures.append(f"hash changed: {relative}")
    _summarise(
        checks,
        "No Phase 1-5 output file was modified or deleted",
        failures,
        max(len(recorded), 1),
    )

    current: set[str] = set()
    for root_name in PHASE15_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                current.add(path.relative_to(PROJECT_ROOT).as_posix())
    unexpected = sorted(current - recorded)
    absent = sorted(recorded - current)
    _check(
        checks,
        "No file was added to or removed from the Phase 1-5 output roots",
        not unexpected and not absent,
        (
            f"current={len(current)}, recorded={len(recorded)}, "
            f"added={unexpected[:5]}, removed={absent[:5]}"
        ),
    )

    roots_recorded = set(manifest.get("roots", []))
    _check(
        checks,
        "Manifest covers every Phase 1-5 output root",
        roots_recorded == set(PHASE15_ROOTS),
        f"roots={sorted(roots_recorded)}",
    )


# ---------------------------------------------------------------------------
# Static leakage checks
# ---------------------------------------------------------------------------
def _check_static_leakage(checks: list[CheckResult]) -> None:
    source_paths = {
        "self_training": PROJECT_ROOT / "src/self_training.py",
        "runner": PROJECT_ROOT / "src/self_training_runner.py",
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    _check(
        checks,
        "Self-training source files exist",
        not missing,
        f"missing={missing}" if missing else "both files present",
    )
    if missing:
        return

    texts = {
        name: path.read_text(encoding="utf-8")
        for name, path in source_paths.items()
    }
    offenders = [
        name
        for name, text in texts.items()
        if "research_unlabelled_labels" in text
    ]
    _check(
        checks,
        "The self-training code never reads stored unlabelled truth",
        not offenders,
        f"offenders={offenders}" if offenders else "no reference found",
    )

    try:
        signature = inspect.signature(run_self_training)
        parameter_names = set(signature.parameters)
        _check(
            checks,
            "run_self_training accepts features, not unlabelled labels",
            not {
                "y_unlabelled",
                "unlabelled_labels",
                "y_true_unlabelled",
            }
            & parameter_names,
            f"parameters={sorted(parameter_names)}",
        )
    except Exception as exc:
        _check(
            checks,
            "run_self_training accepts features, not unlabelled labels",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    call_keywords: list[set[str]] = []
    try:
        tree = ast.parse(texts["runner"])
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                name = getattr(function, "id", None) or getattr(
                    function, "attr", None
                )
                if name == "run_self_training":
                    call_keywords.append(
                        {keyword.arg for keyword in node.keywords if keyword.arg}
                    )
        forbidden = [
            sorted(keywords & {"y_unlabelled", "unlabelled_labels"})
            for keywords in call_keywords
        ]
        _check(
            checks,
            "Runner calls run_self_training without unlabelled truth",
            bool(call_keywords) and all(not item for item in forbidden),
            f"calls={len(call_keywords)}, forbidden={forbidden}",
        )
    except Exception as exc:
        _check(
            checks,
            "Runner calls run_self_training without unlabelled truth",
            False,
            f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_checks(checks: Sequence[CheckResult]) -> None:
    for check in checks:
        label = "PASS" if check.passed else "FAIL"
        print(f"  [{label}] {check.name}: {check.detail}")


def _write_verification_report(
    frame: pd.DataFrame,
    checks_by_dataset: dict[str, list[CheckResult]],
    global_checks: list[CheckResult],
    passed: bool,
) -> Path:
    lines = [
        "Phase 6 self-training verification report",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"Experiments recorded : {len(frame)}",
        f"Completed            : "
        f"{int((frame['status'] == 'completed').sum())}",
        f"Failed               : {int((frame['status'] == 'failed').sum())}",
        (
            "Degenerate fits flagged: "
            f"{int(frame['numerically_degenerate_fit'].map(_as_bool).sum())}"
            if "numerically_degenerate_fit" in frame.columns
            else "Degenerate fits flagged: 0"
        ),
        "",
    ]
    for dataset, checks in checks_by_dataset.items():
        lines.extend([f"[{dataset}]", "-" * 78])
        for check in checks:
            label = "PASS" if check.passed else "FAIL"
            lines.append(f"[{label}] {check.name}: {check.detail}")
        lines.append("")
    lines.extend(["Global checks", "-" * 78])
    for check in global_checks:
        label = "PASS" if check.passed else "FAIL"
        lines.append(f"[{label}] {check.name}: {check.detail}")
    total = sum(len(checks) for checks in checks_by_dataset.values()) + len(
        global_checks
    )
    passed_count = sum(
        check.passed
        for checks in checks_by_dataset.values()
        for check in checks
    ) + sum(check.passed for check in global_checks)
    lines.extend(
        [
            "",
            f"Checks passed : {passed_count} / {total}",
            f"Overall verdict: {'PASS' if passed else 'FAIL'}",
            "",
        ]
    )
    VERIFICATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERIFICATION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return VERIFICATION_REPORT_PATH


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify the Phase 6 semi-supervised self-training results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    datasets = _enabled_datasets(config)
    percentages = _configured_percentages(config)

    print(SEPARATOR)
    print("Phase 6 self-training verification")
    print(SEPARATOR)

    global_checks: list[CheckResult] = []
    frame, payload = _load_artifacts(global_checks)
    if frame is None:
        print("\n[FAIL] Results CSV is missing; nothing else can be verified.")
        return 1

    _check_coverage(frame, datasets, percentages, global_checks)
    _check_metrics_and_counts(frame, config, global_checks)
    _check_failed_rows(frame, global_checks)
    _check_degenerate_flags(frame, global_checks)
    _check_artifact_paths(frame, global_checks)
    if payload is not None:
        _check_json_payload(frame, payload, global_checks)
    _check_comparison(frame, config, global_checks)
    _check_report_and_figures(global_checks)
    _check_static_leakage(global_checks)
    _check_immutability(global_checks)

    try:
        configurations, lda_details = _phase4_metadata()
    except Exception as exc:
        configurations, lda_details = None, None
        _check(
            global_checks,
            "Phase 4 metadata is readable",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    checks_by_dataset: dict[str, list[CheckResult]] = {}
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        checks: list[CheckResult] = []
        _verify_dataset(
            dataset,
            frame,
            config,
            configurations,
            lda_details,
            checks,
        )
        checks_by_dataset[dataset] = checks
        _print_checks(checks)

    print("\n" + SEPARATOR)
    print("GLOBAL CHECKS")
    print(SEPARATOR)
    _print_checks(global_checks)

    all_checks = [
        check for checks in checks_by_dataset.values() for check in checks
    ] + global_checks
    overall_passed = all(check.passed for check in all_checks)
    report_path = _write_verification_report(
        frame, checks_by_dataset, global_checks, overall_passed
    )

    print("\n" + SEPARATOR)
    print(f"Checks passed    : {sum(check.passed for check in all_checks)} / "
          f"{len(all_checks)}")
    print(f"Completed        : "
          f"{int((frame['status'] == 'completed').sum())} / {len(frame)}")
    print(f"Failed           : {int((frame['status'] == 'failed').sum())}")
    if "numerically_degenerate_fit" in frame.columns:
        print(
            "Degenerate fits  : "
            f"{int(frame['numerically_degenerate_fit'].map(_as_bool).sum())} "
            "flagged (reported, not hidden)"
        )
    print(f"Verification report: {report_path}")
    print(f"Overall verdict  : {'PASS' if overall_passed else 'FAIL'}")
    print(SEPARATOR)
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
