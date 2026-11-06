"""Phase 7 analysis: join Phase 5 and Phase 6 results and summarise them.

The analysis is strictly read-only with respect to earlier phases. Every
number in the final report is derived from the Phase 5 and Phase 6 result
tables, their prediction/iteration logs, the Phase 3 split metadata, and the
Phase 4 reduction metadata.

Because each experimental condition was executed exactly once on one fixed
split with one fixed seed, condition-level differences are *not* independent
replicates. The summaries below are descriptive; no p-values or confidence
intervals are computed or implied.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .classifiers import CLASSIFIER_LABELS, CLASSIFIER_NAMES
from .experiment_runner import (
    DATASET_ORDER,
    PROJECT_ROOT,
    REPRESENTATION_ORDER,
    _json_safe,
    load_config,
)

KEY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "labelled_percentage",
    "classifier",
    "feature_representation",
)
METRICS: tuple[str, ...] = ("accuracy", "macro_f1")
METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
}
REPRESENTATION_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "pca": "PCA",
    "lda": "LDA",
}
DATASET_LABELS: dict[str, str] = {
    "breast_cancer": "Breast Cancer Wisconsin (Diagnostic)",
    "mnist": "MNIST",
    "har": "UCI HAR",
}
DATASET_SHORT_LABELS: dict[str, str] = {
    "breast_cancer": "Breast Cancer",
    "mnist": "MNIST",
    "har": "HAR",
}
STOPPING_REASON_LABELS: dict[str, str] = {
    "max_iterations_reached": "Iteration cap reached",
    "no_samples_meeting_confidence_threshold": (
        "No samples met confidence threshold"
    ),
}
TIE_TOLERANCE = 1e-12
EXPECTED_EXPERIMENTS = (
    len(DATASET_ORDER) * 3 * len(REPRESENTATION_ORDER) * len(CLASSIFIER_NAMES)
)

PHASE7_SCRIPTS: tuple[str, ...] = (
    "src/final_analysis.py",
    "src/final_reporting.py",
    "scripts/run_final_analysis.py",
    "scripts/generate_final_report.py",
    "scripts/verify_phase7.py",
)


class AnalysisError(RuntimeError):
    """Raised when the source results are missing or inconsistent."""


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _check(checks: list[dict[str, Any]], name: str, condition: bool, detail: str) -> bool:
    checks.append({"check": name, "passed": bool(condition), "detail": detail})
    return bool(condition)


def _display_dataset(value: str) -> str:
    return DATASET_LABELS.get(value, value)


def _display_representation(value: str) -> str:
    return REPRESENTATION_LABELS.get(value, value)


def _display_classifier(value: str) -> str:
    return CLASSIFIER_LABELS.get(value, value)


def load_sources(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load Phase 5/6 result tables, their JSON siblings, and metadata."""
    settings = config or load_config()
    results = settings.get("results", {})
    phase5_dir = _resolve(
        results.get("classifier_evaluation_dir", "results/classifier_evaluation")
    )
    phase6_dir = _resolve(results.get("self_training_dir", "results/self_training"))
    phase4_dir = _resolve("results/dimensionality_reduction")
    phase3_path = _resolve(
        Path(settings.get("data", {}).get("paths", {}).get("processed_dir", "data/processed"))
        / "preprocessing_splits.json"
    )

    paths = {
        "phase5_csv": phase5_dir / "tables" / "classifier_results.csv",
        "phase5_json": phase5_dir / "tables" / "classifier_results.json",
        "phase6_csv": phase6_dir / "tables" / "self_training_results.csv",
        "phase6_json": phase6_dir / "tables" / "self_training_results.json",
        "phase6_comparison_csv": phase6_dir / "tables" / "supervised_vs_self_training.csv",
        "phase6_comparison_json": phase6_dir / "tables" / "supervised_vs_self_training.json",
        "phase6_settings": phase6_dir / "tables" / "phase6_run_settings.json",
        "phase6_iteration_dir": phase6_dir / "logs" / "iterations",
        "phase6_pseudo_dir": phase6_dir / "logs" / "pseudo_labels",
        "phase6_prediction_dir": phase6_dir / "predictions",
        "phase5_prediction_dir": phase5_dir / "predictions",
        "phase4_metadata": phase4_dir / "metadata" / "reduction_configurations.json",
        "phase4_lda_details": phase4_dir / "metadata" / "lda_details.csv",
        "phase4_pca_variance": phase4_dir / "metadata" / "pca_explained_variance.csv",
        "phase3_splits": phase3_path,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise AnalysisError("Required source artifact(s) missing: " + ", ".join(missing))

    phase5 = pd.read_csv(paths["phase5_csv"])
    phase6 = pd.read_csv(paths["phase6_csv"])
    comparison = pd.read_csv(paths["phase6_comparison_csv"])
    return {
        "paths": paths,
        "phase5": phase5,
        "phase6": phase6,
        "phase6_comparison": comparison,
        "phase5_json": _load_json(paths["phase5_json"]),
        "phase6_json": _load_json(paths["phase6_json"]),
        "comparison_json": _load_json(paths["phase6_comparison_json"]),
        "phase6_settings": _load_json(paths["phase6_settings"]),
        "phase4_metadata": _load_json(paths["phase4_metadata"]),
        "phase3_splits": _load_json(paths["phase3_splits"]),
        "config": settings,
    }


def validate_sources(sources: dict[str, Any]) -> dict[str, Any]:
    """Validate coverage, key matching, and internal consistency."""
    phase5 = sources["phase5"]
    phase6 = sources["phase6"]
    comparison = sources["phase6_comparison"]
    checks: list[dict[str, Any]] = []

    _check(checks, "phase5 row count", len(phase5) == EXPECTED_EXPERIMENTS,
           f"{len(phase5)} rows (expected {EXPECTED_EXPERIMENTS})")
    _check(checks, "phase6 row count", len(phase6) == EXPECTED_EXPERIMENTS,
           f"{len(phase6)} rows (expected {EXPECTED_EXPERIMENTS})")
    _check(checks, "phase6 comparison row count", len(comparison) == EXPECTED_EXPERIMENTS,
           f"{len(comparison)} rows (expected {EXPECTED_EXPERIMENTS})")

    for label, frame in (("phase5", phase5), ("phase6", phase6)):
        missing = [column for column in KEY_COLUMNS if column not in frame.columns]
        _check(checks, f"{label} key columns", not missing, f"missing={missing}")
        duplicated = int(frame.duplicated(subset=list(KEY_COLUMNS)).sum())
        _check(checks, f"{label} unique keys", duplicated == 0,
               f"{duplicated} duplicate key(s)")
        statuses = frame["status"].value_counts().to_dict()
        _check(checks, f"{label} all completed", set(statuses) == {"completed"},
               f"statuses={statuses}")
        errors = int(frame["error_message"].notna().sum()) if "error_message" in frame else 0
        _check(checks, f"{label} no error messages", errors == 0,
               f"{errors} error message(s)")

    keys5 = {
        (row.dataset, int(row.labelled_percentage), row.classifier, row.feature_representation)
        for row in phase5.itertuples()
    }
    keys6 = {
        (row.dataset, int(row.labelled_percentage), row.classifier, row.feature_representation)
        for row in phase6.itertuples()
    }
    _check(checks, "phase5/phase6 identical condition set", keys5 == keys6,
           f"only_phase5={sorted(keys5 - keys6)}; only_phase6={sorted(keys6 - keys5)}")
    expected_keys = {
        (dataset, percentage, classifier, representation)
        for dataset in DATASET_ORDER
        for percentage in (10, 25, 50)
        for representation in REPRESENTATION_ORDER
        for classifier in CLASSIFIER_NAMES
    }
    _check(checks, "full 3x3x3x3 grid present", keys5 == expected_keys,
           f"{len(keys5)} of {len(expected_keys)} expected conditions")

    for metric in METRICS:
        for label, frame in (("phase5", phase5), ("phase6", phase6)):
            values = frame[metric]
            valid = values.notna() & values.between(0.0, 1.0)
            _check(checks, f"{label} {metric} in [0,1]", bool(valid.all()),
                   f"{int((~valid).sum())} invalid value(s)")

    ordered6 = phase6.set_index(list(KEY_COLUMNS)).sort_index()
    ordered5 = phase5.set_index(list(KEY_COLUMNS)).sort_index()
    _check(checks, "phase5/phase6 matched index", ordered5.index.equals(ordered6.index),
           "condition indexes equal")
    for column in ("fit_feature_sha256", "fit_label_sha256"):
        equal = bool((ordered5[column].astype(str) == ordered6[column].astype(str)).all())
        _check(checks, f"matching {column}", equal,
               f"{int((ordered5[column].astype(str) != ordered6[column].astype(str)).sum())} mismatch(es)")
    feature_counts = bool(
        (ordered5["original_feature_count"] == ordered6["original_feature_count"]).all()
        and (ordered5["transformed_feature_count"] == ordered6["transformed_feature_count"]).all()
    )
    _check(checks, "matching feature dimensions", feature_counts,
           "original and transformed feature counts agree")
    labelled_counts = bool(
        (ordered5["n_labelled_train"] == ordered6["n_labelled_train"]).all()
    )
    _check(checks, "matching labelled counts", labelled_counts,
           "n_labelled_train agrees for every condition")

    recomputed = build_final_comparison(phase5, phase6)
    merged_check = comparison.merge(
        recomputed[
            [
                "experiment_id",
                "supervised_accuracy",
                "self_training_accuracy",
                "accuracy_improvement",
                "supervised_macro_f1",
                "self_training_macro_f1",
                "macro_f1_improvement",
            ]
        ],
        on="experiment_id",
        suffixes=("_file", "_recomputed"),
        how="outer",
        validate="one_to_one",
    )
    max_delta = 0.0
    for metric in ("accuracy", "macro_f1"):
        for side in ("supervised", "self_training"):
            left = merged_check[f"{side}_{metric}_file"]
            right = merged_check[f"{side}_{metric}_recomputed"]
            delta = (left - right).abs().max()
            max_delta = max(max_delta, float(delta) if pd.notna(delta) else math.inf)
    _check(checks, "phase6 comparison reproduces from sources", max_delta <= 1e-9,
           f"max absolute difference={max_delta:.3e}")

    phase6_counts = phase6["pseudo_label_count"].astype(int)
    within_pool = bool((phase6_counts <= phase6["n_unlabelled_train"].astype(int)).all())
    _check(checks, "pseudo-label counts within unlabelled pool", within_pool,
           f"max={int(phase6_counts.max())}")
    recomputed_acceptance = (
        phase6_counts / phase6["n_unlabelled_train"].astype(int)
    )
    acceptance_delta = float(
        (recomputed_acceptance - phase6["pseudo_label_acceptance_rate"]).abs().max()
    )
    _check(checks, "acceptance rate definition", acceptance_delta <= 1e-9,
           f"accepted/total unlabelled pool; max difference={acceptance_delta:.3e}")
    iterations_valid = bool(
        phase6["iterations_completed"].between(1, int(sources["phase6_settings"]["max_iterations"])).all()
    )
    _check(checks, "iteration counts within cap", iterations_valid,
           f"max_iterations={sources['phase6_settings']['max_iterations']}")
    reasons = set(phase6["stopping_reason"].astype(str))
    _check(checks, "known stopping reasons",
           reasons <= set(STOPPING_REASON_LABELS),
           f"reasons={sorted(reasons)}")

    iteration_files = sorted(sources["paths"]["phase6_iteration_dir"].glob("*.csv"))
    pseudo_files = sorted(sources["paths"]["phase6_pseudo_dir"].glob("*.csv"))
    prediction6 = sorted(sources["paths"]["phase6_prediction_dir"].glob("*.csv"))
    prediction5 = sorted(sources["paths"]["phase5_prediction_dir"].glob("*.csv"))
    _check(checks, "81 iteration logs", len(iteration_files) == EXPECTED_EXPERIMENTS,
           f"{len(iteration_files)} files")
    _check(checks, "81 pseudo-label audits", len(pseudo_files) == EXPECTED_EXPERIMENTS,
           f"{len(pseudo_files)} files")
    _check(checks, "81 phase6 prediction files", len(prediction6) == EXPECTED_EXPERIMENTS,
           f"{len(prediction6)} files")
    _check(checks, "81 phase5 prediction files", len(prediction5) == EXPECTED_EXPERIMENTS,
           f"{len(prediction5)} files")

    phase5_json = sources["phase5_json"]
    phase6_json = sources["phase6_json"]
    _check(checks, "phase5 JSON experiment count",
           len(phase5_json.get("experiments", [])) == EXPECTED_EXPERIMENTS,
           f"{len(phase5_json.get('experiments', []))} records")
    _check(checks, "phase6 JSON experiment count",
           len(phase6_json.get("experiments", [])) == EXPECTED_EXPERIMENTS,
           f"{len(phase6_json.get('experiments', []))} records")
    _check(checks, "phase6 canonical settings",
           int(sources["phase6_settings"]["max_iterations"]) == 3
           and float(sources["phase6_settings"]["confidence_threshold"]) == 0.9,
           f"max_iterations={sources['phase6_settings']['max_iterations']}, "
           f"threshold={sources['phase6_settings']['confidence_threshold']}")

    failures = [check for check in checks if not check["passed"]]
    if failures:
        details = "; ".join(f"{item['check']}: {item['detail']}" for item in failures)
        raise AnalysisError("Source validation failed: " + details)
    return {
        "expected_experiments": EXPECTED_EXPERIMENTS,
        "matched_experiments": int(len(recomputed)),
        "checks": checks,
        "passed": len(checks),
        "failed": 0,
    }


def build_final_comparison(phase5: pd.DataFrame, phase6: pd.DataFrame) -> pd.DataFrame:
    """Join the two phases on the full experimental condition."""
    left = phase5.copy()
    right = phase6.copy()
    merged = left.merge(
        right,
        on=list(KEY_COLUMNS),
        how="inner",
        suffixes=("_p5", "_p6"),
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_EXPERIMENTS:
        raise AnalysisError(
            f"Expected {EXPECTED_EXPERIMENTS} matched conditions, found {len(merged)}."
        )

    frame = pd.DataFrame(
        {
            "experiment_id": merged["experiment_id_p5"].astype(str),
            "dataset": merged["dataset"],
            "dataset_label": merged["dataset"].map(DATASET_LABELS),
            "labelled_percentage": merged["labelled_percentage"].astype(int),
            "classifier": merged["classifier"],
            "classifier_label": merged["classifier"].map(CLASSIFIER_LABELS),
            "feature_representation": merged["feature_representation"],
            "representation_label": merged["feature_representation"].map(
                REPRESENTATION_LABELS
            ),
            "original_feature_count": merged["original_feature_count_p5"].astype(int),
            "transformed_feature_count": merged["transformed_feature_count_p5"].astype(int),
            "n_labelled_train": merged["n_labelled_train_p5"].astype(int),
            "n_unlabelled_train": merged["n_unlabelled_train"].astype(int),
            "n_test": merged["n_test"].astype(int),
            "supervised_accuracy": merged["accuracy_p5"],
            "self_training_accuracy": merged["accuracy_p6"],
            "accuracy_improvement": merged["accuracy_p6"] - merged["accuracy_p5"],
            "supervised_macro_f1": merged["macro_f1_p5"],
            "self_training_macro_f1": merged["macro_f1_p6"],
            "macro_f1_improvement": merged["macro_f1_p6"] - merged["macro_f1_p5"],
            "supervised_training_time_seconds": merged["training_time_seconds_p5"],
            "self_training_training_time_seconds": merged["training_time_seconds_p6"],
            "training_time_difference_seconds": (
                merged["training_time_seconds_p6"]
                - merged["training_time_seconds_p5"]
            ),
            "self_training_prediction_time_seconds": merged["prediction_time_seconds"],
            "pseudo_label_count": merged["pseudo_label_count"].astype(int),
            "remaining_unlabelled_count": merged["remaining_unlabelled_count"].astype(int),
            "pseudo_label_acceptance_rate": merged["pseudo_label_acceptance_rate"],
            "iterations_completed": merged["iterations_completed"].astype(int),
            "stopping_reason": merged["stopping_reason"],
            "confidence_method": merged["confidence_method"],
            "numerically_degenerate_fit": merged["numerically_degenerate_fit"].astype(bool),
            "phase5_warning": merged["warning_message_p5"],
            "phase6_warning": merged["warning_message_p6"],
            "phase5_prediction_file": merged["prediction_file_p5"],
            "phase6_prediction_file": merged["prediction_file_p6"],
            "phase6_iteration_log_file": merged["iteration_log_file"],
            "fit_feature_sha256": merged["fit_feature_sha256_p5"],
            "fit_label_sha256": merged["fit_label_sha256_p5"],
            "unlabelled_feature_sha256": merged["unlabelled_feature_sha256"],
        }
    )
    frame["training_time_ratio"] = np.where(
        frame["supervised_training_time_seconds"] > 0,
        frame["self_training_training_time_seconds"]
        / frame["supervised_training_time_seconds"],
        np.nan,
    )
    frame["supervised_prediction_time_seconds"] = np.nan

    dataset_rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    representation_rank = {
        name: index for index, name in enumerate(REPRESENTATION_ORDER)
    }
    classifier_rank = {name: index for index, name in enumerate(CLASSIFIER_NAMES)}
    frame["_dataset"] = frame["dataset"].map(dataset_rank)
    frame["_representation"] = frame["feature_representation"].map(representation_rank)
    frame["_classifier"] = frame["classifier"].map(classifier_rank)
    frame = frame.sort_values(
        ["_dataset", "labelled_percentage", "_representation", "_classifier"]
    ).drop(columns=["_dataset", "_representation", "_classifier"])
    return frame.reset_index(drop=True)


def paired_statistics(
    frame: pd.DataFrame,
    metric: str,
    label: str = "all conditions",
) -> dict[str, Any]:
    """Descriptive paired statistics for one metric."""
    difference = frame[f"{metric}_improvement"].dropna().astype(float)
    if difference.empty:
        return {"group": label, "n": 0}
    sd = float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
    mean = float(difference.mean())
    return {
        "group": label,
        "n": int(len(difference)),
        "mean_supervised": float(frame[f"supervised_{metric}"].mean()),
        "mean_self_training": float(frame[f"self_training_{metric}"].mean()),
        "mean_difference": mean,
        "median_difference": float(difference.median()),
        "sd_difference": sd,
        "min_difference": float(difference.min()),
        "q1_difference": float(difference.quantile(0.25)),
        "q3_difference": float(difference.quantile(0.75)),
        "max_difference": float(difference.max()),
        "improved": int((difference > TIE_TOLERANCE).sum()),
        "unchanged": int((difference.abs() <= TIE_TOLERANCE).sum()),
        "regressed": int((difference < -TIE_TOLERANCE).sum()),
        "descriptive_cohens_dz": (mean / sd) if sd > 0 else None,
    }


def grouped_statistics(
    frame: pd.DataFrame,
    factors: Iterable[str],
    metric: str,
) -> list[dict[str, Any]]:
    """Descriptive paired statistics for every level of the given factors."""
    factors = list(factors)
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(factors, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {
            "factors": {factor: value for factor, value in zip(factors, keys)},
        }
        record.update(paired_statistics(group, metric, label="__".join(map(str, keys))))
        records.append(record)
    return records


def _runtime_summary(frame: pd.DataFrame) -> dict[str, Any]:
    p5_total = float(frame["supervised_training_time_seconds"].sum())
    p6_total = float(frame["self_training_training_time_seconds"].sum())
    p6_prediction_total = float(frame["self_training_prediction_time_seconds"].sum())

    by_dataset: list[dict[str, Any]] = []
    for dataset, group in frame.groupby("dataset", sort=False):
        by_dataset.append(
            {
                "dataset": dataset,
                "supervised_training_seconds": float(
                    group["supervised_training_time_seconds"].sum()
                ),
                "self_training_training_seconds": float(
                    group["self_training_training_time_seconds"].sum()
                ),
                "self_training_prediction_seconds": float(
                    group["self_training_prediction_time_seconds"].sum()
                ),
                "training_time_ratio": float(
                    group["self_training_training_time_seconds"].sum()
                    / group["supervised_training_time_seconds"].sum()
                ),
            }
        )
    by_classifier: list[dict[str, Any]] = []
    for dataset, dataset_group in frame.groupby("dataset", sort=False):
        for classifier, group in dataset_group.groupby("classifier", sort=False):
            by_classifier.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "supervised_training_seconds": float(
                        group["supervised_training_time_seconds"].sum()
                    ),
                    "self_training_training_seconds": float(
                        group["self_training_training_time_seconds"].sum()
                    ),
                    "self_training_prediction_seconds": float(
                        group["self_training_prediction_time_seconds"].sum()
                    ),
                }
            )
    mnist_svm = frame[
        (frame["dataset"] == "mnist") & (frame["classifier"] == "svm")
    ]
    return {
        "supervised_training_seconds_total": p5_total,
        "self_training_training_seconds_total": p6_total,
        "self_training_prediction_seconds_total": p6_prediction_total,
        "supervised_prediction_time_available": False,
        "supervised_prediction_time_note": (
            "Phase 5 did not log prediction time; only Phase 6 prediction time "
            "can be reported."
        ),
        "training_time_ratio_total": p6_total / p5_total,
        "mnist_svm_training_seconds": float(
            mnist_svm["self_training_training_time_seconds"].sum()
        ),
        "mnist_svm_share_of_self_training_training": float(
            mnist_svm["self_training_training_time_seconds"].sum() / p6_total
        ),
        "median_condition_training_time_ratio": float(
            frame["training_time_ratio"].median()
        ),
        "min_condition_training_time_ratio": float(
            frame["training_time_ratio"].min()
        ),
        "max_condition_training_time_ratio": float(
            frame["training_time_ratio"].max()
        ),
        "by_dataset": by_dataset,
        "by_dataset_classifier": by_classifier,
    }


def _pseudo_label_summary(
    phase6: pd.DataFrame,
    iteration_dir: Path,
) -> dict[str, Any]:
    iteration_frames = []
    for path in sorted(iteration_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["experiment_id"] = path.stem
        iteration_frames.append(frame)
    iterations = pd.concat(iteration_frames, ignore_index=True)

    rounds = []
    for round_number, group in iterations.groupby("iteration", sort=True):
        accepted = group["pseudo_labels_added"].astype(int)
        weighted_confidence = float(
            (group["mean_accepted_confidence"] * accepted).sum() / accepted.sum()
        ) if accepted.sum() else None
        rounds.append(
            {
                "iteration": int(round_number),
                "experiments_with_round": int(len(group)),
                "pseudo_labels_added": int(accepted.sum()),
                "mean_pseudo_labels_added": float(accepted.mean()),
                "weighted_mean_accepted_confidence": weighted_confidence,
                "mean_training_set_size": float(group["training_set_size"].mean()),
            }
        )

    by_dataset: list[dict[str, Any]] = []
    for dataset, group in phase6.groupby("dataset", sort=False):
        by_dataset.append(
            {
                "dataset": dataset,
                "n": int(len(group)),
                "total_pseudo_labels": int(group["pseudo_label_count"].sum()),
                "mean_pseudo_labels": float(group["pseudo_label_count"].mean()),
                "mean_acceptance_rate": float(
                    group["pseudo_label_acceptance_rate"].mean()
                ),
                "mean_iterations": float(group["iterations_completed"].mean()),
                "iteration_cap_hits": int(
                    (group["stopping_reason"] == "max_iterations_reached").sum()
                ),
                "threshold_stops": int(
                    (
                        group["stopping_reason"]
                        == "no_samples_meeting_confidence_threshold"
                    ).sum()
                ),
            }
        )
    by_representation: list[dict[str, Any]] = []
    for dataset, dataset_group in phase6.groupby("dataset", sort=False):
        for representation, group in dataset_group.groupby(
            "feature_representation", sort=False
        ):
            by_representation.append(
                {
                    "dataset": dataset,
                    "feature_representation": representation,
                    "mean_pseudo_labels": float(group["pseudo_label_count"].mean()),
                    "mean_acceptance_rate": float(
                        group["pseudo_label_acceptance_rate"].mean()
                    ),
                    "mean_iterations": float(group["iterations_completed"].mean()),
                }
            )
    by_proportion: list[dict[str, Any]] = []
    for proportion, group in phase6.groupby("labelled_percentage", sort=True):
        by_proportion.append(
            {
                "labelled_percentage": int(proportion),
                "mean_pseudo_labels": float(group["pseudo_label_count"].mean()),
                "mean_acceptance_rate": float(
                    group["pseudo_label_acceptance_rate"].mean()
                ),
                "mean_iterations": float(group["iterations_completed"].mean()),
            }
        )

    return {
        "total_pseudo_labels": int(phase6["pseudo_label_count"].sum()),
        "mean_pseudo_labels_per_experiment": float(
            phase6["pseudo_label_count"].mean()
        ),
        "median_pseudo_labels_per_experiment": float(
            phase6["pseudo_label_count"].median()
        ),
        "total_unlabelled_available": int(phase6["n_unlabelled_train"].sum()),
        "overall_acceptance_rate": float(
            phase6["pseudo_label_count"].sum()
            / phase6["n_unlabelled_train"].sum()
        ),
        "mean_experiment_acceptance_rate": float(
            phase6["pseudo_label_acceptance_rate"].mean()
        ),
        "min_acceptance_rate": float(
            phase6["pseudo_label_acceptance_rate"].min()
        ),
        "max_acceptance_rate": float(
            phase6["pseudo_label_acceptance_rate"].max()
        ),
        "mean_iterations": float(phase6["iterations_completed"].mean()),
        "median_iterations": float(phase6["iterations_completed"].median()),
        "iteration_distribution": {
            str(int(key)): int(value)
            for key, value in phase6["iterations_completed"]
            .value_counts()
            .sort_index()
            .items()
        },
        "iteration_cap_hits": int(
            (phase6["stopping_reason"] == "max_iterations_reached").sum()
        ),
        "threshold_stops": int(
            (
                phase6["stopping_reason"]
                == "no_samples_meeting_confidence_threshold"
            ).sum()
        ),
        "stopping_reason_counts": {
            str(key): int(value)
            for key, value in phase6["stopping_reason"].value_counts().items()
        },
        "rounds": rounds,
        "by_dataset": by_dataset,
        "by_dataset_representation": by_representation,
        "by_labelled_percentage": by_proportion,
    }


def _warning_audit(
    phase5: pd.DataFrame,
    phase6: pd.DataFrame,
) -> dict[str, Any]:
    def classify(message: Any) -> str | None:
        if not isinstance(message, str) or not message:
            return None
        if message.startswith("ConvergenceWarning"):
            return "ConvergenceWarning"
        if message.startswith("NumericalStabilityWarning"):
            return "NumericalStabilityWarning"
        return "OtherWarning"

    audit: dict[str, Any] = {}
    for label, frame in (("phase5", phase5), ("phase6", phase6)):
        records = frame.copy()
        records["warning_type"] = records["warning_message"].map(classify)
        warning_rows = records[records["warning_type"].notna()]
        by_dataset = {
            str(key): int(value)
            for key, value in warning_rows.groupby("dataset").size().items()
        }
        by_type = {
            str(key): int(value)
            for key, value in warning_rows.groupby("warning_type").size().items()
        }
        audit[label] = {
            "experiments_with_warnings": int(len(warning_rows)),
            "experiments_without_warnings": int(len(records) - len(warning_rows)),
            "by_type": by_type,
            "by_dataset": by_dataset,
            "experiments": [
                {
                    "experiment_id": row.experiment_id,
                    "warning_type": row.warning_type,
                    "warning_message": row.warning_message,
                }
                for row in warning_rows.itertuples()
            ],
        }

    degenerate = phase6[phase6["numerically_degenerate_fit"].astype(bool)]
    audit["degenerate_fits"] = {
        "count": int(len(degenerate)),
        "experiments": [
            {
                "experiment_id": row.experiment_id,
                "accuracy": float(row.accuracy),
                "macro_f1": float(row.macro_f1),
                "fit_diagnostic": row.fit_diagnostic,
                "stopping_reason": row.stopping_reason,
            }
            for row in degenerate.itertuples()
        ],
    }
    return audit


def _design_summary(sources: dict[str, Any]) -> dict[str, Any]:
    splits = sources["phase3_splits"]
    reduction = sources["phase4_metadata"]
    datasets: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        info = splits["datasets"][dataset]
        lda = reduction["datasets"][dataset]["lda"]
        datasets.append(
            {
                "dataset": dataset,
                "n_original": int(info["n_original"]),
                "n_train": int(info["n_train"]),
                "n_test": int(info["n_test"]),
                "n_features": int(info["n_features"]),
                "strategy": info["strategy"],
                "pca_components": int(reduction["datasets"][dataset]["pca_features"]),
                "pca_variance_retained": float(
                    reduction["datasets"][dataset]["pca_variance_retained"]
                ),
                "lda_components": {
                    str(int(float(proportion) * 100)): int(details["components"])
                    for proportion, details in lda.items()
                },
                "labelled_settings": {
                    str(int(float(proportion) * 100)): {
                        "n_labelled": int(details["n_labelled"]),
                        "n_unlabelled": int(details["n_unlabelled"]),
                        "actual_proportion": float(details["actual_proportion"]),
                    }
                    for proportion, details in info["labelled_settings"].items()
                },
            }
        )
    return {
        "random_seed": int(splits["random_seed"]),
        "test_size": float(splits["test_size"]),
        "labelled_proportions": [int(round(value * 100)) for value in splits["labelled_proportions"]],
        "confidence_threshold": float(sources["phase6_settings"]["confidence_threshold"]),
        "max_iterations": int(sources["phase6_settings"]["max_iterations"]),
        "minimum_new_pseudo_labels": int(
            sources["phase6_settings"]["minimum_new_pseudo_labels"]
        ),
        "datasets": datasets,
    }


def _best_configurations(frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for dataset, group in frame.groupby("dataset", sort=False):
        best_self = group.loc[group[f"self_training_{metric}"].idxmax()]
        best_supervised = group.loc[group[f"supervised_{metric}"].idxmax()]
        payload[dataset] = {
            "supervised": {
                "experiment_id": best_supervised["experiment_id"],
                "labelled_percentage": int(best_supervised["labelled_percentage"]),
                "classifier": best_supervised["classifier"],
                "feature_representation": best_supervised["feature_representation"],
                "score": float(best_supervised[f"supervised_{metric}"]),
                "accuracy": float(best_supervised["supervised_accuracy"]),
                "macro_f1": float(best_supervised["supervised_macro_f1"]),
                "training_time_seconds": float(
                    best_supervised["supervised_training_time_seconds"]
                ),
            },
            "self_training": {
                "experiment_id": best_self["experiment_id"],
                "labelled_percentage": int(best_self["labelled_percentage"]),
                "classifier": best_self["classifier"],
                "feature_representation": best_self["feature_representation"],
                "score": float(best_self[f"self_training_{metric}"]),
                "accuracy": float(best_self["self_training_accuracy"]),
                "macro_f1": float(best_self["self_training_macro_f1"]),
                "training_time_seconds": float(
                    best_self["self_training_training_time_seconds"]
                ),
                "prediction_time_seconds": float(
                    best_self["self_training_prediction_time_seconds"]
                ),
                "pseudo_label_count": int(best_self["pseudo_label_count"]),
                "iterations_completed": int(best_self["iterations_completed"]),
                "stopping_reason": best_self["stopping_reason"],
            },
        }
    return payload


def _top_changes(frame: pd.DataFrame, metric: str, count: int = 10) -> dict[str, Any]:
    columns = [
        "experiment_id",
        "dataset",
        "labelled_percentage",
        "classifier",
        "feature_representation",
        f"supervised_{metric}",
        f"self_training_{metric}",
        f"{metric}_improvement",
        "pseudo_label_count",
        "iterations_completed",
        "stopping_reason",
        "numerically_degenerate_fit",
    ]
    ordered = frame.sort_values(f"{metric}_improvement", ascending=False)
    return {
        "largest_improvements": ordered.head(count)[columns].to_dict(orient="records"),
        "largest_regressions": ordered.tail(count)
        .sort_values(f"{metric}_improvement")[columns]
        .to_dict(orient="records"),
    }


def build_analysis(sources: dict[str, Any]) -> dict[str, Any]:
    """Build the complete Phase 7 analysis payload."""
    validation = validate_sources(sources)
    frame = build_final_comparison(sources["phase5"], sources["phase6"])
    nondegenerate = frame[~frame["numerically_degenerate_fit"]]

    overall = {
        metric: {
            "all_conditions": paired_statistics(frame, metric, "all 81 conditions"),
            "excluding_degenerate_fits": paired_statistics(
                nondegenerate, metric, "78 non-degenerate conditions"
            ),
        }
        for metric in METRICS
    }
    factor_summaries: dict[str, Any] = {}
    for metric in METRICS:
        factor_summaries[metric] = {
            "dataset": grouped_statistics(frame, ["dataset"], metric),
            "labelled_percentage": grouped_statistics(
                frame, ["labelled_percentage"], metric
            ),
            "feature_representation": grouped_statistics(
                frame, ["feature_representation"], metric
            ),
            "classifier": grouped_statistics(frame, ["classifier"], metric),
            "dataset_x_representation": grouped_statistics(
                frame, ["dataset", "feature_representation"], metric
            ),
            "dataset_x_classifier": grouped_statistics(
                frame, ["dataset", "classifier"], metric
            ),
            "dataset_x_labelled_percentage": grouped_statistics(
                frame, ["dataset", "labelled_percentage"], metric
            ),
            "representation_x_labelled_percentage": grouped_statistics(
                frame,
                ["feature_representation", "labelled_percentage"],
                metric,
            ),
            "classifier_x_labelled_percentage": grouped_statistics(
                frame, ["classifier", "labelled_percentage"], metric
            ),
        }

    source_paths = sources["paths"]
    provenance = {
        "phase5_results": {
            "path": source_paths["phase5_csv"].relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(source_paths["phase5_csv"]),
            "generated_utc": sources["phase5_json"].get("generated_utc"),
        },
        "phase6_results": {
            "path": source_paths["phase6_csv"].relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(source_paths["phase6_csv"]),
            "generated_utc": sources["phase6_json"].get("generated_utc"),
        },
        "phase6_comparison": {
            "path": source_paths["phase6_comparison_csv"]
            .relative_to(PROJECT_ROOT)
            .as_posix(),
            "sha256": sha256_file(source_paths["phase6_comparison_csv"]),
            "generated_utc": sources["comparison_json"].get("generated_utc"),
        },
        "phase3_splits": {
            "path": source_paths["phase3_splits"].relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(source_paths["phase3_splits"]),
        },
        "phase4_reduction": {
            "path": source_paths["phase4_metadata"]
            .relative_to(PROJECT_ROOT)
            .as_posix(),
            "sha256": sha256_file(source_paths["phase4_metadata"]),
        },
        "phase6_settings": {
            "path": source_paths["phase6_settings"]
            .relative_to(PROJECT_ROOT)
            .as_posix(),
            "sha256": sha256_file(source_paths["phase6_settings"]),
        },
    }

    payload = {
        "phase": 7,
        "title": (
            "Impact of Dimensionality Reduction on the Performance of Subsequent "
            "Supervised Classifiers in a Semi-Supervised Setting"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_scope": (
            "Descriptive comparison of 81 matched Phase 5 supervised and Phase 6 "
            "self-training conditions from one fixed split and one fixed seed."
        ),
        "statistical_policy": {
            "inferential_tests_performed": False,
            "reason": (
                "Each condition was run once on the same fixed split with seed 42. "
                "Conditions share training rows, test rows, and transformations and "
                "are therefore not independent replicates. P-values or confidence "
                "intervals computed across conditions would overstate the evidence. "
                "Only descriptive paired differences and effect sizes are reported."
            ),
            "difference_definition": "self-training minus supervised",
            "tie_tolerance": TIE_TOLERANCE,
        },
        "validation": validation,
        "provenance": provenance,
        "design": _design_summary(sources),
        "coverage": {
            "datasets": list(DATASET_ORDER),
            "labelled_percentages": [10, 25, 50],
            "feature_representations": list(REPRESENTATION_ORDER),
            "classifiers": list(CLASSIFIER_NAMES),
            "phase5_experiments": int(len(sources["phase5"])),
            "phase6_experiments": int(len(sources["phase6"])),
            "matched_experiments": int(len(frame)),
        },
        "overall": overall,
        "factor_summaries": factor_summaries,
        "best_configurations": {
            "accuracy": _best_configurations(frame, "accuracy"),
            "macro_f1": _best_configurations(frame, "macro_f1"),
        },
        "top_changes": {
            "accuracy": _top_changes(frame, "accuracy"),
            "macro_f1": _top_changes(frame, "macro_f1"),
        },
        "runtime": _runtime_summary(frame),
        "pseudo_labels": _pseudo_label_summary(
            sources["phase6"], source_paths["phase6_iteration_dir"]
        ),
        "warnings": _warning_audit(sources["phase5"], sources["phase6"]),
        "notes": [
            "Phase 5 did not record prediction time, so supervised/self-training "
            "prediction-time comparisons are unavailable.",
            "The three numerically degenerate MNIST LDA + Logistic Regression "
            "self-training fits remain in every primary statistic and are also "
            "reported separately.",
            "The canonical Phase 6 run uses max_iterations=3; 68 of 81 "
            "experiments stopped at that cap.",
            "Pseudo-label acceptance is defined as accepted pseudo-labels divided "
            "by the original unlabelled pool size.",
        ],
    }
    return payload


def write_comparison_outputs(
    frame: pd.DataFrame,
    payload: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, Path]:
    directory = _resolve(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "final_results_comparison.csv"
    json_path = directory / "final_results_comparison.json"
    frame.to_csv(csv_path, index=False)
    output_payload = dict(payload)
    output_payload["comparison"] = frame.to_dict(orient="records")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(output_payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return {"csv": csv_path, "json": json_path}


# ---------------------------------------------------------------------------
# Immutability snapshot for Phase 1-6 artifacts
# ---------------------------------------------------------------------------
SNAPSHOT_ROOTS: tuple[str, ...] = (
    "results",
    "data",
    "config",
    "src",
    "scripts",
    "dashboard",
    "notebooks",
    "tests",
)
SNAPSHOT_TOP_FILES: tuple[str, ...] = (
    "README.md",
    "requirements.txt",
    ".gitignore",
)
SNAPSHOT_EXCLUDED_PARTS: tuple[str, ...] = (
    "__pycache__",
    ".pytest_cache",
)
SNAPSHOT_EXCLUDED_PATHS: tuple[str, ...] = (
    "results/final_report",
    *PHASE7_SCRIPTS,
)


def _snapshot_excluded(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    if any(part in normalized.split("/") for part in SNAPSHOT_EXCLUDED_PARTS):
        return True
    return any(
        normalized == excluded or normalized.startswith(excluded + "/")
        for excluded in SNAPSHOT_EXCLUDED_PATHS
    )


def write_immutability_snapshot(
    output_path: str | Path,
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Hash every Phase 1-6 artifact before Phase 7 outputs are written."""
    root_path = Path(root)
    files: dict[str, Any] = {}
    for top in SNAPSHOT_TOP_FILES:
        path = root_path / top
        if path.is_file():
            relative = path.relative_to(root_path).as_posix()
            files[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
    for directory in SNAPSHOT_ROOTS:
        base = root_path / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root_path).as_posix()
            if _snapshot_excluded(relative):
                continue
            files[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
    payload = {
        "purpose": "Phase 1-6 artifact immutability baseline for Phase 7",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope_roots": list(SNAPSHOT_ROOTS),
        "scope_top_files": list(SNAPSHOT_TOP_FILES),
        "excluded": list(SNAPSHOT_EXCLUDED_PATHS),
        "file_count": len(files),
        "total_size_bytes": int(sum(item["size_bytes"] for item in files.values())),
        "files": files,
    }
    target = _resolve(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload
