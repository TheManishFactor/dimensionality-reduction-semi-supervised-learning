"""Phase 6 semi-supervised self-training orchestration.

Experiment grid
---------------
``3 datasets x 3 labelled proportions x 3 feature representations x
3 classifiers = 81 self-training experiments``.

Every experiment reuses the fixed Phase 3 splits and the Phase 4
transformations exactly as in Phase 5. The runner receives only labelled
feature rows, labelled targets, and unlabelled feature rows; unlabelled truth
is never read. Test rows are used only for the final prediction and metrics.

Outputs are written under ``results/self_training/``. Results are persisted
after each experiment so a long run can be inspected or resumed with
``--resume`` without losing completed work.

Run from the project root:

    python scripts/run_self_training.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .classifiers import (
    CLASSIFIER_LABELS,
    CLASSIFIER_NAMES,
    build_calibrated_svm,
    build_classifier,
    build_svm,
    list_classifiers,
)
from .comparison import (
    build_comparison,
    load_phase5_results,
    save_comparison,
    summarise_comparison,
)
from .dimensionality_reduction import run_dimensionality_reduction
from .evaluation import (
    compute_acceptance_rate,
    evaluate_predictions,
    save_metrics,
    to_json_records,
)
from .experiment_runner import (
    DATASET_LABELS,
    DATASET_ORDER,
    LABELLED_PERCENTAGES,
    PROJECT_ROOT,
    REPRESENTATION_ORDER,
    _enabled_datasets,
    _json_safe,
    fingerprint_features,
    fingerprint_labels,
    get_self_training_views,
    load_config,
    set_global_seed,
)
from .preprocessing import EXPECTED_FEATURE_COUNTS, PreprocessedDataset, preprocess_dataset
from .self_training import (
    PSEUDO_LABEL_SOURCE,
    SelfTrainingConfig,
    SelfTrainingResult,
    run_self_training,
)
from .visualization import build_self_training_figure_report

DEFAULT_OUTPUT_DIR = Path("results/self_training")
REQUIRED_SECTIONS = (
    "project",
    "data",
    "dimensionality_reduction",
    "classifiers",
    "self_training",
    "evaluation",
    "results",
)

RESULT_COLUMNS: tuple[str, ...] = (
    "experiment_id",
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
    "n_labelled_train",
    "n_unlabelled_train",
    "iterations_completed",
    "pseudo_label_acceptance_rate",
    "stopping_reason",
    "confidence_method",
    "status",
    "numerically_degenerate_fit",
    "fit_diagnostic",
    "fit_feature_sha256",
    "fit_label_sha256",
    "unlabelled_feature_sha256",
    "prediction_file",
    "pseudo_label_file",
    "iteration_log_file",
    "warning_message",
    "error_message",
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

PSEUDO_AUDIT_COLUMNS: tuple[str, ...] = (
    "unlabelled_position",
    "train_index",
    "dataset_index",
    "pseudo_label",
    "confidence",
    "iteration",
    "pseudo_label_source",
)

CONFIDENCE_METHODS: dict[str, str] = {
    "logistic_regression": "predict_proba (maximum class probability)",
    "svm": (
        "sigmoid-calibrated predict_proba (maximum class probability) from "
        "LinearSVC wrapped with Platt scaling"
    ),
    "knn": "predict_proba (maximum class probability)",
}

FINAL_MODEL_DESCRIPTIONS: dict[str, str] = {
    "logistic_regression": "Logistic Regression fitted on labelled + pseudo-labelled rows",
    "svm": "LinearSVC fitted on labelled + pseudo-labelled rows (no calibration layer)",
    "knn": "KNN fitted on labelled + pseudo-labelled rows",
}


@dataclass
class SelfTrainingRecord:
    """One completed or failed Phase 6 experiment."""

    dataset: str
    labelled_percentage: int
    classifier: str
    feature_representation: str
    original_feature_count: int | None
    transformed_feature_count: int | None
    accuracy: float | None
    macro_f1: float | None
    training_time_seconds: float | None
    prediction_time_seconds: float | None
    pseudo_label_count: int | None
    remaining_unlabelled_count: int | None
    n_labelled_train: int
    n_unlabelled_train: int
    iterations_completed: int | None
    pseudo_label_acceptance_rate: float | None
    stopping_reason: str | None
    confidence_method: str
    status: str
    numerically_degenerate_fit: bool = False
    fit_diagnostic: str | None = None
    fit_feature_sha256: str | None = None
    fit_label_sha256: str | None = None
    unlabelled_feature_sha256: str | None = None
    prediction_file: str | None = None
    pseudo_label_file: str | None = None
    iteration_log_file: str | None = None
    warning_message: str | None = None
    error_message: str | None = None

    @property
    def experiment_id(self) -> str:
        return (
            f"{self.dataset}__labelled_{self.labelled_percentage:02d}__"
            f"{self.feature_representation}__{self.classifier}"
        )

    def to_row(self) -> dict[str, Any]:
        row = {
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "labelled_percentage": self.labelled_percentage,
            "classifier": self.classifier,
            "feature_representation": self.feature_representation,
            "original_feature_count": self.original_feature_count,
            "transformed_feature_count": self.transformed_feature_count,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "training_time_seconds": self.training_time_seconds,
            "prediction_time_seconds": self.prediction_time_seconds,
            "pseudo_label_count": self.pseudo_label_count,
            "remaining_unlabelled_count": self.remaining_unlabelled_count,
            "n_labelled_train": self.n_labelled_train,
            "n_unlabelled_train": self.n_unlabelled_train,
            "iterations_completed": self.iterations_completed,
            "pseudo_label_acceptance_rate": self.pseudo_label_acceptance_rate,
            "stopping_reason": self.stopping_reason,
            "confidence_method": self.confidence_method,
            "status": self.status,
            "numerically_degenerate_fit": self.numerically_degenerate_fit,
            "fit_diagnostic": self.fit_diagnostic,
            "fit_feature_sha256": self.fit_feature_sha256,
            "fit_label_sha256": self.fit_label_sha256,
            "unlabelled_feature_sha256": self.unlabelled_feature_sha256,
            "prediction_file": self.prediction_file,
            "pseudo_label_file": self.pseudo_label_file,
            "iteration_log_file": self.iteration_log_file,
            "warning_message": self.warning_message,
            "error_message": self.error_message,
        }
        return {column: row[column] for column in RESULT_COLUMNS}


# ---------------------------------------------------------------------------
# Paths, configuration, and provenance
# ---------------------------------------------------------------------------
def self_training_layout(config: dict[str, Any]) -> dict[str, Path]:
    base = Path(
        config.get("results", {}).get(
            "self_training_dir", str(DEFAULT_OUTPUT_DIR)
        )
    )
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    layout = {
        "base": base,
        "tables": base / "tables",
        "reports": base / "reports",
        "predictions": base / "predictions",
        "logs": base / "logs",
        "iterations": base / "logs" / "iterations",
        "pseudo_labels": base / "logs" / "pseudo_labels",
        "figures": base / "figures",
    }
    for directory in layout.values():
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_phase15_manifest(layout: dict[str, Path]) -> Path:
    """Record SHA-256 fingerprints of Phase 1-5 outputs before Phase 6 runs."""
    manifest_path = layout["logs"] / "phase1_phase5_artifact_manifest.json"
    if manifest_path.is_file():
        return manifest_path

    roots = [
        PROJECT_ROOT / "results" / "metrics",
        PROJECT_ROOT / "results" / "figures",
        PROJECT_ROOT / "results" / "reports",
        PROJECT_ROOT / "results" / "dimensionality_reduction",
        PROJECT_ROOT / "results" / "classifier_evaluation",
        PROJECT_ROOT / "data" / "processed",
    ]
    files: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            files.append(
                {
                    "relative_path": _relative(path),
                    "size_bytes": int(path.stat().st_size),
                    "mtime_ns": int(path.stat().st_mtime_ns),
                    "sha256": _sha256_file(path),
                }
            )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (
            "Immutability manifest for Phase 1-5 outputs, created before the "
            "Phase 6 self-training run."
        ),
        "roots": [_relative(root) for root in roots if root.exists()],
        "file_count": len(files),
        "files": files,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return manifest_path


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def _save_predictions(
    experiment_id: str,
    preprocessed: PreprocessedDataset,
    y_pred: np.ndarray,
    predictions_dir: Path,
) -> str:
    frame = pd.DataFrame(
        {
            "test_position": np.arange(preprocessed.n_test, dtype=np.int64),
            "test_index": np.asarray(preprocessed.test_indices, dtype=np.int64),
            "y_true": preprocessed.y_test,
            "y_pred": np.asarray(y_pred),
        }
    )
    path = predictions_dir / f"{experiment_id}.csv"
    frame.to_csv(path, index=False)
    return _relative(path)


def _save_iteration_log(
    experiment_id: str,
    result: SelfTrainingResult,
    iterations_dir: Path,
) -> str:
    rows = [record.to_row() for record in result.iterations]
    frame = pd.DataFrame(rows, columns=list(ITERATION_COLUMNS))
    path = iterations_dir / f"{experiment_id}.csv"
    frame.to_csv(path, index=False)
    return _relative(path)


def _save_pseudo_audit(
    experiment_id: str,
    result: SelfTrainingResult,
    unlabelled_train_indices: np.ndarray,
    train_dataset_indices: np.ndarray,
    audit_dir: Path,
) -> str:
    positions = np.asarray(result.pseudo_positions, dtype=np.int64)
    train_indices = np.asarray(unlabelled_train_indices, dtype=np.int64)[positions]
    dataset_indices = np.asarray(train_dataset_indices, dtype=np.int64)[train_indices]
    frame = pd.DataFrame(
        {
            "unlabelled_position": positions,
            "train_index": train_indices,
            "dataset_index": dataset_indices,
            "pseudo_label": np.asarray(result.pseudo_labels),
            "confidence": np.asarray(result.pseudo_confidences, dtype=float),
            "iteration": np.asarray(result.pseudo_iterations, dtype=np.int64),
            "pseudo_label_source": PSEUDO_LABEL_SOURCE,
        },
        columns=list(PSEUDO_AUDIT_COLUMNS),
    )
    path = audit_dir / f"{experiment_id}.csv"
    frame.to_csv(path, index=False)
    return _relative(path)


def _warning_text(warnings_list: Sequence[Any]) -> str | None:
    messages: list[str] = []
    for warning in warnings_list:
        message = f"{warning.category.__name__}: {warning.message}"
        if message not in messages:
            messages.append(message)
    if not messages:
        return None
    text = "; ".join(messages)
    return text[:1000]


def _write_results(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _ordered_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
    if frame.empty:
        return frame
    dataset_rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    representation_rank = {
        name: index for index, name in enumerate(REPRESENTATION_ORDER)
    }
    classifier_rank = {name: index for index, name in enumerate(CLASSIFIER_NAMES)}
    frame = frame.assign(
        _dataset_rank=frame["dataset"].map(
            lambda value: dataset_rank.get(str(value), len(dataset_rank))
        ),
        _representation_rank=frame["feature_representation"].map(
            lambda value: representation_rank.get(
                str(value), len(representation_rank)
            )
        ),
        _classifier_rank=frame["classifier"].map(
            lambda value: classifier_rank.get(str(value), len(classifier_rank))
        ),
    )
    frame = frame.sort_values(
        [
            "_dataset_rank",
            "labelled_percentage",
            "_representation_rank",
            "_classifier_rank",
        ]
    ).drop(
        columns=["_dataset_rank", "_representation_rank", "_classifier_rank"]
    )
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------
def _experiment_id(
    dataset: str, percentage: int, representation: str, classifier: str
) -> str:
    return (
        f"{dataset}__labelled_{percentage:02d}__{representation}__{classifier}"
    )


def _model_factories(
    classifier_name: str,
    split_n_labelled: int,
    config: dict[str, Any],
    random_state: int,
) -> tuple[Any, Any, str, str]:
    params = config.get("classifiers", {}).get(classifier_name, {})
    if classifier_name == "svm":
        calibration = (
            config.get("self_training", {}).get("svm_calibration", {}) or {}
        )
        calibration_fraction = float(
            calibration.get("calibration_fraction", 0.20)
        )
        min_per_class = int(calibration.get("min_per_class", 2))
        max_splits = int(calibration.get("max_splits", 5))

        def factory() -> Any:
            return build_calibrated_svm(
                params,
                n_original_labelled=split_n_labelled,
                calibration_fraction=calibration_fraction,
                min_calibration_per_class=min_per_class,
                max_calibration_splits=max_splits,
                random_state=random_state,
            )

        def final_factory() -> Any:
            return build_svm(params, random_state=random_state)

        return (
            factory,
            final_factory,
            CONFIDENCE_METHODS[classifier_name],
            FINAL_MODEL_DESCRIPTIONS[classifier_name],
        )

    def factory() -> Any:
        return build_classifier(classifier_name, params, random_state=random_state)

    return (
        factory,
        None,
        CONFIDENCE_METHODS[classifier_name],
        FINAL_MODEL_DESCRIPTIONS[classifier_name],
    )


def _max_abs_value(array: np.ndarray) -> float:
    """Largest absolute value without materialising a full copy of the array."""
    column_max = np.asarray(array).max(axis=0)
    column_min = np.asarray(array).min(axis=0)
    return float(max(float(np.max(column_max)), float(-np.min(column_min))))


def _diagnose_final_fit(
    classifier: Any,
    X_labelled: np.ndarray,
    X_unlabelled: np.ndarray,
    y_augmented: np.ndarray,
    n_augmented_rows: int,
) -> tuple[bool, str]:
    """Flag a final fit whose outputs are numerically indistinguishable from uniform.

    LDA fitted on a small labelled sample can project out-of-sample rows to
    extreme magnitudes. When those rows enter the augmented training set,
    some optimizers (notably lbfgs logistic regression) return near-zero
    coefficients and a uniform probability vector. The measured metrics are
    still recorded, but the flag keeps the collapse visible in every artifact.

    The probe uses real labelled rows only: a healthy classifier is expected
    to be confident and discriminative on rows it was fitted on, whereas a
    collapsed optimizer emits the same uniform vector everywhere.
    """
    labelled = np.asarray(X_labelled)
    sample = labelled[: min(2000, labelled.shape[0])]

    classes = getattr(classifier, "classes_", None)
    n_classes = (
        int(np.unique(classes).size)
        if classes is not None
        else int(np.unique(y_augmented).size)
    )
    n_classes = max(n_classes, 2)
    uniform = 1.0 / n_classes

    details = [
        f"augmented_rows={int(n_augmented_rows)}",
        f"labelled_max_abs_feature={_max_abs_value(labelled):.3e}",
        "unlabelled_pool_max_abs_feature="
        f"{_max_abs_value(np.asarray(X_unlabelled)):.3e}",
    ]
    coefficients = getattr(classifier, "coef_", None)
    if coefficients is not None:
        details.append(
            f"max_abs_coefficient={float(np.abs(coefficients).max()):.3e}"
        )

    degenerate = False
    if hasattr(classifier, "predict_proba"):
        probabilities = np.asarray(classifier.predict_proba(sample), dtype=float)
        top = probabilities.max(axis=1)
        degenerate = bool(
            top.mean() <= uniform + 1e-6 and top.max() <= uniform + 1e-3
        )
        details.extend(
            [
                f"mean_top_probability={float(top.mean()):.6f}",
                f"max_top_probability={float(top.max()):.6f}",
                f"uniform_probability={uniform:.6f}",
            ]
        )
    elif hasattr(classifier, "decision_function"):
        scores = np.asarray(classifier.decision_function(sample), dtype=float)
        spread = float(np.nanmax(scores) - np.nanmin(scores))
        degenerate = bool(spread <= 1e-12)
        details.append(f"decision_function_spread={spread:.3e}")
    else:
        predictions = np.asarray(classifier.predict(sample))
        n_predicted = int(np.unique(predictions).size)
        degenerate = bool(n_predicted < 2)
        details.append(f"predicted_classes={n_predicted}")

    return degenerate, "; ".join(details)


def run_single_self_training(
    preprocessed: PreprocessedDataset,
    split: Any,
    X_labelled: np.ndarray,
    X_unlabelled: np.ndarray,
    X_test: np.ndarray,
    unlabelled_train_indices: np.ndarray,
    transformed_count: int,
    classifier_name: str,
    feature_representation: str,
    config: dict[str, Any],
    training_config: SelfTrainingConfig,
    layout: dict[str, Path],
    view_fingerprints: tuple[str, str, str] | None = None,
) -> tuple[SelfTrainingRecord, SelfTrainingResult | None]:
    """Run one self-training experiment and persist its artifacts."""
    percentage = int(round(split.proportion * 100))
    identifier = _experiment_id(
        preprocessed.name, percentage, feature_representation, classifier_name
    )
    seed = int(config["project"]["random_seed"])
    y_test = np.asarray(preprocessed.y_test)
    factory, final_factory, confidence_method, final_description = _model_factories(
        classifier_name, split.n_labelled, config, seed
    )

    if view_fingerprints is None:
        view_fingerprints = (
            fingerprint_features(X_labelled),
            fingerprint_labels(split.y_labelled),
            fingerprint_features(X_unlabelled),
        )
    feature_fingerprint, label_fingerprint, unlabelled_fingerprint = (
        view_fingerprints
    )
    base = {
        "dataset": preprocessed.name,
        "labelled_percentage": percentage,
        "classifier": classifier_name,
        "feature_representation": feature_representation,
        "original_feature_count": preprocessed.n_features,
        "transformed_feature_count": int(transformed_count),
        "n_labelled_train": int(split.n_labelled),
        "n_unlabelled_train": int(split.n_unlabelled),
        "confidence_method": confidence_method,
        "fit_feature_sha256": feature_fingerprint,
        "fit_label_sha256": label_fingerprint,
        "unlabelled_feature_sha256": unlabelled_fingerprint,
    }

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_self_training(
                model_factory=factory,
                X_labelled=X_labelled,
                y_labelled=np.asarray(split.y_labelled),
                X_unlabelled=X_unlabelled,
                config=training_config,
                confidence_method=confidence_method,
                final_model_factory=final_factory,
                final_model_description=final_description,
            )

        if result.n_pseudo_labelled + result.n_remaining != split.n_unlabelled:
            raise RuntimeError(
                "Pseudo-label accounting failed: "
                f"{result.n_pseudo_labelled} + {result.n_remaining} != "
                f"{split.n_unlabelled}."
            )
        if np.unique(result.pseudo_positions).size != result.pseudo_positions.size:
            raise RuntimeError("Pseudo-label positions contain duplicates.")

        y_augmented = np.concatenate(
            [np.asarray(split.y_labelled), result.pseudo_labels]
        )
        degenerate, fit_diagnostic = _diagnose_final_fit(
            result.final_classifier,
            X_labelled,
            X_unlabelled,
            y_augmented,
            int(split.n_labelled) + int(result.n_pseudo_labelled),
        )

        start = time.perf_counter()
        y_pred = np.asarray(result.final_classifier.predict(X_test))
        prediction_time = time.perf_counter() - start
        if y_pred.shape[0] != preprocessed.n_test:
            raise RuntimeError(
                f"Model returned {y_pred.shape[0]} predictions for "
                f"{preprocessed.n_test} test rows."
            )
        metrics = evaluate_predictions(y_test, y_pred)

        warning_message = _warning_text(caught)
        if degenerate:
            collapse = (
                "NumericalStabilityWarning: final fit is degenerate "
                f"({fit_diagnostic})"
            )
            warning_message = (
                f"{warning_message}; {collapse}" if warning_message else collapse
            )

        record = SelfTrainingRecord(
            **base,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            training_time_seconds=float(result.training_time_seconds),
            prediction_time_seconds=float(prediction_time),
            pseudo_label_count=int(result.n_pseudo_labelled),
            remaining_unlabelled_count=int(result.n_remaining),
            iterations_completed=int(result.n_iterations),
            pseudo_label_acceptance_rate=compute_acceptance_rate(
                result.n_pseudo_labelled, result.n_unlabelled_original
            ),
            stopping_reason=result.stopping_reason,
            status="completed",
            numerically_degenerate_fit=bool(degenerate),
            fit_diagnostic=fit_diagnostic,
            warning_message=warning_message,
        )
        record.prediction_file = _save_predictions(
            identifier, preprocessed, y_pred, layout["predictions"]
        )
        record.iteration_log_file = _save_iteration_log(
            identifier, result, layout["iterations"]
        )
        record.pseudo_label_file = _save_pseudo_audit(
            identifier,
            result,
            unlabelled_train_indices,
            np.asarray(preprocessed.train_indices),
            layout["pseudo_labels"],
        )
        return record, result
    except Exception as exc:
        return (
            SelfTrainingRecord(
                **base,
                accuracy=None,
                macro_f1=None,
                training_time_seconds=None,
                prediction_time_seconds=None,
                pseudo_label_count=None,
                remaining_unlabelled_count=None,
                iterations_completed=None,
                pseudo_label_acceptance_rate=None,
                stopping_reason=None,
                status="failed",
                error_message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )


def _failed_record(
    dataset: str,
    percentage: int,
    classifier: str,
    representation: str,
    original_feature_count: int | None,
    error_message: str,
) -> SelfTrainingRecord:
    return SelfTrainingRecord(
        dataset=dataset,
        labelled_percentage=percentage,
        classifier=classifier,
        feature_representation=representation,
        original_feature_count=original_feature_count,
        transformed_feature_count=None,
        accuracy=None,
        macro_f1=None,
        training_time_seconds=None,
        prediction_time_seconds=None,
        pseudo_label_count=None,
        remaining_unlabelled_count=None,
        n_labelled_train=0,
        n_unlabelled_train=0,
        iterations_completed=None,
        pseudo_label_acceptance_rate=None,
        stopping_reason=None,
        confidence_method=CONFIDENCE_METHODS.get(classifier, ""),
        status="failed",
        error_message=error_message,
    )


def _dataset_summary(preprocessed: PreprocessedDataset, suite: Any) -> dict[str, Any]:
    return {
        "n_features": int(preprocessed.n_features),
        "n_train": int(preprocessed.n_train),
        "n_test": int(preprocessed.n_test),
        "target_classes": [str(label) for label in preprocessed.target_names],
        "labelled_proportions": {
            f"{proportion:.2f}": {
                "n_labelled": int(split.n_labelled),
                "n_unlabelled": int(split.n_unlabelled),
            }
            for proportion, split in sorted(preprocessed.semi_supervised.items())
        },
        "pca_components": (
            int(suite.pca.n_components) if suite.pca is not None else None
        ),
        "lda_components": {
            f"{proportion:.2f}": int(result.n_components)
            for proportion, result in sorted(suite.lda.items())
        },
    }


def _run_settings_signature(
    training_config: SelfTrainingConfig,
    datasets: Sequence[str],
    classifiers: Sequence[str],
    proportions: Sequence[float],
) -> dict[str, Any]:
    return {
        "confidence_threshold": float(training_config.confidence_threshold),
        "max_iterations": int(training_config.max_iterations),
        "minimum_new_pseudo_labels": int(
            training_config.minimum_new_pseudo_labels
        ),
        "random_state": int(training_config.random_state),
        "datasets": list(datasets),
        "classifiers": list(classifiers),
        "representations": list(REPRESENTATION_ORDER),
        "labelled_percentages": [int(round(value * 100)) for value in proportions],
    }


def _checkpoint_settings(
    layout: dict[str, Path],
    signature: dict[str, Any],
    resume: bool,
) -> Path:
    """Keep checkpoints from being resumed under different run settings."""
    path = layout["tables"] / "phase6_run_settings.json"
    if resume and path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Checkpoint settings file {_relative(path)} is unreadable: {exc}"
            ) from exc
        if existing != signature:
            raise RuntimeError(
                "Cannot resume: the checkpoint was written with different run "
                "settings. Archive or remove the Phase 6 results directory "
                "before rerunning with the new settings."
            )
    path.write_text(
        json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def run_self_training_grid(
    config: dict[str, Any],
    resume: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute all 81 self-training experiments and return the results frame."""
    seed = int(config["project"]["random_seed"])
    set_global_seed(seed)
    datasets = _enabled_datasets(config)
    classifiers = list_classifiers()
    layout = self_training_layout(config)
    manifest_path = ensure_phase15_manifest(layout)
    training_config = SelfTrainingConfig.from_mapping(
        config.get("self_training", {})
    )
    proportions = [float(value) for value in config["data"]["labelled_proportions"]]
    expected = len(datasets) * len(proportions) * len(REPRESENTATION_ORDER) * len(
        classifiers
    )
    csv_path = layout["tables"] / "self_training_results.csv"
    settings_path = _checkpoint_settings(
        layout,
        _run_settings_signature(
            training_config, datasets, classifiers, proportions
        ),
        resume,
    )

    rows_by_id: dict[str, dict[str, Any]] = {}
    if resume and csv_path.is_file():
        existing = pd.read_csv(csv_path)
        for row in existing.to_dict(orient="records"):
            identifier = str(row.get("experiment_id"))
            prediction = row.get("prediction_file")
            if (
                row.get("status") == "completed"
                and isinstance(prediction, str)
                and (PROJECT_ROOT / prediction).is_file()
            ):
                rows_by_id[identifier] = row
        print(f"Resuming: {len(rows_by_id)} completed experiment(s) loaded.")

    log_path = layout["logs"] / "phase6_run.log"
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(
        f"\n=== Phase 6 self-training run started "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} ===\n"
    )
    log_handle.flush()

    print("=" * 78)
    print("Phase 6 semi-supervised self-training")
    print(f"Datasets            : {datasets}")
    print(f"Labelled %          : {[int(round(value * 100)) for value in proportions]}")
    print(f"Representations     : {list(REPRESENTATION_ORDER)}")
    print(f"Classifiers         : {classifiers}")
    print(f"Confidence threshold: {training_config.confidence_threshold}")
    print(f"Max iterations      : {training_config.max_iterations}")
    print(f"Expected            : {expected} experiments")
    print(f"Seed                : {seed}")
    print(f"Checkpoint settings : {_relative(settings_path)}")
    print(f"Manifest            : {_relative(manifest_path)}")
    print("=" * 78)

    summaries: dict[str, Any] = {}
    counter = 0
    run_start = time.perf_counter()

    for dataset_name in datasets:
        print(f"\nDataset: {DATASET_LABELS.get(dataset_name, dataset_name)}")
        log_handle.write(f"[dataset] {dataset_name}\n")
        try:
            preprocessed = preprocess_dataset(dataset_name, config)
            suite = run_dimensionality_reduction(preprocessed, config)
            if suite.pca is None or not suite.lda:
                raise RuntimeError(
                    "Phase 4 suite is missing the PCA or LDA transformations "
                    "required by Phase 6."
                )
            summaries[dataset_name] = _dataset_summary(preprocessed, suite)
        except Exception as exc:
            error = f"dataset setup failed: {type(exc).__name__}: {exc}"
            print(f"  [FAIL] {error}")
            log_handle.write(f"  [FAIL] {error}\n")
            expected_features = EXPECTED_FEATURE_COUNTS.get(dataset_name)
            for proportion in proportions:
                for representation in REPRESENTATION_ORDER:
                    for classifier in classifiers:
                        counter += 1
                        record = _failed_record(
                            dataset_name,
                            int(round(proportion * 100)),
                            classifier,
                            representation,
                            expected_features,
                            error,
                        )
                        rows_by_id[record.experiment_id] = record.to_row()
            summaries[dataset_name] = {"error": error}
            _write_results(_ordered_frame(list(rows_by_id.values())), csv_path)
            continue

        for proportion in sorted(preprocessed.semi_supervised):
            split = preprocessed.semi_supervised[proportion]
            percentage = int(round(proportion * 100))
            for representation in REPRESENTATION_ORDER:
                # Views and their fingerprints depend only on the split and
                # the representation. Build them once and share them across
                # the three classifiers instead of copying and hashing the
                # same arrays three times.
                pending = [
                    classifier
                    for classifier in classifiers
                    if _experiment_id(
                        dataset_name, percentage, representation, classifier
                    )
                    not in rows_by_id
                ]
                views: tuple[Any, ...] | None = None
                view_error: str | None = None
                if pending:
                    try:
                        (
                            X_labelled,
                            X_unlabelled,
                            X_test,
                            unlabelled_train_indices,
                            transformed_count,
                        ) = get_self_training_views(
                            preprocessed,
                            suite,
                            split,
                            representation,
                            proportion,
                        )
                        views = (
                            X_labelled,
                            X_unlabelled,
                            X_test,
                            unlabelled_train_indices,
                            transformed_count,
                            (
                                fingerprint_features(X_labelled),
                                fingerprint_labels(split.y_labelled),
                                fingerprint_features(X_unlabelled),
                            ),
                        )
                    except Exception as exc:
                        view_error = f"{type(exc).__name__}: {exc}"

                for classifier in classifiers:
                    counter += 1
                    identifier = _experiment_id(
                        dataset_name, percentage, representation, classifier
                    )
                    if identifier in rows_by_id:
                        print(
                            f"  [{counter:>2}/{expected}] {identifier} "
                            "skipped (completed, resume mode)",
                            flush=True,
                        )
                        continue
                    dataset_start = time.perf_counter()
                    try:
                        if views is None or view_error is not None:
                            raise RuntimeError(
                                view_error or "view reconstruction failed"
                            )
                        (
                            X_labelled,
                            X_unlabelled,
                            X_test,
                            unlabelled_train_indices,
                            transformed_count,
                            view_fingerprints,
                        ) = views
                        record, result = run_single_self_training(
                            preprocessed=preprocessed,
                            split=split,
                            X_labelled=X_labelled,
                            X_unlabelled=X_unlabelled,
                            X_test=X_test,
                            unlabelled_train_indices=unlabelled_train_indices,
                            transformed_count=transformed_count,
                            classifier_name=classifier,
                            feature_representation=representation,
                            config=config,
                            training_config=training_config,
                            layout=layout,
                            view_fingerprints=view_fingerprints,
                        )
                    except Exception as exc:
                        record, result = (
                            _failed_record(
                                dataset_name,
                                percentage,
                                classifier,
                                representation,
                                preprocessed.n_features,
                                f"{type(exc).__name__}: {exc}",
                            ),
                            None,
                        )
                    rows_by_id[record.experiment_id] = record.to_row()
                    _write_results(
                        _ordered_frame(list(rows_by_id.values())), csv_path
                    )
                    elapsed = time.perf_counter() - dataset_start
                    if record.status == "completed":
                        line = (
                            f"  [{counter:>2}/{expected}] {identifier} "
                            f"accuracy={record.accuracy:.4f} "
                            f"macro_f1={record.macro_f1:.4f} "
                            f"pseudo={record.pseudo_label_count}/"
                            f"{record.n_unlabelled_train} "
                            f"iters={record.iterations_completed} "
                            f"stop={record.stopping_reason} "
                            f"degenerate={record.numerically_degenerate_fit} "
                            f"fit={record.training_time_seconds:.2f}s "
                            f"elapsed={elapsed:.2f}s"
                        )
                    else:
                        line = (
                            f"  [{counter:>2}/{expected}] {identifier} "
                            f"FAILED: {record.error_message} "
                            f"elapsed={elapsed:.2f}s"
                        )
                    print(line, flush=True)
                    log_handle.write(line + "\n")
                    log_handle.flush()

                if views is not None:
                    del (
                        X_labelled,
                        X_unlabelled,
                        X_test,
                        unlabelled_train_indices,
                        transformed_count,
                        view_fingerprints,
                        views,
                    )

        del suite
        del preprocessed
        gc.collect()

    log_handle.write(
        f"=== Phase 6 run finished after "
        f"{time.perf_counter() - run_start:.2f}s ===\n"
    )
    log_handle.close()
    frame = _ordered_frame(list(rows_by_id.values()))
    return frame, summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _completed(frame: pd.DataFrame) -> pd.DataFrame:
    if "status" not in frame.columns:
        return frame.iloc[0:0]
    return frame[frame["status"] == "completed"]


def _format_group_table(frame: pd.DataFrame, column: str, label: str) -> list[str]:
    completed = _completed(frame)
    lines = [label, "-" * 78]
    if completed.empty:
        return lines + ["No completed experiments."]
    grouped = (
        completed.groupby(column, sort=True)
        .agg(
            experiments=("accuracy", "size"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            pseudo_mean=("pseudo_label_count", "mean"),
            acceptance_mean=("pseudo_label_acceptance_rate", "mean"),
            iterations_mean=("iterations_completed", "mean"),
            fit_seconds_mean=("training_time_seconds", "mean"),
        )
        .reset_index()
    )
    lines.append(
        f"{column:<24} {'n':>3} {'accuracy':>9} {'macro_f1':>9} "
        f"{'pseudo':>9} {'accept':>7} {'iters':>6} {'fit_s':>10}"
    )
    for row in grouped.itertuples(index=False):
        lines.append(
            f"{str(getattr(row, column)):<24} "
            f"{int(row.experiments):>3} "
            f"{float(row.accuracy_mean):>9.4f} "
            f"{float(row.macro_f1_mean):>9.4f} "
            f"{float(row.pseudo_mean):>9.1f} "
            f"{float(row.acceptance_mean):>7.3f} "
            f"{float(row.iterations_mean):>6.1f} "
            f"{float(row.fit_seconds_mean):>10.3f}"
        )
    return lines


def _comparison_group_table(
    comparison: pd.DataFrame, column: str, label: str
) -> list[str]:
    lines = [label, "-" * 78]
    valid = comparison.dropna(subset=["macro_f1_improvement"])
    if valid.empty:
        return lines + ["No comparable experiments."]
    grouped = (
        valid.groupby(column, sort=True)
        .agg(
            experiments=("macro_f1_improvement", "size"),
            supervised=("supervised_macro_f1", "mean"),
            self_training=("self_training_macro_f1", "mean"),
            improvement=("macro_f1_improvement", "mean"),
            improved=("macro_f1_improvement", lambda values: int((values > 1e-12).sum())),
            regressed=("macro_f1_improvement", lambda values: int((values < -1e-12).sum())),
        )
        .reset_index()
    )
    lines.append(
        f"{column:<24} {'n':>3} {'sup_F1':>8} {'st_F1':>8} "
        f"{'delta':>9} {'better':>7} {'worse':>6}"
    )
    for row in grouped.itertuples(index=False):
        lines.append(
            f"{str(getattr(row, column)):<24} "
            f"{int(row.experiments):>3} "
            f"{float(row.supervised):>8.4f} "
            f"{float(row.self_training):>8.4f} "
            f"{float(row.improvement):>+9.4f} "
            f"{int(row.improved):>7} "
            f"{int(row.regressed):>6}"
        )
    return lines


def build_phase6_report(
    frame: pd.DataFrame,
    summaries: dict[str, Any],
    comparison: pd.DataFrame | None,
    comparison_summary: dict[str, Any] | None,
    iteration_logs: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> str:
    """Build the Phase 6 report from measured artifacts only."""
    completed = _completed(frame)
    failed = frame[frame["status"] != "completed"]
    total = len(frame)
    degenerate_count = (
        int(frame["numerically_degenerate_fit"].fillna(False).astype(bool).sum())
        if "numerically_degenerate_fit" in frame.columns
        else 0
    )
    training = SelfTrainingConfig.from_mapping(config.get("self_training", {}))
    lines: list[str] = [
        "Phase 6 semi-supervised self-training report",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "1. Research objective",
        "-" * 78,
        "Measure whether confidence-based self-training can improve the three",
        "supervised classifiers (Logistic Regression, linear SVM, KNN) when only",
        "10%, 25%, or 50% of the training partition is labelled, and whether the",
        "effect depends on the feature representation (baseline, PCA, LDA).",
        "Every Phase 6 result is compared with the matching Phase 5 supervised",
        "experiment on the same split, representation, classifier, and proportion.",
        "",
        "2. Self-training methodology",
        "-" * 78,
        "For each condition the classifier is first fitted on the original",
        "labelled rows only. It then predicts class probabilities for the",
        "remaining unlabelled training rows. Rows whose maximum predicted",
        "probability is at least the confidence threshold receive the predicted",
        "class as a pseudo-label, are removed from the unlabelled pool, and are",
        "appended to the training set. The classifier is refitted on the original",
        "labelled rows plus all accumulated pseudo-labelled rows. This repeats",
        "until a stopping condition is met. Final test predictions come from the",
        "refitted model; test rows never enter the loop.",
        "",
        "3. Confidence calculation for each classifier",
        "-" * 78,
        "* Logistic Regression: predict_proba; confidence is the maximum class",
        "  probability.",
        "* KNN: predict_proba; confidence is the maximum class probability.",
        "* Linear SVM: Phase 5 LinearSVC has no predict_proba. A class-stratified",
        "  subset of the ORIGINAL labelled rows is held out and used to fit",
        "  sigmoid (Platt) calibrators with StratifiedKFold folds via",
        "  CalibratedClassifierCV and a frozen LinearSVC. The base SVM keeps the",
        "  Phase 5 configuration and is fitted on the remaining labelled rows plus",
        "  pseudo-labelled rows. Final test predictions use a plain Phase 5",
        "  LinearSVC refitted on all labelled plus pseudo-labelled rows.",
        "Calibrated SVM probabilities are approximations and are not guaranteed to",
        "be on the same scale as Logistic Regression or KNN probabilities.",
        "",
        "4. Confidence threshold and iteration settings",
        "-" * 78,
        f"Confidence threshold      : {training.confidence_threshold}",
        f"Maximum iterations        : {training.max_iterations}",
        f"Minimum new pseudo-labels : {training.minimum_new_pseudo_labels}",
        f"Random seed               : {training.random_state}",
        "The threshold is fixed for all experiments and is never tuned using",
        "test-set performance or unlabelled ground-truth labels.",
        "",
        "5. Leakage prevention methodology",
        "-" * 78,
        "* Only the labelled training rows and their labels are passed to the",
        "  initial fit; unlabelled truth is never read by the self-training loop.",
        "* Pseudo-label indices come from the original Phase 3 unlabelled training",
        "  pool, are unique, and are disjoint from the labelled indices.",
        "* Test rows are used only for final prediction and metrics.",
        "* Preprocessing and PCA/LDA transformations are reused from Phase 3/4;",
        "  no transformation is fitted on test rows or unlabelled labels.",
        "* SHA-256 fingerprints of the labelled features, labelled targets, and",
        "  unlabelled features are stored for every experiment.",
        "",
        "6. Number of experiments completed",
        "-" * 78,
        f"Rows recorded        : {total}",
        f"Completed            : {len(completed)}",
        f"Failed               : {len(failed)}",
        f"Numerically degenerate fits: {degenerate_count}",
        f"Expected             : 81 (3 datasets x 3 proportions x 3 "
        "representations x 3 classifiers)",
    ]

    if not completed.empty:
        overall_acceptance = (
            completed["pseudo_label_count"].sum()
            / completed["n_unlabelled_train"].sum()
        )
        lines.extend(
            [
                "",
                "7. Pseudo-label acceptance observations",
                "-" * 78,
                f"Total pseudo-labels accepted : "
                f"{int(completed['pseudo_label_count'].sum()):,}",
                f"Overall acceptance rate      : {overall_acceptance:.4f}",
                f"Mean pseudo-labels/experiment: "
                f"{completed['pseudo_label_count'].mean():.1f}",
                f"Acceptance rate min/mean/max : "
                f"{completed['pseudo_label_acceptance_rate'].min():.4f} / "
                f"{completed['pseudo_label_acceptance_rate'].mean():.4f} / "
                f"{completed['pseudo_label_acceptance_rate'].max():.4f}",
                f"Iterations min/mean/max      : "
                f"{int(completed['iterations_completed'].min())} / "
                f"{completed['iterations_completed'].mean():.2f} / "
                f"{int(completed['iterations_completed'].max())}",
                "",
                "Stopping reasons:",
            ]
        )
        for reason, count in (
            completed["stopping_reason"].value_counts().sort_index().items()
        ):
            lines.append(f"  {reason}: {count}")
        lines.append("")
        for classifier in CLASSIFIER_NAMES:
            subset = completed[completed["classifier"] == classifier]
            if subset.empty:
                continue
            lines.append(
                f"  {CLASSIFIER_LABELS.get(classifier, classifier):<24} "
                f"mean accepted={subset['pseudo_label_count'].mean():.1f}, "
                f"mean rate={subset['pseudo_label_acceptance_rate'].mean():.4f}, "
                f"mean iterations={subset['iterations_completed'].mean():.2f}"
            )
        lines.append(
            "Acceptance depends on the classifier's probability scale; the same "
            "0.90 threshold is not equally strict for every classifier."
        )

    for label, column in (
        ("8. Results by dataset", "dataset"),
        ("9. Results by classifier", "classifier"),
        ("10. Results by feature representation", "feature_representation"),
        ("11. Results by labelled percentage", "labelled_percentage"),
    ):
        lines.extend(["", label, "-" * 78])
        lines.extend(_format_group_table(frame, column, label)[2:])

    lines.extend(["", "12. Supervised vs self-training comparison", "-" * 78])
    if comparison is None or comparison_summary is None:
        lines.append("Comparison could not be computed; see the error log.")
    else:
        metric_summary = comparison_summary.get("macro_f1", {})
        lines.extend(
            [
                f"Matched experiments   : {comparison_summary.get('matched_experiments')}",
                f"Comparable experiments: {comparison_summary.get('comparable_experiments')}",
                f"Mean supervised macro F1  : "
                f"{metric_summary.get('mean_supervised', float('nan')):.4f}",
                f"Mean self-training macro F1: "
                f"{metric_summary.get('mean_self_training', float('nan')):.4f}",
                f"Mean macro-F1 improvement  : "
                f"{metric_summary.get('mean_improvement', float('nan')):+.4f}",
                f"Improved / unchanged / regressed: "
                f"{metric_summary.get('improved_count')} / "
                f"{metric_summary.get('unchanged_count')} / "
                f"{metric_summary.get('regressed_count')}",
                "",
            ]
        )
        for label, column in (
            ("Comparison by dataset", "dataset"),
            ("Comparison by classifier", "classifier"),
            ("Comparison by representation", "feature_representation"),
            ("Comparison by labelled percentage", "labelled_percentage"),
        ):
            lines.extend(_comparison_group_table(comparison, column, label))
            lines.append("")

    lines.extend(["", "13. Best and worst configurations (self-training)", "-" * 78])
    if completed.empty:
        lines.append("No completed experiments.")
    else:
        best = completed.sort_values(
            ["macro_f1", "accuracy"], ascending=[False, False]
        ).head(5)
        worst = completed.sort_values(
            ["macro_f1", "accuracy"], ascending=[True, True]
        ).head(5)
        lines.append("Best five by macro F1:")
        for row in best.itertuples(index=False):
            lines.append(
                f"  {row.experiment_id}: accuracy={row.accuracy:.4f}, "
                f"macro_f1={row.macro_f1:.4f}, pseudo={row.pseudo_label_count}"
            )
        lines.append("Worst five by macro F1:")
        for row in worst.itertuples(index=False):
            lines.append(
                f"  {row.experiment_id}: accuracy={row.accuracy:.4f}, "
                f"macro_f1={row.macro_f1:.4f}, pseudo={row.pseudo_label_count}"
            )
        for dataset in DATASET_ORDER:
            subset = completed[completed["dataset"] == dataset]
            if subset.empty:
                continue
            best_dataset = subset.sort_values(
                ["macro_f1", "accuracy"], ascending=[False, False]
            ).iloc[0]
            lines.append(
                f"  Best [{dataset}]: {best_dataset['experiment_id']} "
                f"(accuracy={best_dataset['accuracy']:.4f}, "
                f"macro_f1={best_dataset['macro_f1']:.4f})"
            )

    lines.extend(["", "14. Cases where self-training improved performance", "-" * 78])
    if comparison is None:
        lines.append("No comparison available.")
    else:
        improved = (
            comparison.dropna(subset=["macro_f1_improvement"])
            .sort_values("macro_f1_improvement", ascending=False)
            .head(10)
        )
        if improved.empty:
            lines.append("No comparable experiments.")
        else:
            for row in improved.itertuples(index=False):
                lines.append(
                    f"  {row.experiment_id}: macro_f1 "
                    f"{row.supervised_macro_f1:.4f} -> "
                    f"{row.self_training_macro_f1:.4f} "
                    f"({row.macro_f1_improvement:+.4f})"
                )

    lines.extend(["", "15. Cases where self-training reduced performance", "-" * 78])
    if comparison is None:
        lines.append("No comparison available.")
    else:
        regressed = (
            comparison.dropna(subset=["macro_f1_improvement"])
            .sort_values("macro_f1_improvement", ascending=True)
            .head(10)
        )
        if regressed.empty:
            lines.append("No comparable experiments.")
        else:
            for row in regressed.itertuples(index=False):
                lines.append(
                    f"  {row.experiment_id}: macro_f1 "
                    f"{row.supervised_macro_f1:.4f} -> "
                    f"{row.self_training_macro_f1:.4f} "
                    f"({row.macro_f1_improvement:+.4f})"
                )

    warnings_present = (
        int(completed["warning_message"].notna().sum())
        if not completed.empty
        else 0
    )
    lines.extend(
        [
            "",
            "16. Convergence warnings and limitations",
            "-" * 78,
            f"Completed experiments with warnings: {warnings_present}",
            f"Numerically degenerate final fits: {degenerate_count}",
            "* One fixed split per dataset (seed 42); no repeated splits and no",
            "  confidence intervals or significance tests.",
            "* A single fixed confidence threshold (0.90) is used; no threshold",
            "  tuning is performed, by design.",
            "* A single fixed calibration fraction is used for the linear SVM;",
            "  calibrated probabilities are approximate and depend on the",
            "  calibration subset size, which is small for Breast Cancer at 10%.",
            "* Self-training can confirm its own errors: pseudo-labels are model",
            "  predictions, so high confidence does not guarantee correctness.",
            "* KNN probabilities are discrete/weighted neighbour votes and are not",
            "  directly comparable with logistic or calibrated SVM probabilities.",
            "* Runtime grows with the pseudo-labelled set; iterative refitting was",
            "  performed sequentially.",
        ]
    )
    if not completed.empty:
        warning_messages = completed["warning_message"].dropna().unique()
        for message in warning_messages[:5]:
            lines.append(f"  Warning: {message}")
    if degenerate_count and "numerically_degenerate_fit" in frame.columns:
        flagged = frame[
            frame["numerically_degenerate_fit"].fillna(False).astype(bool)
        ]
        lines.extend(
            [
                "",
                "Numerically degenerate final fits (metrics recorded as observed;",
                "the optimizer returned a near-uniform model, so these results show",
                "a collapse rather than a meaningful self-training outcome):",
            ]
        )
        for row in flagged.itertuples(index=False):
            lines.append(
                f"  {row.experiment_id}: accuracy={row.accuracy:.4f}, "
                f"macro_f1={row.macro_f1:.4f}, {row.fit_diagnostic}"
            )
        lines.append(
            "The extreme magnitudes are produced by the Phase 4 LDA projection "
            "fitted on a small labelled sample; no Phase 4 or Phase 5 artifact is "
            "modified, and no feature rescaling is applied in Phase 6."
        )

    lines.extend(
        [
            "",
            "17. Scientific interpretation",
            "-" * 78,
        ]
    )
    if comparison is None or not comparison_summary or "macro_f1" not in comparison_summary:
        lines.append("Comparison unavailable; no interpretation is claimed.")
    else:
        metric_summary = comparison_summary["macro_f1"]
        mean_delta = metric_summary["mean_improvement"]
        if mean_delta > 0.005:
            direction = "improved on average"
        elif mean_delta < -0.005:
            direction = "decreased on average"
        else:
            direction = "changed little on average"
        lines.extend(
            [
                f"Across {comparison_summary['comparable_experiments']} comparable",
                f"experiments, self-training {direction} in macro F1 "
                f"(mean {mean_delta:+.4f}).",
                f"{metric_summary['improved_count']} experiments improved and "
                f"{metric_summary['regressed_count']} regressed. The effect is",
                "therefore not uniform: it depends on the dataset, classifier, and",
                "feature representation. These results describe one fixed split",
                "and one threshold; they do not establish statistical significance.",
            ]
        )
        if comparison_summary.get("accuracy"):
            accuracy_summary = comparison_summary["accuracy"]
            lines.append(
                f"Mean accuracy improvement: {accuracy_summary['mean_improvement']:+.4f} "
                f"({accuracy_summary['improved_count']} improved, "
                f"{accuracy_summary['regressed_count']} regressed)."
            )
        if degenerate_count:
            lines.append(
                f"{degenerate_count} numerically degenerate fits are included in these "
                "averages; they are reported as observed collapses, not as valid "
                "classifier performance."
            )

    lines.extend(
        [
            "",
            "18. Future improvements",
            "-" * 78,
            "* Repeat with multiple random splits and report variability.",
            "* Study threshold sensitivity without using test labels for selection.",
            "* Compare with other semi-supervised methods (label propagation,",
            "  co-training, pseudo-label weighting) and class-balanced acceptance.",
            "* Use cross-fitted calibration for the linear SVM to reduce the",
            "  calibration holdout cost.",
            "* Evaluate per-class precision/recall of pseudo-labels using a",
            "  research-only audit that is never fed back into training.",
            "",
            "Dataset summaries",
            "-" * 78,
        ]
    )
    for dataset, summary in summaries.items():
        lines.append(f"[{dataset}] {json.dumps(_json_safe(summary), sort_keys=True)}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Combined outputs
# ---------------------------------------------------------------------------
def load_iteration_logs(layout: dict[str, Path]) -> dict[str, pd.DataFrame]:
    logs: dict[str, pd.DataFrame] = {}
    for path in sorted(layout["iterations"].glob("*.csv")):
        logs[path.stem] = pd.read_csv(path)
    return logs


def save_self_training_outputs(
    frame: pd.DataFrame,
    summaries: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Write the CSV/JSON, comparison, iteration logs, report, and figures."""
    layout = self_training_layout(config)
    csv_path = layout["tables"] / "self_training_results.csv"
    save_metrics(frame, csv_path)
    completed = _completed(frame)
    failed = frame[frame["status"] != "completed"]

    payload = {
        "phase": 6,
        "method": "confidence-based self-training",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": {
            "random_seed": int(config["project"]["random_seed"]),
            "expected_experiments": 81,
            "recorded_experiments": int(len(frame)),
            "completed_experiments": int(len(completed)),
            "failed_experiments": int(len(failed)),
            "confidence_threshold": float(
                config["self_training"]["confidence_threshold"]
            ),
            "max_iterations": int(config["self_training"]["max_iterations"]),
            "minimum_new_pseudo_labels": int(
                config["self_training"]["minimum_new_pseudo_labels"]
            ),
            "classifiers": {
                name: CLASSIFIER_LABELS.get(name, name) for name in CLASSIFIER_NAMES
            },
            "feature_representations": list(REPRESENTATION_ORDER),
            "labelled_percentages": list(LABELLED_PERCENTAGES),
            "confidence_methods": CONFIDENCE_METHODS,
            "pseudo_label_source": PSEUDO_LABEL_SOURCE,
            "datasets": summaries,
        },
        "experiments": to_json_records(frame),
    }
    json_path = layout["tables"] / "self_training_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")

    iteration_logs = load_iteration_logs(layout)
    iteration_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiments": {
            identifier: log.to_dict(orient="records")
            for identifier, log in iteration_logs.items()
        },
    }
    with (layout["logs"] / "iteration_logs.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_json_safe(iteration_payload), handle, indent=2)
        handle.write("\n")

    comparison_frame: pd.DataFrame | None = None
    comparison_summary: dict[str, Any] | None = None
    comparison_paths: dict[str, Path] | None = None
    try:
        supervised, supervised_path = load_phase5_results(config)
        comparison_frame = build_comparison(supervised, frame)
        comparison_summary = summarise_comparison(comparison_frame)
        comparison_paths = save_comparison(
            comparison_frame,
            layout["tables"],
            supervised_path=supervised_path,
            self_training_path=csv_path,
        )
    except Exception as exc:
        warning = f"comparison failed: {type(exc).__name__}: {exc}"
        print(f"WARNING: {warning}")
        (layout["logs"] / "comparison_error.txt").write_text(
            warning + "\n", encoding="utf-8"
        )

    report_path = layout["reports"] / "phase6_self_training_report.txt"
    report_path.write_text(
        build_phase6_report(
            frame,
            summaries,
            comparison_frame,
            comparison_summary,
            iteration_logs,
            config,
        ),
        encoding="utf-8",
    )

    figure_paths: list[Path] = []
    try:
        figure_paths = build_self_training_figure_report(
            frame,
            comparison_frame,
            iteration_logs,
            layout["figures"],
        )
    except Exception as exc:
        warning = f"figure generation failed: {type(exc).__name__}: {exc}"
        print(f"WARNING: {warning}")
        (layout["logs"] / "figure_error.txt").write_text(
            warning + "\n", encoding="utf-8"
        )

    return {
        "csv": csv_path,
        "json": json_path,
        "report": report_path,
        "comparison_csv": comparison_paths["csv"] if comparison_paths else None,
        "comparison_json": comparison_paths["json"] if comparison_paths else None,
        "figures": layout["figures"],
        "figure_paths": figure_paths,
        "comparison": comparison_frame,
        "comparison_summary": comparison_summary,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for the Phase 6 self-training run."""
    parser = argparse.ArgumentParser(
        description="Run the Phase 6 semi-supervised self-training grid."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments already completed in the results table.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        frame, summaries = run_self_training_grid(config, resume=args.resume)
        paths = save_self_training_outputs(frame, summaries, config)
    except Exception as exc:
        print(f"Phase 6 execution failed: {type(exc).__name__}: {exc}")
        return 1

    completed = int((frame["status"] == "completed").sum())
    failed = len(frame) - completed
    degenerate = (
        int(frame["numerically_degenerate_fit"].fillna(False).astype(bool).sum())
        if "numerically_degenerate_fit" in frame.columns
        else 0
    )
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Experiments recorded : {len(frame)}")
    print(f"Completed            : {completed}")
    print(f"Failed               : {failed}")
    print(f"Degenerate fits      : {degenerate}")
    print(f"Results CSV          : {paths['csv']}")
    print(f"Results JSON         : {paths['json']}")
    print(f"Comparison CSV       : {paths['comparison_csv']}")
    print(f"Report               : {paths['report']}")
    print(f"Figures              : {paths['figures']} "
          f"({len(paths['figure_paths'])} files)")
    if failed:
        overall = "ACTION REQUIRED (see failed experiments)"
    elif degenerate:
        overall = "COMPLETED WITH FLAGGED NUMERICALLY DEGENERATE FITS"
    else:
        overall = "PASS"
    print(f"Overall              : {overall}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
