"""Phase 5 supervised experiment orchestration.

Experiment grid
---------------
``3 datasets x 3 labelled proportions x 3 feature representations x
3 classifiers = 81 experiments``.

For each dataset the runner calls the Phase 4 dimensionality-reduction suite
exactly once. PCA is therefore fitted once per dataset on the complete training
pool, and LDA is fitted once per labelled proportion on labelled rows only.
Every classifier and proportion reuses those fitted transformations; PCA is
never refitted per classifier, and neither PCA nor LDA sees test rows.

Classifiers are fitted on the labelled training rows of the current proportion
only. The held-out test rows are used exclusively for prediction and metrics.
Because the Phase 4 objects are not persisted to disk, they are held in memory
for the duration of this run and released before the next dataset.

Run from the project root:

    python -m src.experiment_runner --config config/config.yaml
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from .classifiers import (
    CLASSIFIER_LABELS,
    CLASSIFIER_NAMES,
    build_classifier,
    list_classifiers,
)
from .dimensionality_reduction import (
    DimensionalityReductionSuite,
    run_dimensionality_reduction,
)
from .evaluation import (
    evaluate_predictions,
    save_metrics,
    to_json_records,
)
from .preprocessing import (
    EXPECTED_FEATURE_COUNTS,
    PreprocessedDataset,
    SemiSupervisedSplit,
    preprocess_dataset,
)
from .visualization import build_classifier_figure_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SECTIONS: tuple[str, ...] = (
    "project",
    "data",
    "dimensionality_reduction",
    "classifiers",
    "self_training",
    "evaluation",
    "results",
)

DATASET_ORDER: tuple[str, ...] = ("breast_cancer", "mnist", "har")
DATASET_LABELS: dict[str, str] = {
    "breast_cancer": "Breast Cancer",
    "mnist": "MNIST",
    "har": "HAR",
}
REPRESENTATION_ORDER: tuple[str, ...] = ("baseline", "pca", "lda")
LABELLED_PERCENTAGES: tuple[int, ...] = (10, 25, 50)
DEFAULT_OUTPUT_DIR = Path("results/classifier_evaluation")

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
    "status",
    "n_labelled_train",
    "n_test",
    "fit_feature_sha256",
    "fit_label_sha256",
    "prediction_file",
    "warning_message",
    "error_message",
)


@dataclass
class ExperimentRecord:
    """One completed or failed supervised experiment."""

    dataset: str
    labelled_percentage: int
    classifier: str
    feature_representation: str
    original_feature_count: int | None
    transformed_feature_count: int | None
    accuracy: float | None
    macro_f1: float | None
    training_time_seconds: float | None
    status: str
    n_labelled_train: int
    n_test: int
    fit_feature_sha256: str | None
    fit_label_sha256: str | None
    prediction_file: str | None = None
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
            "status": self.status,
            "n_labelled_train": self.n_labelled_train,
            "n_test": self.n_test,
            "fit_feature_sha256": self.fit_feature_sha256,
            "fit_label_sha256": self.fit_label_sha256,
            "prediction_file": self.prediction_file,
            "warning_message": self.warning_message,
            "error_message": self.error_message,
        }
        return {column: row[column] for column in RESULT_COLUMNS}


# ---------------------------------------------------------------------------
# Configuration and reproducibility helpers
# ---------------------------------------------------------------------------
def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load and minimally validate the YAML experiment configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")

    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Configuration is missing required sections: {missing}")
    return config


def set_global_seed(seed: int) -> None:
    """Seed Python's ``random`` module and NumPy's global RNG."""
    random.seed(int(seed))
    np.random.seed(int(seed))


def _enabled_datasets(config: dict[str, Any]) -> list[str]:
    enabled = [
        name
        for name in DATASET_ORDER
        if bool(
            (config["data"].get("datasets", {}).get(name, {}) or {}).get(
                "enabled", True
            )
        )
    ]
    if not enabled:
        raise ValueError("No datasets are enabled in config.data.datasets.")
    return enabled


def _output_layout(config: dict[str, Any]) -> dict[str, Path]:
    base = Path(
        config.get("results", {}).get(
            "classifier_evaluation_dir", str(DEFAULT_OUTPUT_DIR)
        )
    )
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    layout = {
        "base": base,
        "tables": base / "tables",
        "reports": base / "reports",
        "predictions": base / "predictions",
        "figures": base / "figures",
    }
    for directory in layout.values():
        directory.mkdir(parents=True, exist_ok=True)
    return layout


# ---------------------------------------------------------------------------
# Provenance fingerprints
# ---------------------------------------------------------------------------
def fingerprint_features(features: np.ndarray) -> str:
    """SHA-256 of a feature matrix's dtype, shape, and values."""
    array = np.ascontiguousarray(np.asarray(features))
    if array.dtype == object:
        raise TypeError("Feature matrices must have a numeric dtype for hashing.")
    digest = hashlib.sha256()
    digest.update(b"features")
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def fingerprint_labels(labels: np.ndarray) -> str:
    """SHA-256 of labels, independent of object-array memory addresses."""
    array = np.asarray(labels)
    digest = hashlib.sha256()
    digest.update(b"labels")
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    for value in array.tolist():
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Feature views
# ---------------------------------------------------------------------------
def get_feature_views(
    preprocessed: PreprocessedDataset,
    suite: DimensionalityReductionSuite,
    split: SemiSupervisedSplit,
    feature_representation: str,
    labelled_proportion: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return labelled training features, test features, and reduced width.

    Every returned training matrix contains exactly the labelled rows of the
    current proportion. Test rows are returned separately for evaluation only.
    """
    if feature_representation not in REPRESENTATION_ORDER:
        raise ValueError(
            f"Unknown feature representation {feature_representation!r}; "
            f"choose from {REPRESENTATION_ORDER}."
        )

    if feature_representation == "baseline":
        X_train = preprocessed.X_train[split.labelled_indices]
        X_test = preprocessed.X_test
        transformed_count = preprocessed.n_features
    elif feature_representation == "pca":
        if suite.pca is None:
            raise RuntimeError("PCA result is missing from the Phase 4 suite.")
        X_train = suite.pca.X_train[split.labelled_indices]
        X_test = suite.pca.X_test
        transformed_count = suite.pca.n_components
    else:
        result = suite.lda.get(float(labelled_proportion))
        if result is None:
            raise RuntimeError(
                f"LDA result is missing for labelled proportion "
                f"{labelled_proportion:.2f}."
            )
        X_train = result.X_train[split.labelled_indices]
        X_test = result.X_test
        transformed_count = result.n_components

    if X_train.shape[0] != split.n_labelled:
        raise RuntimeError(
            f"Feature view has {X_train.shape[0]} training rows but the labelled "
            f"subset has {split.n_labelled}."
        )
    if X_test.shape[0] != preprocessed.n_test:
        raise RuntimeError(
            f"Feature view has {X_test.shape[0]} test rows but the test set has "
            f"{preprocessed.n_test}."
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise RuntimeError(
            "Training and test feature widths differ for "
            f"{feature_representation}: {X_train.shape[1]} vs {X_test.shape[1]}."
        )
    return X_train, X_test, int(transformed_count)


def get_self_training_views(
    preprocessed: PreprocessedDataset,
    suite: DimensionalityReductionSuite,
    split: SemiSupervisedSplit,
    feature_representation: str,
    labelled_proportion: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return labelled, unlabelled, and test views for self-training.

    The labelled and unlabelled matrices both come from the same Phase 4
    transformation (baseline, PCA, or LDA), so unlabelled rows are projected
    with exactly the transformation fitted for the labelled rows. No
    transformation is fitted here, and test rows are returned separately for
    final evaluation only.
    """
    if feature_representation not in REPRESENTATION_ORDER:
        raise ValueError(
            f"Unknown feature representation {feature_representation!r}; "
            f"choose from {REPRESENTATION_ORDER}."
        )

    if feature_representation == "baseline":
        source_train = preprocessed.X_train
        X_test = preprocessed.X_test
        transformed_count = preprocessed.n_features
    elif feature_representation == "pca":
        if suite.pca is None:
            raise RuntimeError("PCA result is missing from the Phase 4 suite.")
        source_train = suite.pca.X_train
        X_test = suite.pca.X_test
        transformed_count = suite.pca.n_components
    else:
        result = suite.lda.get(float(labelled_proportion))
        if result is None:
            raise RuntimeError(
                f"LDA result is missing for labelled proportion "
                f"{labelled_proportion:.2f}."
            )
        source_train = result.X_train
        X_test = result.X_test
        transformed_count = result.n_components

    X_labelled = source_train[split.labelled_indices]
    X_unlabelled = source_train[split.unlabelled_indices]

    if X_labelled.shape[0] != split.n_labelled:
        raise RuntimeError(
            f"Labelled view has {X_labelled.shape[0]} rows but the split has "
            f"{split.n_labelled}."
        )
    if X_unlabelled.shape[0] != split.n_unlabelled:
        raise RuntimeError(
            f"Unlabelled view has {X_unlabelled.shape[0]} rows but the split has "
            f"{split.n_unlabelled}."
        )
    if X_test.shape[0] != preprocessed.n_test:
        raise RuntimeError(
            f"Test view has {X_test.shape[0]} rows but the test set has "
            f"{preprocessed.n_test}."
        )
    if not (
        X_labelled.shape[1] == X_unlabelled.shape[1] == X_test.shape[1]
    ):
        raise RuntimeError(
            "Feature widths differ across labelled, unlabelled, and test views "
            f"for {feature_representation}."
        )
    return (
        X_labelled,
        X_unlabelled,
        X_test,
        np.asarray(split.unlabelled_indices),
        int(transformed_count),
    )


# ---------------------------------------------------------------------------
# Experiment execution
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
            "y_pred": y_pred,
        }
    )
    path = predictions_dir / f"{experiment_id}.csv"
    frame.to_csv(path, index=False)
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _warning_text(warnings_list: Sequence[Any]) -> str | None:
    messages = []
    for warning in warnings_list:
        message = f"{warning.category.__name__}: {warning.message}"
        if message not in messages:
            messages.append(message)
    if not messages:
        return None
    text = "; ".join(messages)
    return text[:1000]


def run_single_experiment(
    preprocessed: PreprocessedDataset,
    split: SemiSupervisedSplit,
    X_fit: np.ndarray,
    X_test: np.ndarray,
    transformed_count: int,
    classifier_name: str,
    feature_representation: str,
    config: dict[str, Any],
    predictions_dir: Path,
    random_state: int,
) -> ExperimentRecord:
    """Fit one classifier and evaluate it on the held-out test set."""
    labelled_percentage = int(round(split.proportion * 100))
    y_fit = split.y_labelled
    y_test = preprocessed.y_test
    fit_feature_sha256 = fingerprint_features(X_fit)
    fit_label_sha256 = fingerprint_labels(y_fit)
    training_time: float | None = None

    base = {
        "dataset": preprocessed.name,
        "labelled_percentage": labelled_percentage,
        "classifier": classifier_name,
        "feature_representation": feature_representation,
        "original_feature_count": preprocessed.n_features,
        "transformed_feature_count": int(transformed_count),
        "n_labelled_train": int(split.n_labelled),
        "n_test": preprocessed.n_test,
        "fit_feature_sha256": fit_feature_sha256,
        "fit_label_sha256": fit_label_sha256,
    }

    try:
        model = build_classifier(
            classifier_name,
            config.get("classifiers", {}).get(classifier_name, {}),
            random_state=random_state,
        )
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_fit, y_fit)
        training_time = time.perf_counter() - start

        y_pred = np.asarray(model.predict(X_test))
        if y_pred.shape[0] != preprocessed.n_test:
            raise RuntimeError(
                f"Model returned {y_pred.shape[0]} predictions for "
                f"{preprocessed.n_test} test rows."
            )
        metrics = evaluate_predictions(y_test, y_pred)

        record = ExperimentRecord(
            **base,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            training_time_seconds=float(training_time),
            status="completed",
            warning_message=_warning_text(caught),
        )
        record.prediction_file = _save_predictions(
            record.experiment_id, preprocessed, y_pred, predictions_dir
        )
        return record
    except Exception as exc:
        return ExperimentRecord(
            **base,
            accuracy=None,
            macro_f1=None,
            training_time_seconds=(
                float(training_time) if training_time is not None else None
            ),
            status="failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )


def _failed_record(
    dataset: str,
    labelled_percentage: int,
    classifier: str,
    feature_representation: str,
    original_feature_count: int | None,
    error_message: str,
) -> ExperimentRecord:
    return ExperimentRecord(
        dataset=dataset,
        labelled_percentage=labelled_percentage,
        classifier=classifier,
        feature_representation=feature_representation,
        original_feature_count=original_feature_count,
        transformed_feature_count=None,
        accuracy=None,
        macro_f1=None,
        training_time_seconds=None,
        status="failed",
        n_labelled_train=0,
        n_test=0,
        fit_feature_sha256=None,
        fit_label_sha256=None,
        error_message=error_message,
    )


def _dataset_summary(
    preprocessed: PreprocessedDataset,
    suite: DimensionalityReductionSuite,
) -> dict[str, Any]:
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


def run_experiment_grid(
    config: dict[str, Any],
) -> tuple[list[ExperimentRecord], dict[str, Any]]:
    """Execute all 81 supervised experiments and return records and metadata."""
    seed = int(config["project"]["random_seed"])
    set_global_seed(seed)
    datasets = _enabled_datasets(config)
    classifiers = list_classifiers()
    layout = _output_layout(config)
    proportions = [float(value) for value in config["data"]["labelled_proportions"]]
    expected = len(datasets) * len(proportions) * len(REPRESENTATION_ORDER) * len(classifiers)

    print("=" * 78)
    print("Phase 5 supervised classifier evaluation")
    print(f"Datasets        : {datasets}")
    print(f"Labelled %      : {[int(round(value * 100)) for value in proportions]}")
    print(f"Representations : {list(REPRESENTATION_ORDER)}")
    print(f"Classifiers     : {classifiers}")
    print(f"Expected        : {expected} experiments")
    print(f"Seed            : {seed}")
    print("=" * 78)

    records: list[ExperimentRecord] = []
    summaries: dict[str, Any] = {}
    counter = 0

    for dataset_name in datasets:
        print(f"\nDataset: {DATASET_LABELS.get(dataset_name, dataset_name)}")
        try:
            preprocessed = preprocess_dataset(dataset_name, config)
            suite = run_dimensionality_reduction(preprocessed, config)
            if suite.pca is None or not suite.lda:
                raise RuntimeError(
                    "Phase 4 suite is missing the PCA or LDA transformations "
                    "required by Phase 5."
                )
            summaries[dataset_name] = _dataset_summary(preprocessed, suite)
        except Exception as exc:
            error = f"dataset setup failed: {type(exc).__name__}: {exc}"
            print(f"  [FAIL] {error}")
            expected_features = EXPECTED_FEATURE_COUNTS.get(dataset_name)
            for proportion in proportions:
                for representation in REPRESENTATION_ORDER:
                    for classifier in classifiers:
                        counter += 1
                        records.append(
                            _failed_record(
                                dataset_name,
                                int(round(proportion * 100)),
                                classifier,
                                representation,
                                expected_features,
                                error,
                            )
                        )
            summaries[dataset_name] = {"error": error}
            continue

        for proportion in sorted(preprocessed.semi_supervised):
            split = preprocessed.semi_supervised[proportion]
            for representation in REPRESENTATION_ORDER:
                for classifier in classifiers:
                    counter += 1
                    experiment_id = (
                        f"{dataset_name}__labelled_"
                        f"{int(round(proportion * 100)):02d}__"
                        f"{representation}__{classifier}"
                    )
                    try:
                        X_fit, X_test, transformed_count = get_feature_views(
                            preprocessed, suite, split, representation, proportion
                        )
                        record = run_single_experiment(
                            preprocessed=preprocessed,
                            split=split,
                            X_fit=X_fit,
                            X_test=X_test,
                            transformed_count=transformed_count,
                            classifier_name=classifier,
                            feature_representation=representation,
                            config=config,
                            predictions_dir=layout["predictions"],
                            random_state=seed,
                        )
                    except Exception as exc:
                        record = _failed_record(
                            dataset_name,
                            int(round(proportion * 100)),
                            classifier,
                            representation,
                            preprocessed.n_features,
                            f"{type(exc).__name__}: {exc}",
                        )
                    records.append(record)
                    if record.status == "completed":
                        print(
                            f"  [{counter:>2}/{expected}] {experiment_id} "
                            f"accuracy={record.accuracy:.4f} "
                            f"macro_f1={record.macro_f1:.4f} "
                            f"fit={record.training_time_seconds:.4f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"  [{counter:>2}/{expected}] {experiment_id} "
                            f"FAILED: {record.error_message}",
                            flush=True,
                        )

        del suite
        del preprocessed
        gc.collect()

    return records, summaries


# ---------------------------------------------------------------------------
# Persistence and reporting
# ---------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _format_group_table(frame: pd.DataFrame, group_column: str, label: str) -> list[str]:
    completed = frame[frame["status"] == "completed"]
    lines = [f"{label}", "-" * 78]
    if completed.empty:
        return lines + ["No completed experiments."]
    grouped = (
        completed.groupby(group_column, sort=True)
        .agg(
            experiments=("accuracy", "size"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            training_time_mean=("training_time_seconds", "mean"),
        )
        .reset_index()
    )
    lines.append(
        f"{group_column:<26} {'n':>4} {'accuracy':>10} {'macro_f1':>10} "
        f"{'fit_seconds':>12}"
    )
    for row in grouped.itertuples(index=False):
        lines.append(
            f"{str(getattr(row, group_column)):<26} "
            f"{int(row.experiments):>4} "
            f"{float(row.accuracy_mean):>10.4f} "
            f"{float(row.macro_f1_mean):>10.4f} "
            f"{float(row.training_time_mean):>12.4f}"
        )
    return lines


def build_phase5_report(
    frame: pd.DataFrame,
    summaries: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Build the textual Phase 5 results report from measured rows only."""
    completed = frame[frame["status"] == "completed"]
    failed = frame[frame["status"] != "completed"]
    lines = [
        "Phase 5 supervised classifier evaluation report",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"Total experiments recorded : {len(frame)}",
        f"Completed experiments      : {len(completed)}",
        f"Failed experiments         : {len(failed)}",
        f"Expected experiments       : 81 (3 datasets x 3 proportions x 3 "
        "representations x 3 classifiers)",
        "",
        "This phase is the supervised baseline. No unlabelled data, "
        "pseudo-labelling, self-training, or test fitting is used.",
        "",
    ]

    for label, column in (
        ("Results by dataset", "dataset"),
        ("Results by classifier", "classifier"),
        ("Results by feature representation", "feature_representation"),
        ("Results by labelled percentage", "labelled_percentage"),
    ):
        lines.extend(_format_group_table(frame, column, label))
        lines.append("")

    lines.extend(["Best-performing configuration per dataset", "-" * 78])
    if completed.empty:
        lines.append("No completed experiments.")
    else:
        for dataset in DATASETS_IN_REPORT:
            subset = completed[completed["dataset"] == dataset]
            if subset.empty:
                lines.append(f"[{dataset}] no completed experiments")
                continue
            best = subset.sort_values(
                ["macro_f1", "accuracy"], ascending=[False, False]
            ).iloc[0]
            lines.append(
                f"[{dataset}] {best['feature_representation']} + "
                f"{CLASSIFIER_LABELS.get(best['classifier'], best['classifier'])} "
                f"at {int(best['labelled_percentage'])}% labelled: "
                f"accuracy={best['accuracy']:.4f}, macro_f1={best['macro_f1']:.4f} "
                f"(fit {best['training_time_seconds']:.4f}s)"
            )
    lines.append("")

    lines.extend(["Training-time observations", "-" * 78])
    if completed.empty:
        lines.append("No completed experiments.")
    else:
        for classifier in CLASSIFIER_NAMES:
            subset = completed[completed["classifier"] == classifier]
            if subset.empty:
                continue
            lines.append(
                f"{CLASSIFIER_LABELS.get(classifier, classifier):<24} "
                f"mean={subset['training_time_seconds'].mean():.4f}s, "
                f"min={subset['training_time_seconds'].min():.4f}s, "
                f"max={subset['training_time_seconds'].max():.4f}s"
            )
        lines.append(
            f"Total measured fit time{' ' * 4}: "
            f"{completed['training_time_seconds'].sum():.2f}s"
        )
        lines.append(
            "KNN fitting only stores the training data; its neighbour search "
            "happens during prediction and is not part of training_time_seconds."
        )
    lines.append("")

    lines.extend(["Failed or incomplete experiments", "-" * 78])
    if failed.empty:
        lines.append("None. All experiments completed successfully.")
    else:
        for row in failed.itertuples(index=False):
            lines.append(
                f"{row.experiment_id}: {row.error_message or 'unknown error'}"
            )
    lines.append("")

    warnings_present = completed["warning_message"].dropna()
    lines.extend(["Training warnings", "-" * 78])
    lines.append(
        f"{len(warnings_present)} completed experiment(s) emitted warnings."
        if not warnings_present.empty
        else "No training warnings were recorded."
    )
    if not warnings_present.empty:
        for message in warnings_present.head(10):
            lines.append(f"- {message}")
    lines.append("")

    lines.extend(
        [
            "Limitations and observations",
            "-" * 78,
            "* One fixed train/test split per dataset (seed 42); no repeated "
            "sampling or confidence intervals.",
            "* No hyperparameter tuning is performed in Phase 5.",
            "* The SVM is deliberately linear (LinearSVC) for tractability on "
            "MNIST and provides hard labels only.",
            "* The LDA representation is fitted per labelled proportion because "
            "it is supervised; PCA is fitted once per dataset on the complete "
            "training pool.",
            "* Macro F1 gives each class equal weight, which is important for "
            "imbalanced datasets.",
            "* Test rows are used only for prediction and metric computation; "
            "fit-input SHA-256 fingerprints are stored for every experiment.",
            "",
            "Dataset summaries",
            "-" * 78,
        ]
    )
    for dataset, summary in summaries.items():
        lines.append(f"[{dataset}] {json.dumps(_json_safe(summary), sort_keys=True)}")

    return "\n".join(lines) + "\n"


DATASETS_IN_REPORT: tuple[str, ...] = DATASET_ORDER


def save_experiment_outputs(
    records: Sequence[ExperimentRecord],
    summaries: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Path]:
    """Write CSV, JSON, report, predictions, and figures; return paths."""
    layout = _output_layout(config)
    rows = [record.to_row() for record in records]
    frame = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))

    csv_path = save_metrics(frame, layout["tables"] / "classifier_results.csv")
    completed = frame[frame["status"] == "completed"]
    failed = frame[frame["status"] != "completed"]

    warnings_present = int(completed["warning_message"].notna().sum())
    payload = {
        "phase": 5,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": {
            "random_seed": int(config["project"]["random_seed"]),
            "expected_experiments": 81,
            "recorded_experiments": int(len(frame)),
            "completed_experiments": int(len(completed)),
            "failed_experiments": int(len(failed)),
            "experiments_with_warnings": warnings_present,
            "classifiers": {
                name: CLASSIFIER_LABELS.get(name, name) for name in CLASSIFIER_NAMES
            },
            "feature_representations": list(REPRESENTATION_ORDER),
            "labelled_percentages": list(LABELLED_PERCENTAGES),
            "metrics": ["accuracy", "macro_f1"],
            "datasets": summaries,
        },
        "experiments": to_json_records(frame),
    }
    json_path = layout["tables"] / "classifier_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")

    report_path = layout["reports"] / "phase5_classifier_report.txt"
    report_path.write_text(
        build_phase5_report(frame, summaries, config), encoding="utf-8"
    )

    figure_paths = build_classifier_figure_report(frame, layout["figures"])

    return {
        "csv": csv_path,
        "json": json_path,
        "report": report_path,
        "predictions": layout["predictions"],
        "figures": layout["figures"],
        "figure_paths": figure_paths,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run the Phase 5 supervised classifier experiment grid."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m src.experiment_runner``."""
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        records, summaries = run_experiment_grid(config)
        paths = save_experiment_outputs(records, summaries, config)
    except Exception as exc:
        print(f"Phase 5 execution failed: {type(exc).__name__}: {exc}")
        return 1

    completed = sum(record.status == "completed" for record in records)
    failed = len(records) - completed
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Experiments recorded : {len(records)}")
    print(f"Completed            : {completed}")
    print(f"Failed               : {failed}")
    print(f"Results CSV          : {paths['csv']}")
    print(f"Results JSON         : {paths['json']}")
    print(f"Report               : {paths['report']}")
    print(f"Predictions          : {paths['predictions']}")
    print(f"Figures              : {paths['figures']} "
          f"({len(paths['figure_paths'])} files)")
    print(f"Overall              : {'PASS' if failed == 0 else 'ACTION REQUIRED'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
