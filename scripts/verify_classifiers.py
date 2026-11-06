"""Verify the Phase 5 supervised classifier experiments.

Run from the project root:

    python scripts/verify_classifiers.py

The script reuses the fixed Phase 3 splits and the Phase 4 transformations,
reads the saved Phase 5 artifacts, and checks that:

* all 3 datasets, 3 labelled percentages, 3 feature representations, and
  3 classifiers were evaluated (81 experiments, no duplicates or gaps);
* every completed experiment reports finite accuracy and macro F1 in [0, 1]
  and a non-negative model-fitting time;
* stored metrics reproduce exactly from the saved test predictions;
* predictions cover the fixed held-out test set with valid class labels;
* baseline widths equal the original feature count, and PCA/LDA widths equal
  the Phase 4 components with the ``n_classes - 1`` LDA cap respected;
* every classifier was fitted on exactly the labelled training rows of its
  proportion, proven by re-deriving the fit arrays and SHA-256 fingerprints;
* PCA was fitted on the complete training pool and LDA per proportion on
  labelled rows only, so neither saw test rows or unlabelled true labels;
* the CSV, strict JSON, report, and 15 figures exist.

No classifier is refitted and no dataset is downloaded or modified. The Phase 4
transformations are recomputed only to reconstruct the expected fit inputs.

Exit code 0 means every check passed; 1 means at least one check failed.
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.classifiers import CLASSIFIER_NAMES
from src.dimensionality_reduction import run_dimensionality_reduction
from src.experiment_runner import (
    DATASET_ORDER,
    LABELLED_PERCENTAGES,
    REPRESENTATION_ORDER,
    RESULT_COLUMNS,
    fingerprint_features,
    fingerprint_labels,
    get_feature_views,
    load_config,
)
from src.preprocessing import CheckResult, preprocess_dataset

SEPARATOR = "=" * 78
OUTPUT_ROOT = PROJECT_ROOT / "results" / "classifier_evaluation"
TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_DIR = OUTPUT_ROOT / "reports"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
FIGURE_DIR = OUTPUT_ROOT / "figures"
CSV_PATH = TABLE_DIR / "classifier_results.csv"
JSON_PATH = TABLE_DIR / "classifier_results.json"
RESULTS_REPORT_PATH = REPORT_DIR / "phase5_classifier_report.txt"
VERIFICATION_REPORT_PATH = REPORT_DIR / "phase5_verification_report.txt"

EXPECTED_EXPERIMENTS = 81
METRIC_TOLERANCE = 1e-9
REQUIRED_COLUMNS = (
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
)
FIGURE_TEMPLATES = (
    "accuracy_by_representation_{dataset}.png",
    "macro_f1_by_representation_{dataset}.png",
    "classifier_comparison_{dataset}.png",
    "performance_by_labelled_proportion_{dataset}.png",
    "training_time_by_classifier_{dataset}.png",
)


def _check(
    checks: list[CheckResult],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append(CheckResult(name=name, passed=bool(passed), detail=detail))


def _summarise(
    checks: list[CheckResult],
    name: str,
    failures: Sequence[str],
    total: int,
) -> None:
    """Record one check covering ``total`` items and list the failures."""
    if failures:
        shown = " | ".join(failures[:6])
        if len(failures) > 6:
            shown += f" | (+{len(failures) - 6} more)"
        detail = f"{len(failures)}/{total} failed: {shown}"
    else:
        detail = f"{total}/{total} passed"
    _check(checks, name, not failures, detail)


def _enabled_datasets(config: dict[str, Any]) -> list[str]:
    return [
        name
        for name in DATASET_ORDER
        if bool(
            (config["data"]["datasets"].get(name, {}) or {}).get("enabled", True)
        )
    ]


def _configured_percentages(config: dict[str, Any]) -> list[int]:
    return [
        int(round(float(value) * 100))
        for value in config["data"]["labelled_proportions"]
    ]


def _expected_keys(
    datasets: Sequence[str],
    percentages: Sequence[int],
) -> set[tuple[str, int, str, str]]:
    return {
        (dataset, percentage, representation, classifier)
        for dataset in datasets
        for percentage in percentages
        for representation in REPRESENTATION_ORDER
        for classifier in CLASSIFIER_NAMES
    }


def _actual_keys(frame: pd.DataFrame) -> set[tuple[str, int, str, str]]:
    return {
        (
            str(row.dataset),
            int(row.labelled_percentage),
            str(row.feature_representation),
            str(row.classifier),
        )
        for row in frame.itertuples(index=False)
    }


def _resolve_project_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


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
            _check(checks, "Results JSON is strict and parseable", True, str(JSON_PATH))
        except Exception as exc:
            _check(
                checks,
                "Results JSON is strict and parseable",
                False,
                f"{type(exc).__name__}: {exc}",
            )
    else:
        _check(checks, "Results JSON is strict and parseable", False, f"Missing {JSON_PATH}")

    return frame, payload


def _check_schema_and_coverage(
    frame: pd.DataFrame,
    datasets: Sequence[str],
    percentages: Sequence[int],
    checks: list[CheckResult],
) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    _check(
        checks,
        "Required result columns are present",
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

    duplicates = frame.duplicated(
        subset=["dataset", "labelled_percentage", "feature_representation", "classifier"]
    ).sum()
    _check(
        checks,
        "No duplicate experiment conditions",
        int(duplicates) == 0,
        f"duplicate rows={int(duplicates)}",
    )

    expected = _expected_keys(datasets, percentages)
    actual = _actual_keys(frame)
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

    completed = int((frame["status"] == "completed").sum())
    failed = int((frame["status"] != "completed").sum())
    _check(
        checks,
        "Every experiment completed",
        completed == EXPECTED_EXPERIMENTS and failed == 0,
        f"completed={completed}, failed={failed}",
    )

    for dataset in datasets:
        subset = frame[frame["dataset"] == dataset]
        expected_dataset = len(percentages) * len(REPRESENTATION_ORDER) * len(
            CLASSIFIER_NAMES
        )
        _check(
            checks,
            f"[{dataset}] experiment coverage",
            len(subset) == expected_dataset,
            f"rows={len(subset)} (expected {expected_dataset})",
        )

    for percentage in percentages:
        subset = frame[frame["labelled_percentage"] == percentage]
        expected_percentage = len(datasets) * len(REPRESENTATION_ORDER) * len(
            CLASSIFIER_NAMES
        )
        _check(
            checks,
            f"[{percentage}%] experiment coverage",
            len(subset) == expected_percentage,
            f"rows={len(subset)} (expected {expected_percentage})",
        )


def _check_json_payload(
    frame: pd.DataFrame,
    payload: dict[str, Any],
    checks: list[CheckResult],
) -> None:
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    experiments = payload.get("experiments", []) if isinstance(payload, dict) else []
    _check(
        checks,
        "JSON records match the CSV row count",
        len(experiments) == len(frame) == EXPECTED_EXPERIMENTS,
        f"json={len(experiments)}, csv={len(frame)}",
    )
    _check(
        checks,
        "JSON metadata reports 81 completed experiments",
        int(metadata.get("expected_experiments", -1)) == EXPECTED_EXPERIMENTS
        and int(metadata.get("completed_experiments", -1)) == EXPECTED_EXPERIMENTS
        and int(metadata.get("failed_experiments", -1)) == 0,
        (
            f"expected={metadata.get('expected_experiments')}, "
            f"completed={metadata.get('completed_experiments')}, "
            f"failed={metadata.get('failed_experiments')}"
        ),
    )
    json_ids = {str(record.get("experiment_id")) for record in experiments}
    csv_ids = {str(value) for value in frame["experiment_id"]}
    _check(
        checks,
        "JSON and CSV experiment identifiers agree",
        json_ids == csv_ids,
        f"json={len(json_ids)}, csv={len(csv_ids)}",
    )


def _check_metrics(
    dataset_frame: pd.DataFrame,
    checks: list[CheckResult],
) -> None:
    total = len(dataset_frame)
    failures: list[str] = []
    for row in dataset_frame.itertuples(index=False):
        identifier = (
            f"{row.dataset}/{row.labelled_percentage}%/{row.feature_representation}/"
            f"{row.classifier}"
        )
        if row.status != "completed":
            failures.append(f"{identifier}: status={row.status}")
            continue
        for metric in ("accuracy", "macro_f1"):
            value = getattr(row, metric)
            if value is None or not np.isfinite(float(value)):
                failures.append(f"{identifier}: {metric}={value!r}")
            elif not 0.0 <= float(value) <= 1.0:
                failures.append(f"{identifier}: {metric}={value!r} outside [0,1]")
    _summarise(checks, "Accuracy/macro-F1 are finite and in [0,1]", failures, total)

    time_failures = [
        f"{row.dataset}/{row.labelled_percentage}%/{row.feature_representation}/"
        f"{row.classifier}: {row.training_time_seconds!r}"
        for row in dataset_frame.itertuples(index=False)
        if row.training_time_seconds is None
        or not np.isfinite(float(row.training_time_seconds))
        or float(row.training_time_seconds) < 0.0
    ]
    _summarise(
        checks,
        "Training times are finite and non-negative",
        time_failures,
        total,
    )


def _check_dimensions(
    preprocessed: Any,
    suite: Any,
    dataset_frame: pd.DataFrame,
    checks: list[CheckResult],
) -> None:
    failures: list[str] = []
    for row in dataset_frame.itertuples(index=False):
        identifier = (
            f"{row.labelled_percentage}%/{row.feature_representation}/{row.classifier}"
        )
        if int(row.original_feature_count) != preprocessed.n_features:
            failures.append(
                f"{identifier}: original={row.original_feature_count}, "
                f"expected={preprocessed.n_features}"
            )
        if row.feature_representation == "baseline":
            expected = preprocessed.n_features
        elif row.feature_representation == "pca":
            expected = suite.pca.n_components if suite.pca is not None else -1
        else:
            proportion = round(float(row.labelled_percentage) / 100.0, 2)
            result = suite.lda.get(proportion)
            expected = result.n_components if result is not None else -1
        if int(row.transformed_feature_count) != int(expected):
            failures.append(
                f"{identifier}: transformed={row.transformed_feature_count}, "
                f"expected={expected}"
            )
    _summarise(
        checks,
        "Feature widths match Phase 3/4 representations",
        failures,
        len(dataset_frame),
    )

    baseline_rows = dataset_frame[dataset_frame["feature_representation"] == "baseline"]
    baseline_failures = [
        f"{row.labelled_percentage}%: {row.transformed_feature_count}"
        for row in baseline_rows.itertuples(index=False)
        if int(row.transformed_feature_count) != preprocessed.n_features
    ]
    _summarise(
        checks,
        "Baseline dimensions equal original features",
        baseline_failures,
        len(baseline_rows),
    )

    if suite.pca is not None:
        pca_failures = [
            f"{row.labelled_percentage}%: {row.transformed_feature_count}"
            for row in dataset_frame[
                dataset_frame["feature_representation"] == "pca"
            ].itertuples(index=False)
            if int(row.transformed_feature_count) != suite.pca.n_components
        ]
        _summarise(
            checks,
            "PCA dimensions match Phase 4",
            pca_failures,
            int((dataset_frame["feature_representation"] == "pca").sum()),
        )
        _check(
            checks,
            "PCA fitted on the complete training pool (no test rows)",
            suite.pca.fitted_on == "complete_training_pool"
            and suite.pca.n_samples_fit == preprocessed.n_train,
            f"fitted_on={suite.pca.fitted_on}, n_fit={suite.pca.n_samples_fit}, "
            f"n_train={preprocessed.n_train}, n_test={preprocessed.n_test}",
        )

    lda_failures: list[str] = []
    cap_failures: list[str] = []
    for proportion, split in sorted(preprocessed.semi_supervised.items()):
        result = suite.lda.get(float(proportion))
        if result is None:
            lda_failures.append(f"{proportion:.2f}: missing LDA result")
            continue
        n_classes = int(np.unique(split.y_labelled).size)
        cap = min(preprocessed.n_features, n_classes - 1)
        rows = dataset_frame[
            (dataset_frame["feature_representation"] == "lda")
            & (dataset_frame["labelled_percentage"] == int(round(proportion * 100)))
        ]
        for row in rows.itertuples(index=False):
            if int(row.transformed_feature_count) != result.n_components:
                lda_failures.append(
                    f"{proportion:.2f}/{row.classifier}: "
                    f"{row.transformed_feature_count} != {result.n_components}"
                )
        if result.n_components > cap:
            cap_failures.append(f"{proportion:.2f}: {result.n_components} > {cap}")
        if result.n_samples_fit != split.n_labelled:
            cap_failures.append(
                f"{proportion:.2f}: fit rows {result.n_samples_fit} != "
                f"labelled rows {split.n_labelled}"
            )
        if not str(result.fitted_on).startswith("labelled_training_subset"):
            cap_failures.append(f"{proportion:.2f}: fitted_on={result.fitted_on!r}")
        class_counts = (result.metadata or {}).get("class_counts", {})
        if int(sum(class_counts.values())) != split.n_labelled:
            cap_failures.append(
                f"{proportion:.2f}: LDA class counts sum to "
                f"{sum(class_counts.values())}, not {split.n_labelled}"
            )

    _summarise(
        checks,
        "LDA dimensions match Phase 4",
        lda_failures,
        int((dataset_frame["feature_representation"] == "lda").sum()),
    )
    _summarise(
        checks,
        "LDA component cap and labelled-only fit scope",
        cap_failures,
        len(preprocessed.semi_supervised),
    )


def _check_fit_provenance(
    preprocessed: Any,
    suite: Any,
    dataset_frame: pd.DataFrame,
    checks: list[CheckResult],
) -> None:
    feature_failures: list[str] = []
    label_failures: list[str] = []
    row_failures: list[str] = []

    for row in dataset_frame.itertuples(index=False):
        identifier = (
            f"{row.labelled_percentage}%/{row.feature_representation}/{row.classifier}"
        )
        proportion = round(float(row.labelled_percentage) / 100.0, 2)
        split = preprocessed.semi_supervised[float(proportion)]
        X_fit, X_test, _ = get_feature_views(
            preprocessed, suite, split, str(row.feature_representation), float(proportion)
        )
        expected_feature_hash = fingerprint_features(X_fit)
        expected_label_hash = fingerprint_labels(split.y_labelled)

        if expected_feature_hash != str(row.fit_feature_sha256):
            feature_failures.append(identifier)
        if expected_label_hash != str(row.fit_label_sha256):
            label_failures.append(identifier)
        if X_fit.shape[0] != split.n_labelled or X_fit.shape[1] != int(
            row.transformed_feature_count
        ):
            row_failures.append(
                f"{identifier}: fit shape {X_fit.shape}, "
                f"labelled={split.n_labelled}, width={row.transformed_feature_count}"
            )
        if X_test.shape[0] != preprocessed.n_test:
            row_failures.append(
                f"{identifier}: test shape {X_test.shape}, "
                f"expected rows={preprocessed.n_test}"
            )
        if split.n_labelled + split.n_unlabelled != preprocessed.n_train:
            row_failures.append(
                f"{identifier}: labelled+unlabelled != training partition"
            )

    total = len(dataset_frame)
    _summarise(
        checks,
        "Fit features equal labelled-only training rows (SHA-256)",
        feature_failures,
        total,
    )
    _summarise(
        checks,
        "Fit labels equal labelled targets only (SHA-256)",
        label_failures,
        total,
    )
    _summarise(
        checks,
        "Fit/test shapes respect the fixed split",
        row_failures,
        total,
    )


def _check_predictions(
    preprocessed: Any,
    dataset_frame: pd.DataFrame,
    checks: list[CheckResult],
) -> None:
    file_failures: list[str] = []
    count_failures: list[str] = []
    label_failures: list[str] = []
    metric_failures: list[str] = []

    valid_labels = set(np.unique(preprocessed.y_train).tolist())

    for row in dataset_frame.itertuples(index=False):
        identifier = (
            f"{row.labelled_percentage}%/{row.feature_representation}/{row.classifier}"
        )
        path = _resolve_project_path(row.prediction_file)
        if path is None or not path.is_file():
            file_failures.append(f"{identifier}: {row.prediction_file!r}")
            continue
        try:
            predictions = pd.read_csv(path)
        except Exception as exc:
            file_failures.append(f"{identifier}: {type(exc).__name__}: {exc}")
            continue

        required = {"test_position", "test_index", "y_true", "y_pred"}
        if not required.issubset(predictions.columns):
            file_failures.append(f"{identifier}: missing prediction columns")
            continue

        if len(predictions) != preprocessed.n_test:
            count_failures.append(
                f"{identifier}: {len(predictions)} rows != {preprocessed.n_test}"
            )
            continue
        if not np.array_equal(
            predictions["test_index"].to_numpy(dtype=np.int64),
            preprocessed.test_indices.astype(np.int64),
        ):
            count_failures.append(f"{identifier}: test_index does not match fixed split")
        if not np.array_equal(
            predictions["y_true"].to_numpy(),
            preprocessed.y_test,
        ):
            count_failures.append(f"{identifier}: y_true does not match fixed test labels")
        if predictions[["y_true", "y_pred"]].isna().any().any():
            label_failures.append(f"{identifier}: missing labels")
            continue

        predicted = predictions["y_pred"].to_numpy()
        if not set(np.unique(predicted).tolist()).issubset(valid_labels):
            invalid = sorted(set(np.unique(predicted).tolist()).difference(valid_labels))
            label_failures.append(f"{identifier}: invalid labels {invalid}")

        recomputed_accuracy = float(
            accuracy_score(predictions["y_true"].to_numpy(), predicted)
        )
        recomputed_f1 = float(
            f1_score(
                predictions["y_true"].to_numpy(),
                predicted,
                average="macro",
                zero_division=0,
            )
        )
        if abs(recomputed_accuracy - float(row.accuracy)) > METRIC_TOLERANCE:
            metric_failures.append(
                f"{identifier}: accuracy {recomputed_accuracy} != {row.accuracy}"
            )
        if abs(recomputed_f1 - float(row.macro_f1)) > METRIC_TOLERANCE:
            metric_failures.append(
                f"{identifier}: macro_f1 {recomputed_f1} != {row.macro_f1}"
            )

    total = len(dataset_frame)
    _summarise(checks, "Prediction files exist", file_failures, total)
    _summarise(
        checks,
        "Predictions cover the fixed test set",
        count_failures,
        total,
    )
    _summarise(
        checks,
        "Predicted labels belong to the dataset classes",
        label_failures,
        total,
    )
    _summarise(
        checks,
        "Stored metrics reproduce from predictions",
        metric_failures,
        total,
    )


def _check_figures_and_report(checks: list[CheckResult]) -> None:
    missing_figures: list[str] = []
    for dataset in DATASET_ORDER:
        for template in FIGURE_TEMPLATES:
            path = FIGURE_DIR / template.format(dataset=dataset)
            if not path.is_file() or path.stat().st_size == 0:
                missing_figures.append(path.name)
    _check(
        checks,
        "All 15 Phase 5 figures exist",
        not missing_figures,
        (
            f"15 figures under {FIGURE_DIR}"
            if not missing_figures
            else f"Missing/empty: {missing_figures}"
        ),
    )

    if not RESULTS_REPORT_PATH.is_file():
        _check(checks, "Phase 5 report exists", False, f"Missing {RESULTS_REPORT_PATH}")
        return
    text = RESULTS_REPORT_PATH.read_text(encoding="utf-8")
    required_sections = (
        "Results by dataset",
        "Results by classifier",
        "Results by feature representation",
        "Results by labelled percentage",
        "Best-performing configuration per dataset",
        "Training-time observations",
        "Failed or incomplete experiments",
        "Limitations and observations",
    )
    missing_sections = [section for section in required_sections if section not in text]
    _check(
        checks,
        "Phase 5 report exists with all required sections",
        not missing_sections,
        (
            f"{RESULTS_REPORT_PATH}"
            if not missing_sections
            else f"Missing sections: {missing_sections}"
        ),
    )


def _write_report(
    checks_by_dataset: dict[str, list[CheckResult]],
    global_checks: Sequence[CheckResult],
    frame: pd.DataFrame,
    overall_passed: bool,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    completed = (
        int((frame["status"] == "completed").sum()) if "status" in frame.columns else 0
    )
    failed = (
        int((frame["status"] != "completed").sum()) if "status" in frame.columns else 0
    )
    lines = [
        "Phase 5 supervised classifier verification report",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "No classifiers were refitted. Phase 4 transformations were recomputed "
        "only to reconstruct the expected labelled-only fit inputs.",
        "",
        "Summary",
        "-" * 78,
        f"Results CSV          : {CSV_PATH}",
        f"Results JSON         : {JSON_PATH}",
        f"Prediction directory : {PREDICTION_DIR}",
        f"Figure directory     : {FIGURE_DIR}",
        f"Experiments recorded : {len(frame)}",
        f"Completed            : {completed}",
        f"Failed               : {failed}",
        "",
        "Checks",
        "-" * 78,
    ]

    total = 0
    passed = 0
    for dataset, checks in checks_by_dataset.items():
        lines.append(f"[{dataset}]")
        for check in checks:
            total += 1
            passed += int(check.passed)
            lines.append(
                f"  {'PASS' if check.passed else 'FAIL'}  {check.name}: {check.detail}"
            )
        lines.append("")

    lines.append("[global]")
    for check in global_checks:
        total += 1
        passed += int(check.passed)
        lines.append(
            f"  {'PASS' if check.passed else 'FAIL'}  {check.name}: {check.detail}"
        )

    lines.extend(
        [
            "",
            f"Checks passed     : {passed} / {total}",
            f"Overall verdict   : {'PASS' if overall_passed else 'FAIL'}",
            "",
            "Leakage controls",
            "-" * 78,
            "* Every classifier was fitted on exactly the labelled rows of its "
            "proportion; stored SHA-256 fingerprints were re-derived from the "
            "fixed Phase 3 split and match.",
            "* Stored fit-label fingerprints match the labelled targets only, so "
            "unlabelled true labels never entered classifier fitting.",
            "* PCA was fitted once per dataset on the complete training pool and "
            "LDA per proportion on labelled rows only; test rows never entered "
            "either fit.",
            "* Metrics were recomputed from the saved test predictions and match "
            "the stored values.",
            "",
            "Limitations",
            "-" * 78,
            "* Phase 5 uses one fixed train/test split and seed per dataset, so no "
            "confidence intervals are reported.",
            "* No hyperparameter tuning is performed.",
            "* The SVM is linear (LinearSVC) and provides hard labels only.",
            "* KNN fit time only stores the training data; neighbour search happens "
            "during prediction.",
        ]
    )

    path = REPORT_DIR / "phase5_verification_report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    datasets = _enabled_datasets(config)
    percentages = _configured_percentages(config)

    print(SEPARATOR)
    print("Phase 5 supervised classifier verification")
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Datasets         : {datasets}")
    print(f"Labelled %       : {percentages}")
    print(f"Representations  : {list(REPRESENTATION_ORDER)}")
    print(f"Classifiers      : {list(CLASSIFIER_NAMES)}")
    print(
        f"Expected grid    : {len(datasets)} x {len(percentages)} x "
        f"{len(REPRESENTATION_ORDER)} x {len(CLASSIFIER_NAMES)} = "
        f"{len(datasets) * len(percentages) * len(REPRESENTATION_ORDER) * len(CLASSIFIER_NAMES)}"
    )
    print(SEPARATOR)

    global_checks: list[CheckResult] = []
    frame, payload = _load_artifacts(global_checks)
    if frame is None:
        _print_checks(global_checks)
        overall = False
        report_path = _write_report({}, global_checks, pd.DataFrame(), overall)
        print(f"\nVerification report: {report_path}")
        return 1

    _check_schema_and_coverage(frame, datasets, percentages, global_checks)
    if payload is not None:
        _check_json_payload(frame, payload, global_checks)
    _check_figures_and_report(global_checks)

    checks_by_dataset: dict[str, list[CheckResult]] = {}
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        checks: list[CheckResult] = []
        dataset_frame = frame[frame["dataset"] == dataset].copy()
        _check_metrics(dataset_frame, checks)
        try:
            preprocessed = preprocess_dataset(dataset, config)
            suite = run_dimensionality_reduction(preprocessed, config)
            if suite.pca is None or not suite.lda:
                raise RuntimeError("Phase 4 suite is missing PCA or LDA results.")
            _check_dimensions(preprocessed, suite, dataset_frame, checks)
            _check_fit_provenance(preprocessed, suite, dataset_frame, checks)
            _check_predictions(preprocessed, dataset_frame, checks)
            _check(
                checks,
                "Test rows excluded from classifier fitting",
                all(
                    int(row.transformed_feature_count) > 0
                    for row in dataset_frame.itertuples(index=False)
                )
                and all(
                    preprocessed.semi_supervised[
                        float(round(row.labelled_percentage / 100.0, 2))
                    ].n_labelled
                    == int(row.n_labelled_train)
                    for row in dataset_frame.itertuples(index=False)
                ),
                "stored n_labelled_train equals the labelled subset size for every "
                "experiment",
            )
            del suite
            del preprocessed
            gc.collect()
        except Exception as exc:
            _check(
                checks,
                "Dataset reconstruction",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        checks_by_dataset[dataset] = checks
        _print_checks(checks)

    all_checks = [check for checks in checks_by_dataset.values() for check in checks]
    all_checks.extend(global_checks)
    overall_passed = all(check.passed for check in all_checks)
    report_path = _write_report(checks_by_dataset, global_checks, frame, overall_passed)

    print("\n" + SEPARATOR)
    print("GLOBAL CHECKS")
    print(SEPARATOR)
    _print_checks(global_checks)
    total = len(all_checks)
    passed = sum(check.passed for check in all_checks)
    print("\n" + SEPARATOR)
    print(f"Checks passed   : {passed} / {total}")
    print(f"Verification report: {report_path}")
    print(f"Overall verdict : {'PASS' if overall_passed else 'FAIL'}")
    print(SEPARATOR)
    return 0 if overall_passed else 1


def _print_checks(checks: Sequence[CheckResult]) -> None:
    print("\nVerification checks")
    for check in checks:
        label = "PASS" if check.passed else "FAIL"
        print(f"  [{label}] {check.name}: {check.detail}")


if __name__ == "__main__":
    raise SystemExit(main())
