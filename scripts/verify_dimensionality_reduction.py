"""Verify Phase 4 dimensionality reduction: baseline, PCA, and LDA.

Run from the project root:

    python scripts/verify_dimensionality_reduction.py

The script reuses the Phase 3 preprocessing (datasets are never downloaded or
modified), then for each dataset:

* keeps the baseline feature space unchanged;
* fits PCA once on the complete scaled training pool (labelled + unlabelled)
  and never on test rows;
* fits LDA separately for 10% / 25% / 50% labelled proportions using only the
  labelled rows and their labels;
* checks component limits, finiteness, and train/test width agreement;
* reruns the full reduction with the same seed and compares content hashes.

Artifacts are written under ``results/dimensionality_reduction/``:
metadata (JSON/CSV), five figures, and a text report. No classifier is trained
and no classification metrics are produced.

Exit code 0 means every check passed; 1 means at least one check failed.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.dimensionality_reduction import (
    DimensionalityReductionSuite,
    ReductionResult,
    run_dimensionality_reduction,
)
from src.experiment_runner import load_config
from src.preprocessing import (
    CheckResult,
    PreprocessedDataset,
    preprocess_dataset,
    validate_preprocessing,
)
from src.visualization import build_reduction_figure_report

SEPARATOR = "=" * 78
DATASET_ORDER: tuple[str, ...] = ("breast_cancer", "mnist", "har")
METHOD_ORDER: tuple[str, ...] = ("none", "pca", "lda")
METHOD_LABELS: dict[str, str] = {"none": "baseline", "pca": "PCA", "lda": "LDA"}
OUTPUT_ROOT = PROJECT_ROOT / "results" / "dimensionality_reduction"
METADATA_DIR = OUTPUT_ROOT / "metadata"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "reports"


def _check(
    checks: list[CheckResult],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append(CheckResult(name=name, passed=bool(passed), detail=detail))


def _enabled_methods(config: dict[str, Any]) -> list[str]:
    settings = config.get("dimensionality_reduction", {})
    methods: list[str] = []
    for configured in settings.get("methods", METHOD_ORDER):
        key = str(configured).strip().lower()
        if key == "baseline":
            key = "none"
        section = "baseline" if key == "none" else key
        if bool((settings.get(section, {}) or {}).get("enabled", True)):
            methods.append(key)
    return methods


def _format_seconds(value: float) -> str:
    return f"{value:.4f}s"


def _array_equal(first: np.ndarray, second: np.ndarray) -> bool:
    return (
        np.asarray(first).shape == np.asarray(second).shape
        and np.array_equal(first, second)
    )


def _hash_array(digest: Any, array: Any) -> None:
    contiguous = np.ascontiguousarray(np.asarray(array))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())


_TRANSFORMER_ATTRIBUTES: tuple[str, ...] = (
    "components_",
    "explained_variance_",
    "explained_variance_ratio_",
    "singular_values_",
    "mean_",
    "scalings_",
    "means_",
    "priors_",
    "classes_",
    "xbar_",
    "n_components_",
    "n_samples_",
    "n_features_in_",
)


def _result_fingerprint(result: ReductionResult) -> str:
    """Hash one projection and its matrices, excluding measured timings."""
    digest = hashlib.sha256()
    digest.update(
        f"{result.dataset}|{result.method}|{result.n_components}|"
        f"{result.n_samples_fit}|{result.fitted_on}".encode("utf-8")
    )
    _hash_array(digest, result.X_train)
    _hash_array(digest, result.X_test)
    if result.explained_variance_ratio is not None:
        _hash_array(digest, result.explained_variance_ratio)
    if result.cumulative_explained_variance is not None:
        _hash_array(digest, result.cumulative_explained_variance)
    if result.transformer is not None:
        for attribute in _TRANSFORMER_ATTRIBUTES:
            value = getattr(result.transformer, attribute, None)
            if value is not None:
                digest.update(attribute.encode("utf-8"))
                _hash_array(digest, value)
    return digest.hexdigest()


def _suite_fingerprint(suite: DimensionalityReductionSuite) -> str:
    cache: dict[int, str] = {}
    digest = hashlib.sha256()
    for configuration in sorted(
        suite.configurations,
        key=lambda item: (item.labelled_proportion, METHOD_ORDER.index(item.method)),
    ):
        key = id(configuration.result)
        if key not in cache:
            cache[key] = _result_fingerprint(configuration.result)
        digest.update(
            f"{configuration.method}|{configuration.labelled_proportion:.2f}|"
            f"{cache[key]}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _validate_baseline(
    checks: list[CheckResult],
    preprocessed: PreprocessedDataset,
    suite: DimensionalityReductionSuite,
) -> None:
    baseline = suite.baseline
    if baseline is None:
        _check(checks, "baseline_present", False, "Baseline result is missing.")
        return

    dimensions_ok = (
        baseline.n_components == preprocessed.n_features
        and baseline.feature_names == list(preprocessed.feature_names)
    )
    _check(
        checks,
        "baseline_preserves_dimensions",
        dimensions_ok,
        f"{baseline.n_components:,} features retained "
        f"(Phase 3 features {preprocessed.n_features:,}); "
        f"feature names preserved={baseline.feature_names == list(preprocessed.feature_names)}.",
    )

    same_train = _array_equal(baseline.X_train, preprocessed.X_train)
    same_test = _array_equal(baseline.X_test, preprocessed.X_test)
    _check(
        checks,
        "baseline_matches_phase3_scaled_matrices",
        same_train and same_test,
        f"train identical={same_train}, test identical={same_test}; "
        f"shapes {baseline.X_train.shape} / {baseline.X_test.shape}.",
    )

    _check(
        checks,
        "baseline_uses_no_fitted_reducer",
        baseline.transformer is None and baseline.fit_time_seconds == 0.0,
        "No PCA/LDA object was fitted; baseline reports fit time 0.0s and a "
        f"reference transform time of {_format_seconds(baseline.transform_time_seconds)}.",
    )


def _validate_pca(
    checks: list[CheckResult],
    preprocessed: PreprocessedDataset,
    suite: DimensionalityReductionSuite,
    target_variance: float,
) -> None:
    result = suite.pca
    if result is None:
        _check(checks, "pca_present", False, "PCA result is missing.")
        return

    cumulative = result.cumulative_explained_variance
    retained = result.explained_variance_retained
    if cumulative is None or retained is None:
        _check(
            checks,
            "pca_target_variance_met",
            False,
            "PCA did not record cumulative/retained explained variance.",
        )
        return

    attained = float(cumulative[-1])
    previous = float(cumulative[-2]) if cumulative.size > 1 else 0.0
    meets_target = attained + 1e-12 >= target_variance
    minimal = result.n_components == 1 or previous < target_variance
    _check(
        checks,
        "pca_target_variance_met",
        meets_target and minimal,
        f"target={target_variance:.2f}; {result.n_components:,} components retain "
        f"{attained:.6f} ({attained:.4%}); cumulative with one fewer component "
        f"= {previous:.6f} (< target={previous < target_variance}); "
        f"minimal selection={minimal}.",
    )

    transformer = result.transformer
    n_seen = int(getattr(transformer, "n_samples_", -1))
    # Sklearn computes PCA.mean_ in the input dtype (MNIST is float32), so
    # compare against the same-dtype mean with a small numeric tolerance.
    training_mean = np.mean(preprocessed.X_train, axis=0)
    mean_matches = np.allclose(
        np.asarray(transformer.mean_), training_mean, rtol=1e-5, atol=1e-5
    )
    _check(
        checks,
        "pca_fit_scope_complete_training_pool",
        n_seen == preprocessed.n_train and mean_matches,
        f"PCA n_samples_={n_seen:,} equals training pool "
        f"{preprocessed.n_train:,} (test rows {preprocessed.n_test:,} excluded); "
        f"fitted mean matches the training-pool mean={mean_matches}; "
        f"fitted_on={result.fitted_on!r}.",
    )

    recomputed_test = transformer.transform(preprocessed.X_test)
    reuse_ok = np.allclose(recomputed_test, result.X_test, rtol=0.0, atol=1e-10)
    recomputed_train = transformer.transform(preprocessed.X_train)
    reuse_train_ok = np.allclose(
        recomputed_train, result.X_train, rtol=0.0, atol=1e-10
    )
    _check(
        checks,
        "pca_same_fitted_object_transforms_train_and_test",
        reuse_train_ok and reuse_ok,
        f"stored train/test matrices reproduce the fitted object's transform "
        f"(train={reuse_train_ok}, test={reuse_ok}).",
    )

    solver = result.metadata.get("resolved_solver")
    configured = result.metadata.get("configured_solver")
    target_value = result.metadata.get("configured_n_components")
    solver_ok = (
        np.isclose(float(target_value), target_variance)
        and solver in ("full", "auto")
        and result.n_components == int(transformer.n_components_)
    )
    _check(
        checks,
        "pca_solver_supports_fractional_target",
        solver_ok,
        f"configured solver={configured!r} -> resolved solver={solver!r}; "
        f"target unchanged={np.isclose(float(target_value), target_variance)}; "
        f"sklearn n_components_={int(transformer.n_components_)} equals reduced "
        f"width {result.n_components}.",
    )

    names_ok = result.feature_names == [
        f"PC{index + 1}" for index in range(result.n_components)
    ]
    _check(
        checks,
        "pca_component_names",
        names_ok and len(result.explained_variance_ratio) == result.n_components,
        f"{result.n_components:,} PCA components named PC1..PC{result.n_components:,}; "
        "one explained-variance ratio per component.",
    )


def _validate_lda(
    checks: list[CheckResult],
    preprocessed: PreprocessedDataset,
    suite: DimensionalityReductionSuite,
    automatic_components: bool,
) -> None:
    if not suite.lda:
        _check(checks, "lda_present", False, "LDA results are missing.")
        return

    for proportion in sorted(preprocessed.semi_supervised):
        split = preprocessed.semi_supervised[proportion]
        result = suite.lda.get(proportion)
        percent = f"{proportion:.0%}"
        if result is None:
            _check(
                checks,
                f"lda_fit_only_labelled_{percent}",
                False,
                f"No LDA result for labelled proportion {percent}.",
            )
            continue

        transformer = result.transformer
        X_labelled = preprocessed.X_train[split.labelled_indices]
        y_labelled = split.y_labelled
        classes = np.unique(y_labelled)
        counts = np.array([(y_labelled == label).sum() for label in classes], dtype=float)
        expected_priors = counts / counts.sum()
        expected_means = np.vstack(
            [X_labelled[y_labelled == label].mean(axis=0) for label in classes]
        )

        priors_ok = np.allclose(
            np.asarray(transformer.priors_), expected_priors, rtol=1e-6, atol=1e-7
        )
        means_ok = np.allclose(
            np.asarray(transformer.means_), expected_means, rtol=1e-5, atol=1e-5
        )
        classes_ok = np.array_equal(np.asarray(transformer.classes_), classes)
        labelled_mean = np.mean(X_labelled, axis=0)
        xbar_ok = np.allclose(
            np.asarray(transformer.xbar_), labelled_mean, rtol=1e-5, atol=1e-5
        )
        scope_ok = (
            result.n_samples_fit == split.n_labelled
            and result.fitted_on == f"labelled_training_subset_{proportion:.2f}"
            and xbar_ok
        )
        _check(
            checks,
            f"lda_fit_only_labelled_{percent}",
            scope_ok and priors_ok and means_ok and classes_ok,
            f"fit rows={result.n_samples_fit:,} equals labelled rows "
            f"{split.n_labelled:,} (unlabelled rows {split.n_unlabelled:,} and "
            f"test rows {preprocessed.n_test:,} excluded); fit-data mean xbar_ "
            f"matches labelled rows={xbar_ok}; priors match labelled "
            f"frequencies={priors_ok}; class means match labelled rows={means_ok}; "
            f"classes={classes.tolist()}; unlabelled/test labels were never passed.",
        )

        max_allowed = min(preprocessed.n_features, classes.size - 1)
        within_cap = 1 <= result.n_components <= max_allowed
        limit_ok = within_cap and (
            result.n_components == max_allowed if automatic_components else True
        )
        selection = result.metadata.get("component_selection", "unknown")
        _check(
            checks,
            f"lda_component_limit_{percent}",
            limit_ok,
            f"{result.n_components} components; cap min(n_features="
            f"{preprocessed.n_features:,}, n_classes-1={classes.size - 1})"
            f"={max_allowed}; within cap={within_cap}; selection={selection!r}; "
            f"automatic selection uses the full cap={automatic_components}.",
        )

        X_unlabelled = preprocessed.X_train[split.unlabelled_indices]
        labelled_reuse = np.allclose(
            transformer.transform(X_labelled),
            result.X_train[split.labelled_indices],
            rtol=0.0,
            atol=1e-10,
        )
        unlabelled_reuse = np.allclose(
            transformer.transform(X_unlabelled),
            result.X_train[split.unlabelled_indices],
            rtol=0.0,
            atol=1e-10,
        )
        test_reuse = np.allclose(
            transformer.transform(preprocessed.X_test),
            result.X_test,
            rtol=0.0,
            atol=1e-10,
        )
        _check(
            checks,
            f"lda_same_fitted_object_transforms_all_sets_{percent}",
            _labelled_shape_ok(result, preprocessed)
            and labelled_reuse
            and unlabelled_reuse
            and test_reuse,
            f"full training matrix {result.X_train.shape} covers all "
            f"{preprocessed.n_train:,} rows; same fitted LDA reproduces labelled="
            f"{labelled_reuse}, unlabelled={unlabelled_reuse}, test={test_reuse}.",
        )

    transformer_ids = {id(result.transformer) for result in suite.lda.values()}
    _check(
        checks,
        "lda_refitted_per_proportion",
        len(transformer_ids) == len(suite.lda),
        f"{len(suite.lda)} proportions produced {len(transformer_ids)} distinct "
        "fitted LDA objects (one refit per labelled proportion).",
    )


def _labelled_shape_ok(
    result: ReductionResult,
    preprocessed: PreprocessedDataset,
) -> bool:
    return result.X_train.shape[0] == preprocessed.n_train and np.isfinite(
        result.X_train
    ).all()


def _validate_common(
    checks: list[CheckResult],
    preprocessed: PreprocessedDataset,
    suite: DimensionalityReductionSuite,
    expected_methods: Sequence[str],
) -> None:
    results: list[tuple[str, ReductionResult]] = []
    if suite.baseline is not None:
        results.append(("baseline", suite.baseline))
    if suite.pca is not None:
        results.append(("pca", suite.pca))
    for proportion, result in sorted(suite.lda.items()):
        results.append((f"lda@{proportion:.0%}", result))

    width_ok = all(
        result.X_train.shape[1] == result.X_test.shape[1] == result.n_components
        and len(result.feature_names) == result.n_components
        for _, result in results
    )
    _check(
        checks,
        "reduced_train_test_widths_match",
        width_ok,
        "; ".join(
            f"{label}: train {result.X_train.shape[1]} / test "
            f"{result.X_test.shape[1]} / n_components {result.n_components}"
            for label, result in results
        ),
    )

    non_finite = [
        label
        for label, result in results
        if not np.isfinite(result.X_train).all() or not np.isfinite(result.X_test).all()
    ]
    _check(
        checks,
        "all_transformed_matrices_finite",
        not non_finite,
        "All transformed train/test matrices contain only finite values."
        if not non_finite
        else "Non-finite values found in: " + ", ".join(non_finite),
    )

    combination = {
        (configuration.method, configuration.labelled_proportion)
        for configuration in suite.configurations
    }
    expected = {
        (method, float(proportion))
        for method in expected_methods
        for proportion in preprocessed.semi_supervised
    }
    _check(
        checks,
        "dataset_configuration_grid_complete",
        len(suite.configurations) == len(expected) and combination == expected,
        f"{len(suite.configurations)} configurations for "
        f"{len(expected_methods)} methods x "
        f"{len(preprocessed.semi_supervised)} proportions; "
        f"grid matches expected={combination == expected}.",
    )

    baseline_ids = {
        id(configuration.result)
        for configuration in suite.configurations
        if configuration.method == "none"
    }
    pca_ids = {
        id(configuration.result)
        for configuration in suite.configurations
        if configuration.method == "pca"
    }
    lda_ids = {
        id(configuration.result)
        for configuration in suite.configurations
        if configuration.method == "lda"
    }
    reuse_ok = (
        (not baseline_ids or len(baseline_ids) == 1)
        and (not pca_ids or len(pca_ids) == 1)
        and len(lda_ids) == len(suite.lda)
    )
    _check(
        checks,
        "projection_reuse_policy",
        reuse_ok,
        f"baseline fitted once={len(baseline_ids)} object(s); PCA fitted once="
        f"{len(pca_ids)} object(s); LDA objects={len(lda_ids)} for "
        f"{len(suite.lda)} proportions.",
    )

    timing_problems: list[str] = []
    for label, result in results:
        if not np.isfinite(result.fit_time_seconds) or result.fit_time_seconds < 0:
            timing_problems.append(f"{label} fit time")
        if (
            not np.isfinite(result.transform_time_seconds)
            or result.transform_time_seconds < 0
        ):
            timing_problems.append(f"{label} transform time")
        if result.method in ("pca", "lda") and result.fit_time_seconds <= 0.0:
            timing_problems.append(f"{label} fit time not positive")
    _check(
        checks,
        "fit_and_transform_times_reported",
        not timing_problems,
        "; ".join(
            f"{label} fit={_format_seconds(result.fit_time_seconds)}, "
            f"transform={_format_seconds(result.transform_time_seconds)}"
            for label, result in results
        )
        if not timing_problems
        else "Invalid timing values: " + ", ".join(timing_problems),
    )


def _validate_suite(
    preprocessed: PreprocessedDataset,
    suite: DimensionalityReductionSuite,
    config: dict[str, Any],
    expected_methods: Sequence[str],
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    target = float(
        config["dimensionality_reduction"]["pca"].get(
            "explained_variance",
            config["dimensionality_reduction"]["pca"].get("n_components", 0.95),
        )
    )

    preprocessing_report = validate_preprocessing(preprocessed, config)
    _check(
        checks,
        "phase3_preprocessing_preserved",
        preprocessing_report.passed,
        f"{sum(check.passed for check in preprocessing_report.checks)} of "
        f"{len(preprocessing_report.checks)} Phase 3 checks still pass.",
    )

    if "none" in expected_methods:
        _validate_baseline(checks, preprocessed, suite)
    if "pca" in expected_methods:
        _validate_pca(checks, preprocessed, suite, target)
    if "lda" in expected_methods:
        automatic = bool(
            config["dimensionality_reduction"]["lda"].get(
                "automatic_components", True
            )
        )
        _validate_lda(checks, preprocessed, suite, automatic)

    _validate_common(checks, preprocessed, suite, expected_methods)
    return checks


def _statistics_rows(suite: DimensionalityReductionSuite) -> list[dict[str, Any]]:
    return [
        configuration.statistics() for configuration in suite.configurations
    ]


def _lda_detail_rows(
    suite: DimensionalityReductionSuite,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proportion in sorted(suite.lda):
        result = suite.lda[proportion]
        metadata = result.metadata
        rows.append(
            {
                "dataset": result.dataset,
                "labelled_proportion": float(proportion),
                "n_labelled_fit_rows": int(result.n_samples_fit),
                "n_classes": metadata.get("n_classes"),
                "class_counts": json.dumps(
                    metadata.get("class_counts", {}), sort_keys=True
                ),
                "components": int(result.n_components),
                "max_components": metadata.get("max_components"),
                "component_selection": metadata.get("component_selection"),
                "solver": metadata.get("solver"),
                "explained_variance_retained": result.explained_variance_retained,
                "fit_time_seconds": float(result.fit_time_seconds),
                "transform_time_seconds": float(result.transform_time_seconds),
            }
        )
    return rows


def _pca_curve(
    suite: DimensionalityReductionSuite,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    result = suite.pca
    if result is None or result.cumulative_explained_variance is None:
        return None
    ratios = np.asarray(result.explained_variance_ratio, dtype=np.float64).copy()
    cumulative = np.asarray(
        result.cumulative_explained_variance, dtype=np.float64
    ).copy()
    return ratios, cumulative, int(result.n_components)


def _print_dataset_header(
    preprocessed: PreprocessedDataset,
) -> None:
    print(f"\n{SEPARATOR}")
    print(f"DATASET: {preprocessed.name.upper()}")
    print(SEPARATOR)
    print(
        f"Original features : {preprocessed.n_features:,} | "
        f"train rows: {preprocessed.n_train:,} | test rows: {preprocessed.n_test:,}"
    )
    print(
        "Labelled settings : "
        + ", ".join(
            f"{proportion:.0%}={preprocessed.semi_supervised[proportion].n_labelled:,}"
            for proportion in sorted(preprocessed.semi_supervised)
        )
    )


def _print_reduction_summary(
    preprocessed: PreprocessedDataset,
    suite: DimensionalityReductionSuite,
) -> None:
    print("\nMeasured reduction")
    if suite.baseline is not None:
        baseline = suite.baseline
        print(
            f"  Baseline       : {baseline.n_components:,} features "
            f"(0.0% reduction), fit {_format_seconds(baseline.fit_time_seconds)}, "
            f"transform {_format_seconds(baseline.transform_time_seconds)}"
        )
    if suite.pca is not None:
        pca = suite.pca
        retained = pca.explained_variance_retained
        print(
            f"  PCA            : {pca.n_components:,} / {pca.n_original_features:,} "
            f"features ({pca.reduction_percent:.2f}% reduction), "
            f"variance retained {retained:.4%}, "
            f"fit {_format_seconds(pca.fit_time_seconds)}, "
            f"transform {_format_seconds(pca.transform_time_seconds)}"
        )
    for proportion in sorted(suite.lda):
        lda = suite.lda[proportion]
        print(
            f"  LDA ({proportion:>4.0%})    : {lda.n_components} / "
            f"{lda.n_original_features:,} features "
            f"({lda.reduction_percent:.2f}% reduction), "
            f"fit rows {lda.n_samples_fit:,}, "
            f"fit {_format_seconds(lda.fit_time_seconds)}, "
            f"transform {_format_seconds(lda.transform_time_seconds)}"
        )


def _print_checks(checks: Sequence[CheckResult]) -> None:
    print("\nVerification checks")
    for check in checks:
        label = "PASS" if check.passed else "FAIL"
        print(f"  [{label}] {check.name}: {check.detail}")


def _json_safe(value: Any) -> Any:
    """Convert numpy/missing values into strict-JSON-compatible objects."""
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
    return value


def _write_metadata(
    statistics: pd.DataFrame,
    pca_variance: pd.DataFrame,
    lda_details: pd.DataFrame,
    config: dict[str, Any],
    dataset_summaries: dict[str, Any],
) -> dict[str, Path]:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    statistics_path = METADATA_DIR / "reduction_configurations.csv"
    lda_path = METADATA_DIR / "lda_details.csv"
    pca_path = METADATA_DIR / "pca_explained_variance.csv"
    json_path = METADATA_DIR / "reduction_configurations.json"

    statistics.to_csv(statistics_path, index=False)
    lda_details.to_csv(lda_path, index=False)
    pca_variance.to_csv(pca_path, index=False)

    payload = {
        "phase": 4,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": int(config["project"]["random_seed"]),
        "n_configurations": int(len(statistics)),
        "pca_target_variance": float(
            config["dimensionality_reduction"]["pca"]["explained_variance"]
        ),
        "leakage_policy": {
            "baseline": "Phase 3 scaled features used unchanged.",
            "pca": "Fitted once on the complete scaled training pool "
            "(labelled + unlabelled); test rows never used.",
            "lda": "Fitted separately per labelled proportion on labelled rows "
            "and labels only; unlabelled true labels and test rows/labels never used.",
        },
        "datasets": dataset_summaries,
        "configurations": statistics.to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")

    return {
        "statistics_csv": statistics_path,
        "statistics_json": json_path,
        "pca_variance_csv": pca_path,
        "lda_details_csv": lda_path,
    }


def _write_report(
    checks_by_dataset: dict[str, list[CheckResult]],
    global_checks: Sequence[CheckResult],
    statistics: pd.DataFrame,
    paths: dict[str, Path],
    figure_paths: Sequence[Path],
    overall_passed: bool,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "phase4_dimensionality_reduction_report.txt"
    lines = [
        "Phase 4 dimensionality-reduction verification report",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "No classifiers were trained and no classification metrics are reported.",
        "",
        "Configuration table",
        "-" * 78,
    ]
    columns = [
        "dataset",
        "method",
        "labelled_proportion",
        "original_features",
        "reduced_features",
        "dimensionality_reduction_percent",
        "explained_variance_retained",
        "fit_time_seconds",
        "transform_time_seconds",
        "fitted_on",
        "n_samples_fit",
    ]
    lines.append(
        "dataset\tmethod\tproportion\toriginal\treduced\treduction_%\t"
        "variance_retained\tfit_s\ttransform_s\tfitted_on\tn_fit_rows"
    )
    for row in statistics[columns].itertuples(index=False):
        variance = row.explained_variance_retained
        variance_text = "n/a" if variance is None else f"{float(variance):.6f}"
        lines.append(
            f"{row.dataset}\t{row.method}\t{row.labelled_proportion:.2f}\t"
            f"{row.original_features}\t{row.reduced_features}\t"
            f"{row.dimensionality_reduction_percent:.4f}\t{variance_text}\t"
            f"{row.fit_time_seconds:.4f}\t{row.transform_time_seconds:.4f}\t"
            f"{row.fitted_on}\t{row.n_samples_fit}"
        )

    lines.extend(["", "Checks", "-" * 78])
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
    lines.append("")

    lines.extend(
        [
            "Artifacts",
            "-" * 78,
            f"Metadata JSON     : {paths['statistics_json']}",
            f"Metadata CSV      : {paths['statistics_csv']}",
            f"PCA variance CSV  : {paths['pca_variance_csv']}",
            f"LDA details CSV   : {paths['lda_details_csv']}",
        ]
    )
    for path in figure_paths:
        lines.append(f"Figure            : {path}")

    lines.extend(
        [
            "",
            f"Checks passed     : {passed} / {total}",
            f"Overall verdict   : {'PASS' if overall_passed else 'FAIL'}",
            "",
            "PCA and LDA in plain terms",
            "-" * 78,
            "PCA (Principal Component Analysis) is an unsupervised compression step.",
            "It looks at how the features vary together and builds new axes (principal",
            "components) that point along the directions of greatest variance. The axes",
            "are ordered, so keeping the first few components preserves most of the",
            "information (measured by explained variance) while discarding redundancy.",
            "It never looks at class labels, which is why it may be fitted on the whole",
            "training pool, including unlabelled rows, without leaking test information.",
            "",
            "LDA (Linear Discriminant Analysis) is a supervised projection. It uses the",
            "known class labels to find axes that maximise the separation between class",
            "means relative to the spread within each class. Because it needs labels, it",
            "is fitted only on the labelled training subset for each proportion. The",
            "number of useful discriminant axes is at most n_classes - 1, so Breast",
            "Cancer gives 1, MNIST gives 9, and HAR gives 5.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    seed = int(config["project"]["random_seed"])
    target = float(config["dimensionality_reduction"]["pca"]["explained_variance"])
    expected_methods = _enabled_methods(config)
    datasets = [
        name
        for name in DATASET_ORDER
        if bool(
            (config["data"]["datasets"].get(name, {}) or {}).get("enabled", True)
        )
    ]

    print(SEPARATOR)
    print("Phase 4 dimensionality-reduction verification")
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Random seed      : {seed}")
    print(f"Methods          : {expected_methods}")
    print(f"PCA target       : {target:.2%} explained variance")
    print(f"LDA selection    : automatic, capped at n_classes - 1")
    print(f"Expected grid    : {len(datasets)} datasets x "
          f"{len(expected_methods)} methods x 3 proportions = "
          f"{len(datasets) * len(expected_methods) * 3}")
    print(SEPARATOR)

    checks_by_dataset: dict[str, list[CheckResult]] = {}
    statistics_rows: list[dict[str, Any]] = []
    lda_rows: list[dict[str, Any]] = []
    pca_curves: dict[str, Any] = {}
    pca_variance_rows: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    reproducibility: dict[str, bool] = {}

    for dataset_name in datasets:
        preprocessed: PreprocessedDataset | None = None
        try:
            preprocessed = preprocess_dataset(dataset_name, config)
            _print_dataset_header(preprocessed)

            suite = run_dimensionality_reduction(preprocessed, config)
            checks = _validate_suite(preprocessed, suite, config, expected_methods)
            _print_reduction_summary(preprocessed, suite)

            digest_first = _suite_fingerprint(suite)
            first_components = {
                (configuration.method, configuration.labelled_proportion):
                configuration.result.n_components
                for configuration in suite.configurations
            }
            first_variance = {
                (configuration.method, configuration.labelled_proportion):
                configuration.result.explained_variance_retained
                for configuration in suite.configurations
            }
            statistics_rows.extend(_statistics_rows(suite))
            lda_rows.extend(_lda_detail_rows(suite))

            curve = _pca_curve(suite)
            if curve is not None:
                ratios, cumulative, n_components = curve
                pca_curves[preprocessed.name] = SimpleNamespace(
                    n_components=n_components,
                    cumulative_explained_variance=cumulative,
                )
                pca_variance_rows.extend(
                    {
                        "dataset": preprocessed.name,
                        "component": index + 1,
                        "explained_variance_ratio": float(ratio),
                        "cumulative_explained_variance": float(cumulative_value),
                    }
                    for index, (ratio, cumulative_value) in enumerate(
                        zip(ratios, cumulative)
                    )
                )

            dataset_summaries[preprocessed.name] = {
                "n_original_features": int(preprocessed.n_features),
                "n_train": int(preprocessed.n_train),
                "n_test": int(preprocessed.n_test),
                "baseline_features": (
                    int(suite.baseline.n_components)
                    if suite.baseline is not None
                    else None
                ),
                "pca_features": (
                    int(suite.pca.n_components) if suite.pca is not None else None
                ),
                "pca_variance_retained": (
                    suite.pca.explained_variance_retained
                    if suite.pca is not None
                    else None
                ),
                "lda": {
                    f"{proportion:.2f}": {
                        "components": int(result.n_components),
                        "n_samples_fit": int(result.n_samples_fit),
                        "fit_time_seconds": float(result.fit_time_seconds),
                        "transform_time_seconds": float(
                            result.transform_time_seconds
                        ),
                    }
                    for proportion, result in sorted(suite.lda.items())
                },
                "fingerprint_sha256": digest_first,
            }

            del suite
            gc.collect()

            repeated = run_dimensionality_reduction(preprocessed, config)
            digest_second = _suite_fingerprint(repeated)
            second_components = {
                (configuration.method, configuration.labelled_proportion):
                configuration.result.n_components
                for configuration in repeated.configurations
            }
            second_variance = {
                (configuration.method, configuration.labelled_proportion):
                configuration.result.explained_variance_retained
                for configuration in repeated.configurations
            }
            reproducible = (
                digest_first == digest_second
                and first_components == second_components
                and first_variance == second_variance
            )
            reproducibility[preprocessed.name] = reproducible
            _check(
                checks,
                "same_seed_reproducibility",
                reproducible,
                f"SHA-256 content fingerprint {digest_first[:16]}... repeated "
                f"{digest_second[:16]}...; components identical="
                f"{first_components == second_components}; explained variance "
                f"identical={first_variance == second_variance}.",
            )

            del repeated
            gc.collect()
        except Exception as exc:  # Report the exact failure and continue.
            print(f"\n{SEPARATOR}")
            print(f"DATASET: {dataset_name.upper()}")
            print(SEPARATOR)
            checks = [
                CheckResult(
                    name="dimensionality_reduction_execution",
                    passed=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            ]
            reproducibility[dataset_name] = False
        checks_by_dataset[dataset_name] = checks
        _print_checks(checks)

        if preprocessed is not None:
            del preprocessed
        gc.collect()

    global_checks: list[CheckResult] = []
    statistics = pd.DataFrame(statistics_rows)
    expected_count = len(datasets) * len(expected_methods) * 3
    if statistics.empty:
        _check(
            global_checks,
            "all_configurations_present",
            False,
            "No reduction statistics were produced.",
        )
    else:
        keys = {
            (
                str(row.dataset),
                str(row.method),
                float(row.labelled_proportion),
            )
            for row in statistics.itertuples(index=False)
        }
        expected_keys = {
            (dataset, method, float(proportion))
            for dataset in datasets
            for method in expected_methods
            for proportion in config["data"]["labelled_proportions"]
        }
        _check(
            global_checks,
            "all_configurations_present",
            len(statistics) == expected_count and keys == expected_keys,
            f"{len(statistics)} rows (expected {expected_count}); dataset/method/"
            f"proportion combinations unique and complete={keys == expected_keys}.",
        )

    _check(
        global_checks,
        "all_datasets_reproducible",
        len(reproducibility) == len(datasets) and all(reproducibility.values()),
        "; ".join(
            f"{name}={'PASS' if reproducibility.get(name) else 'FAIL'}"
            for name in datasets
        ),
    )

    pca_variance = pd.DataFrame(
        pca_variance_rows,
        columns=[
            "dataset",
            "component",
            "explained_variance_ratio",
            "cumulative_explained_variance",
        ],
    )
    lda_details = pd.DataFrame(lda_rows)
    paths = _write_metadata(
        statistics,
        pca_variance,
        lda_details,
        config,
        dataset_summaries,
    )

    if statistics.empty:
        figure_paths = []
        _check(
            global_checks,
            "figures_generated",
            False,
            "No reduction statistics were produced, so no figures were written.",
        )
    else:
        figure_paths = build_reduction_figure_report(
            statistics,
            pca_curves,
            figures_dir=FIGURE_DIR,
            target_variance=target,
        )
        _check(
            global_checks,
            "figures_generated",
            len(figure_paths) == 5 and all(path.exists() for path in figure_paths),
            f"{len(figure_paths)} figures written to {FIGURE_DIR}.",
        )

    all_checks = [
        check for checks in checks_by_dataset.values() for check in checks
    ] + global_checks
    overall_passed = bool(all_checks) and all(check.passed for check in all_checks)

    report_path = _write_report(
        checks_by_dataset,
        global_checks,
        statistics,
        paths,
        figure_paths,
        overall_passed,
    )

    print(f"\n{SEPARATOR}\nGLOBAL CHECKS\n{SEPARATOR}")
    _print_checks(global_checks)

    print(f"\n{SEPARATOR}\nSUMMARY\n{SEPARATOR}")
    for dataset_name in datasets:
        checks = checks_by_dataset.get(dataset_name, [])
        passed = sum(check.passed for check in checks)
        failures = ", ".join(check.name for check in checks if not check.passed) or "none"
        print(
            f"{dataset_name:<16}: {'PASS' if checks and passed == len(checks) else 'FAIL'} "
            f"({passed}/{len(checks)} checks; failures: {failures})"
        )
    print(
        f"Configurations  : {len(statistics)} / {expected_count} represented"
    )
    print(f"Metadata        : {paths['statistics_json'].parent}")
    print(f"Figures         : {FIGURE_DIR} ({len(figure_paths)} files)")
    print(f"Report          : {report_path}")
    print(f"\nOverall         : {'ALL CHECKS PASSED' if overall_passed else 'ACTION REQUIRED'}")
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
