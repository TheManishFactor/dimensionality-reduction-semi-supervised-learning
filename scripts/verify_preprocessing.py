"""Verify Phase 3 preprocessing, scaling, and semi-supervised splits.

Run from the project root:

    python scripts/verify_preprocessing.py

The script loads the three staged datasets (it never downloads anything),
builds the fixed train/test partitions, fits ``StandardScaler`` on training
data only, creates the 10% / 25% / 50% labelled settings, and prints a
PASS/FAIL report for every leakage and integrity check. Split metadata
(seed, sizes, index hashes, subject IDs) is written to
``data/processed/preprocessing_splits.json``.

Exit code 0 means every dataset passed; 1 means at least one check failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.experiment_runner import load_config
from src.preprocessing import (
    PreprocessedDataset,
    PreprocessingReport,
    build_split_metadata,
    format_preprocessing_report,
    preprocess_dataset,
    save_split_metadata,
    validate_preprocessing,
)

SEPARATOR = "=" * 78


def _format_distribution(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    return ", ".join(
        f"{label}: {count:,} ({100.0 * count / total:.2f}%)"
        for label, count in counts.items()
    )


def _print_scaling(result: PreprocessedDataset) -> None:
    scaler = result.scaler
    n_seen = scaler.n_samples_seen_
    n_seen_int = int(np.max(n_seen)) if np.ndim(n_seen) else int(n_seen)
    nonconstant = np.asarray(scaler.var_) > 0

    train_mean = np.mean(result.X_train, axis=0, dtype=np.float64)
    train_std = np.std(result.X_train, axis=0, dtype=np.float64)

    print("\nScaling")
    print(
        f"  Method / scope : {result.metadata['scaling']['method']} / "
        f"{result.metadata['scaling']['fit_scope']}"
    )
    print(
        f"  Rows seen      : {n_seen_int:,} "
        f"(train rows = {result.n_train:,}, test rows = {result.n_test:,})"
    )
    print(
        f"  Max |mean|     : {np.max(np.abs(train_mean[nonconstant])):.2e}"
        if nonconstant.any()
        else "  Max |mean|     : n/a"
    )
    if nonconstant.any():
        print(
            f"  Std range      : {train_std[nonconstant].min():.6f} to "
            f"{train_std[nonconstant].max():.6f}"
        )
    print(f"  Constant feats : {int((~nonconstant).sum())}")
    print(
        f"  Train range    : {float(result.X_train.min()):.4f} to "
        f"{float(result.X_train.max()):.4f}"
    )
    print(
        f"  Test range     : {float(result.X_test.min()):.4f} to "
        f"{float(result.X_test.max()):.4f}"
    )


def _print_labelled_settings(result: PreprocessedDataset) -> None:
    print("\nLabelled / unlabelled settings")
    for proportion in sorted(result.semi_supervised):
        split = result.semi_supervised[proportion]
        labels, counts = np.unique(split.y_labelled, return_counts=True)
        labelled_counts = {
            str(label): int(count) for label, count in zip(labels, counts)
        }
        print(
            f"  {proportion:>4.0%}: {split.n_labelled:>7,} labelled / "
            f"{split.n_unlabelled:>7,} unlabelled "
            f"(actual {split.actual_proportion:.4f})"
        )
        print(f"          labelled classes: {_format_distribution(labelled_counts)}")
        print(
            "          unlabelled true labels stored separately: "
            f"{result.get_research_unlabelled_labels(proportion).size:,}"
        )


def _print_subjects(result: PreprocessedDataset) -> None:
    print("\nHAR subject separation")
    train_subjects = np.unique(result.train_subjects)
    test_subjects = np.unique(result.test_subjects)
    overlap = np.intersect1d(train_subjects, test_subjects)
    print(
        f"  Train subjects ({len(train_subjects)}): "
        f"{train_subjects.tolist()}"
    )
    print(f"  Test subjects  ({len(test_subjects)}): {test_subjects.tolist()}")
    print(f"  Overlap        : {overlap.size}")


def _print_report(result: PreprocessedDataset, report: PreprocessingReport) -> None:
    details = report.details
    print(f"\n{SEPARATOR}\nDATASET: {result.name.upper()}\n{SEPARATOR}")
    print(f"Strategy         : {details['strategy']}")
    print(f"Original rows    : {details['n_original']:,}")
    print(f"Train / test     : {details['n_train']:,} / {details['n_test']:,}")
    print(
        f"Features         : {details['n_features']:,} "
        f"(feature names preserved: {len(result.feature_names)})"
    )
    print(
        f"Target classes   : {len(result.target_names)} "
        f"({', '.join(result.target_names)})"
    )
    print(f"Train classes    : {_format_distribution(details['train_class_counts'])}")
    print(f"Test classes     : {_format_distribution(details['test_class_counts'])}")
    print(f"Train/test drift : {details['train_test_drift']:.4f}")

    _print_scaling(result)
    _print_labelled_settings(result)
    if result.train_subjects is not None:
        _print_subjects(result)

    print("\nVerification checks")
    for check in report.checks:
        label = "PASS" if check.passed else "FAIL"
        print(f"  [{label}] {check.name}: {check.detail}")
    if report.warnings:
        print("  Warnings: " + "; ".join(report.warnings))
    print(f"\nDataset verdict  : {'PASS' if report.passed else 'FAIL'}")


def main() -> int:
    config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    seed = int(config["project"]["random_seed"])
    proportions = config["data"]["labelled_proportions"]

    print(SEPARATOR)
    print("Phase 3 preprocessing verification")
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Random seed      : {seed}")
    print(f"Test size        : {config['data']['test_size']:.2f} (holdout datasets)")
    print(f"Labelled splits  : {proportions}")
    print(f"Scaling          : {config['data']['scaling']['method']} "
          f"(fit on {config['data']['scaling']['fit_scope']})")
    print(SEPARATOR)

    report_by_dataset: dict[str, PreprocessingReport] = {}
    metadata_by_dataset: dict[str, dict] = {}

    for dataset_name in ("breast_cancer", "mnist", "har"):
        try:
            result = preprocess_dataset(dataset_name, config)
            report = validate_preprocessing(result, config)
            _print_report(result, report)
            report_by_dataset[dataset_name] = report
            metadata_by_dataset[dataset_name] = build_split_metadata(result, config)
        except Exception as exc:  # Report and continue with the other datasets.
            print(f"\n{SEPARATOR}\nDATASET: {dataset_name.upper()}\n{SEPARATOR}")
            print(f"[FAIL] preprocessing raised {type(exc).__name__}: {exc}")

    print(f"\n{SEPARATOR}\nSUMMARY\n{SEPARATOR}")
    for dataset_name in ("breast_cancer", "mnist", "har"):
        report = report_by_dataset.get(dataset_name)
        if report is None:
            print(f"{dataset_name:<16}: FAIL (preprocessing error)")
            continue
        failures = ", ".join(check.name for check in report.failures) or "none"
        print(
            f"{dataset_name:<16}: {'PASS' if report.passed else 'FAIL'} "
            f"({len(report.checks)} checks; failures: {failures})"
        )

    all_passed = (
        len(report_by_dataset) == 3
        and all(report.passed for report in report_by_dataset.values())
    )

    if metadata_by_dataset:
        path = save_split_metadata(metadata_by_dataset, config)
        print(f"\nSplit metadata   : {path}")

    print(f"\nOverall          : {'ALL CHECKS PASSED' if all_passed else 'ACTION REQUIRED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
