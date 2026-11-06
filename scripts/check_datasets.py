"""Load and verify all three research datasets.

Run from the project root:

    python scripts/check_datasets.py

The script reports dimensions, columns/feature counts, class distributions,
missing values, duplicate rows, and usability for:

* Breast Cancer Wisconsin (Diagnostic)
* MNIST Digit Recognizer (labelled training file)
* Human Activity Recognition (train and test splits, with subjects)

It never downloads data or trains models. Missing or malformed files produce a
clear failure message and a non-zero exit code.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data_loader import (
    HARData,
    DatasetSplit,
    format_validation_report,
    load_breast_cancer,
    load_har,
    load_mnist,
    summarize_dataset,
    validate_dataset,
)

SEPARATOR = "=" * 78


def _format_distribution(class_counts: dict[str, int]) -> str:
    total = sum(class_counts.values())
    parts = [
        f"{label}: {count:,} ({(100.0 * count / total) if total else 0.0:.2f}%)"
        for label, count in class_counts.items()
    ]
    return ", ".join(parts)


def _print_split_details(split: DatasetSplit) -> None:
    report = validate_dataset(split)
    summary = summarize_dataset(split)

    print(f"Name             : {split.name}")
    print(
        f"Dimensions       : {report.n_samples:,} samples x "
        f"{report.n_features:,} features"
    )
    print(f"Feature dtype    : {summary['feature_dtype']}")
    print(f"Classes ({report.n_classes})       : {_format_distribution(report.class_counts)}")
    print(f"Missing values   : {report.missing_feature_values:,} feature, "
          f"{report.missing_labels:,} label")
    print(f"Duplicate rows   : {report.duplicate_rows:,}")
    print(f"Infinite values  : {report.infinite_values:,}")
    if split.subjects is not None:
        unique_subjects = np.unique(split.subjects)
        print(
            f"Subjects         : {len(unique_subjects)} unique "
            f"(min={int(unique_subjects.min())}, max={int(unique_subjects.max())})"
        )
    print(f"Usable           : {'YES' if report.is_usable else 'NO'}")


def check_breast_cancer() -> bool:
    """Load and report the Breast Cancer dataset."""
    print(f"\n{SEPARATOR}\n1. Breast Cancer Wisconsin (Diagnostic)\n{SEPARATOR}")
    split = load_breast_cancer()
    report = validate_dataset(split)

    print(f"Source file      : {split.metadata['source_file']}")
    print(f"Target column    : {split.metadata['target_column']!r}")
    print(f"ID column        : {split.metadata['id_column']!r} (non-feature)")
    print(f"Dropped columns  : {split.metadata['dropped_columns']}")
    print(f"Feature columns  : {len(split.feature_names)}")
    print("Feature names    : " + ", ".join(split.feature_names))
    _print_split_details(split)
    print("\nValidation report:")
    print(format_validation_report(report))
    return report.is_usable


def check_mnist() -> bool:
    """Load and report the MNIST Digit Recognizer training data."""
    print(f"\n{SEPARATOR}\n2. MNIST Digit Recognizer (train.csv)\n{SEPARATOR}")
    split = load_mnist()
    report = validate_dataset(split)
    pixel_min, pixel_max = split.metadata["pixel_value_range"]

    print(f"Source file      : {split.metadata['source_file']}")
    print(f"Label column     : {split.metadata['target_column']!r}")
    print(f"Pixel features   : {len(split.feature_names)} "
          f"({split.feature_names[0]} ... {split.feature_names[-1]})")
    print(f"Image structure  : {split.metadata['image_shape'][0]} x "
          f"{split.metadata['image_shape'][1]} (784 = 28 x 28)")
    print(f"Pixel range      : {pixel_min:g} to {pixel_max:g}")
    _print_split_details(split)
    print("\nValidation report:")
    print(format_validation_report(report))
    return report.is_usable


def check_har() -> bool:
    """Load and report the HAR train/test splits and their subjects."""
    print(f"\n{SEPARATOR}\n3. Human Activity Recognition Using Smartphones\n{SEPARATOR}")
    data: HARData = load_har()
    train_report = validate_dataset(data.train)
    test_report = validate_dataset(data.test)

    subjects_train = set(np.unique(data.train.subjects).tolist())
    subjects_test = set(np.unique(data.test.subjects).tolist())
    overlap = subjects_train.intersection(subjects_test)

    source_location = data.train.metadata.get("source_file") or data.train.metadata.get(
        "source_dir"
    )
    print(f"Source location  : {source_location}")
    print(f"Activity labels  : {data.activity_labels}")
    print(f"Sensor features  : {len(data.train.feature_names)}")
    print("Sample features  : " + ", ".join(data.train.feature_names[:8]) + ", ...")
    print(f"Total samples    : {data.n_samples:,}")
    print(f"Train subjects   : {len(subjects_train)} "
          f"({min(subjects_train)}-{max(subjects_train)})")
    print(f"Test subjects    : {len(subjects_test)} "
          f"({min(subjects_test)}-{max(subjects_test)})")
    print(f"Subject overlap  : {len(overlap)}")

    print("\n-- Train split --")
    _print_split_details(data.train)
    print("\n-- Test split --")
    _print_split_details(data.test)

    print("\nValidation report (train):")
    print(format_validation_report(train_report))
    print("\nValidation report (test):")
    print(format_validation_report(test_report))

    return train_report.is_usable and test_report.is_usable


def main() -> int:
    """Check all datasets and return a process exit code."""
    print(SEPARATOR)
    print("Dataset verification for: DR x Semi-Supervised Classification")
    print(f"Project root: {PROJECT_ROOT}")
    print(SEPARATOR)

    checks = [
        ("Breast Cancer", check_breast_cancer),
        ("MNIST", check_mnist),
        ("HAR", check_har),
    ]

    results: list[tuple[str, bool]] = []
    for label, check in checks:
        try:
            results.append((label, bool(check())))
        except Exception as exc:  # Report and continue checking other datasets.
            results.append((label, False))
            print(f"\n[FAILED] {label}: {type(exc).__name__}: {exc}")

    print(f"\n{SEPARATOR}\nSummary\n{SEPARATOR}")
    for label, usable in results:
        print(f"{label:<16}: {'USABLE' if usable else 'NOT USABLE'}")

    all_usable = all(usable for _, usable in results)
    print(f"\nOverall          : {'ALL DATASETS USABLE' if all_usable else 'ACTION REQUIRED'}")
    return 0 if all_usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
