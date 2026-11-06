"""Dataset loading, validation, and summary utilities.

Supported datasets
------------------
* ``breast_cancer`` -- Breast Cancer Wisconsin (Diagnostic), Kaggle dataset
  ``uciml/breast-cancer-wisconsin-data`` (``data.csv``).
* ``mnist`` -- MNIST Digit Recognizer, Kaggle competition files
  (``train.csv`` with 42,000 labelled samples and 784 pixel features).
* ``har`` -- UCI Human Activity Recognition Using Smartphones, either as the
  original UCI text layout or the UCI-owned Kaggle CSV version, with subject
  identifiers preserved for subject-wise train/test separation.

Raw files are expected under ``data/raw/<dataset>/``. No download is performed
implicitly by the loaders. See ``data/README.md`` for reproducible download
commands and manual fallback instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

MNIST_N_PIXELS = 28 * 28
HAR_N_FEATURES = 561
HAR_N_ACTIVITIES = 6
HAR_ACTIVITIES: tuple[str, ...] = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)

DATASET_SOURCES: dict[str, dict[str, Any]] = {
    "breast_cancer": {
        "display_name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "Kaggle: uciml/breast-cancer-wisconsin-data",
        "url": "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data",
        "expected_files": ["data.csv"],
    },
    "mnist": {
        "display_name": "MNIST Digit Recognizer",
        "source": "Kaggle: Digit Recognizer competition",
        "url": "https://www.kaggle.com/competitions/digit-recognizer/data",
        "expected_files": ["train.csv"],
        "notes": (
            "The official competition download requires Kaggle authentication "
            "and accepting competition rules. See data/README.md."
        ),
    },
    "har": {
        "display_name": "Human Activity Recognition Using Smartphones",
        "source": (
            "UCI Machine Learning Repository dataset 240 "
            "(Kaggle copy: uciml/human-activity-recognition-with-smartphones)"
        ),
        "url": "https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones",
        "expected_files": ["train.csv", "test.csv"],
        "notes": (
            "The loader also accepts the original UCI text layout with "
            "UCI HAR Dataset/features.txt, X_*.txt, y_*.txt, and subject_*.txt."
        ),
    },
}


@dataclass
class DatasetSplit:
    """A feature matrix and labels for one dataset split.

    Attributes
    ----------
    name : str
        Human-readable split name, e.g. ``"breast_cancer"`` or ``"har_train"``.
    X : ndarray of shape (n_samples, n_features)
        Numeric feature matrix.
    y : ndarray of shape (n_samples,)
        Class labels.
    feature_names : list of str
        Feature (column) names.
    target_names : list of str
        Class names present in the dataset.
    subjects : ndarray of shape (n_samples,), optional
        Subject identifiers. Present for HAR and retained for subject-wise
        train/test separation in later phases.
    metadata : dict
        Source and schema details useful for reporting.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    subjects: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError(f"{self.name}: X must be two-dimensional, got {self.X.shape}.")
        if self.X.shape[0] != len(self.y):
            raise ValueError(
                f"{self.name}: X has {self.X.shape[0]} rows but y has {len(self.y)} labels."
            )
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"{self.name}: X has {self.X.shape[1]} columns but "
                f"{len(self.feature_names)} feature names were provided."
            )
        if self.subjects is not None and len(self.subjects) != len(self.y):
            raise ValueError(
                f"{self.name}: subjects has {len(self.subjects)} values but y has "
                f"{len(self.y)} labels."
            )

    @property
    def n_samples(self) -> int:
        """Number of samples in the split."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Number of features in the split."""
        return int(self.X.shape[1])


@dataclass
class HARData:
    """Train/test splits for HAR with activity labels and subject identifiers."""

    train: DatasetSplit
    test: DatasetSplit
    activity_labels: dict[int, str]

    @property
    def n_samples(self) -> int:
        """Total number of samples across both splits."""
        return self.train.n_samples + self.test.n_samples


@dataclass
class ValidationReport:
    """Result of validating one dataset split."""

    name: str
    n_samples: int
    n_features: int
    n_classes: int
    class_counts: dict[str, int]
    missing_feature_values: int
    missing_labels: int
    duplicate_rows: int
    infinite_values: int
    warnings: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """Whether the split meets the minimum requirements for modeling."""
        return (
            self.n_samples > 0
            and self.n_features > 0
            and self.n_classes > 1
            and self.missing_feature_values == 0
            and self.missing_labels == 0
            and self.infinite_values == 0
        )


def list_datasets() -> list[str]:
    """Return the registry keys of all supported datasets."""
    return sorted(DATASET_SOURCES)


def _resolve_dir(raw_dir: str | Path | None, default_subdir: str) -> Path:
    if raw_dir is None:
        return RAW_DATA_DIR / default_subdir
    return Path(raw_dir)


def _require_file(path: Path, dataset: str) -> None:
    if path.is_file():
        return
    source = DATASET_SOURCES[dataset]
    expected = ", ".join(source["expected_files"])
    raise FileNotFoundError(
        f"Missing {dataset} file: {path}\n"
        f"Expected files: {expected}\n"
        f"Source: {source['url']}\n"
        "See data/README.md for download and manual placement instructions."
    )


def load_breast_cancer(raw_dir: str | Path | None = None) -> DatasetSplit:
    """Load and schema-check the Kaggle Breast Cancer Wisconsin dataset.

    Parameters
    ----------
    raw_dir : str or Path, optional
        Directory containing ``data.csv``. Defaults to
        ``data/raw/breast_cancer``.

    Returns
    -------
    DatasetSplit
        ``X`` contains the 30 numeric diagnostic features; ``y`` contains the
        string labels ``"benign"`` and ``"malignant"``.

    Raises
    ------
    FileNotFoundError
        If ``data.csv`` is missing.
    ValueError
        If required columns, numeric features, or expected diagnosis codes are
        missing.

    Notes
    -----
    The Kaggle file contains non-feature columns ``id`` and ``diagnosis`` plus
    a fully empty ``Unnamed: 32`` column. Both non-feature columns are excluded
    from ``X``; the empty column is dropped and recorded in ``metadata``.
    """
    path = _resolve_dir(raw_dir, "breast_cancer") / "data.csv"
    _require_file(path, "breast_cancer")

    frame = pd.read_csv(path)

    required = {"id", "diagnosis"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Breast Cancer file {path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dropped_columns = [
        column
        for column in frame.columns
        if column.lower().startswith("unnamed") or frame[column].isna().all()
    ]
    dropped_columns = [c for c in dropped_columns if c not in required]
    frame = frame.drop(columns=dropped_columns)

    diagnosis = frame["diagnosis"].astype(str).str.strip().str.upper()
    label_mapping = {"B": "benign", "M": "malignant"}
    unknown_labels = sorted(set(diagnosis.unique()).difference(label_mapping))
    if unknown_labels:
        raise ValueError(
            f"Unexpected diagnosis values {unknown_labels}; expected 'B' and 'M'."
        )

    feature_columns = [
        column
        for column in frame.columns
        if column not in {"id", "diagnosis"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not feature_columns:
        raise ValueError(f"No numeric feature columns found in {path}.")

    non_numeric = [
        column
        for column in frame.columns
        if column not in {"id", "diagnosis"}
        and column not in feature_columns
    ]
    if non_numeric:
        raise ValueError(f"Unexpected non-numeric feature columns: {non_numeric}")

    X = frame[feature_columns].to_numpy(dtype=np.float64)
    y = diagnosis.map(label_mapping).to_numpy(dtype=object)

    return DatasetSplit(
        name="breast_cancer",
        X=X,
        y=y,
        feature_names=list(feature_columns),
        target_names=["benign", "malignant"],
        metadata={
            "source": DATASET_SOURCES["breast_cancer"]["source"],
            "source_file": str(path),
            "target_column": "diagnosis",
            "id_column": "id",
            "id_values": frame["id"].to_numpy(),
            "dropped_columns": dropped_columns,
            "label_mapping": label_mapping,
        },
    )


def load_mnist(raw_dir: str | Path | None = None) -> DatasetSplit:
    """Load the labelled MNIST Digit Recognizer training file.

    Parameters
    ----------
    raw_dir : str or Path, optional
        Directory containing ``train.csv``. Defaults to ``data/raw/mnist``.

    Returns
    -------
    DatasetSplit
        ``X`` has shape ``(n_samples, 784)`` with pixel intensities 0-255;
        ``y`` contains integer labels 0-9. The flattened 784 values represent
        a 28 x 28 image (row-major).

    Raises
    ------
    FileNotFoundError
        If ``train.csv`` is missing.
    ValueError
        If the label column or 784 expected pixel columns are missing, or if
        the labels are not the ten digits 0-9.
    """
    path = _resolve_dir(raw_dir, "mnist") / "train.csv"
    _require_file(path, "mnist")

    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise ValueError(f"MNIST file {path} is missing the 'label' column.")

    pixel_columns = [f"pixel{i}" for i in range(MNIST_N_PIXELS)]
    missing_pixels = [column for column in pixel_columns if column not in frame.columns]
    if missing_pixels:
        raise ValueError(
            f"MNIST file {path} is missing {len(missing_pixels)} expected pixel "
            f"columns; first missing: {missing_pixels[0]}."
        )

    unexpected_columns = [
        column for column in frame.columns if column not in {"label", *pixel_columns}
    ]
    if unexpected_columns:
        raise ValueError(
            f"MNIST file {path} contains unexpected columns: {unexpected_columns}"
        )

    labels = frame["label"]
    if labels.isna().any():
        raise ValueError(f"MNIST file {path} contains missing labels.")
    if not pd.api.types.is_integer_dtype(labels):
        raise ValueError(f"MNIST labels must be integers, got dtype {labels.dtype}.")

    observed_classes = set(labels.unique().tolist())
    expected_classes = set(range(10))
    if observed_classes != expected_classes:
        raise ValueError(
            f"MNIST labels must be exactly 0-9; observed {sorted(observed_classes)}."
        )

    X = frame[pixel_columns].to_numpy(dtype=np.float32)
    y = labels.to_numpy(dtype=np.int64)
    pixel_min = float(np.nanmin(X))
    pixel_max = float(np.nanmax(X))

    return DatasetSplit(
        name="mnist",
        X=X,
        y=y,
        feature_names=pixel_columns,
        target_names=[str(digit) for digit in range(10)],
        metadata={
            "source": DATASET_SOURCES["mnist"]["source"],
            "source_file": str(path),
            "target_column": "label",
            "image_shape": (28, 28),
            "pixel_value_range": (pixel_min, pixel_max),
        },
    )


def _find_har_csv_files(raw_dir: str | Path | None) -> tuple[Path, Path] | None:
    base = _resolve_dir(raw_dir, "har")
    train_path = base / "train.csv"
    test_path = base / "test.csv"
    if train_path.is_file() and test_path.is_file():
        return train_path, test_path
    if train_path.exists() or test_path.exists():
        raise FileNotFoundError(
            f"Incomplete HAR CSV layout under {base}: both train.csv and "
            "test.csv are required. See data/README.md."
        )
    return None


def _resolve_har_root(raw_dir: str | Path | None) -> Path:
    base = _resolve_dir(raw_dir, "har")
    candidates = [base, base / "UCI HAR Dataset"]
    for candidate in candidates:
        if (
            (candidate / "features.txt").is_file()
            and (candidate / "train" / "X_train.txt").is_file()
            and (candidate / "test" / "X_test.txt").is_file()
        ):
            return candidate

    raise FileNotFoundError(
        f"Could not find the UCI HAR dataset under {base}.\n"
        "Expected 'features.txt', 'train/X_train.txt', and 'test/X_test.txt' "
        "either directly in that directory or inside a 'UCI HAR Dataset' "
        "subdirectory. See data/README.md for download instructions."
    )


def _read_har_activity_labels(root: Path) -> dict[int, str]:
    path = root / "activity_labels.txt"
    _require_file(path, "har")
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["code", "activity"],
        dtype={0: int, 1: str},
    )
    labels = {int(row.code): str(row.activity) for row in frame.itertuples(index=False)}
    if len(labels) != HAR_N_ACTIVITIES:
        raise ValueError(
            f"Expected {HAR_N_ACTIVITIES} activity labels in {path}, got {len(labels)}."
        )
    return labels


def _read_har_features(root: Path) -> list[str]:
    path = root / "features.txt"
    _require_file(path, "har")
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["index", "feature"],
        dtype={0: int, 1: str},
    )
    feature_names = frame["feature"].astype(str).tolist()
    if len(feature_names) != HAR_N_FEATURES:
        raise ValueError(
            f"Expected {HAR_N_FEATURES} HAR features in {path}, got {len(feature_names)}."
        )
    return feature_names


def _load_har_split(
    root: Path,
    split: str,
    feature_names: list[str],
    activity_labels: dict[int, str],
) -> DatasetSplit:
    split_dir = root / split
    X_path = split_dir / f"X_{split}.txt"
    y_path = split_dir / f"y_{split}.txt"
    subject_path = split_dir / f"subject_{split}.txt"
    for path in (X_path, y_path, subject_path):
        _require_file(path, "har")

    X = pd.read_csv(X_path, sep=r"\s+", header=None).to_numpy(dtype=np.float64)
    y_codes = pd.read_csv(y_path, sep=r"\s+", header=None).to_numpy(dtype=np.int64).ravel()
    subjects = (
        pd.read_csv(subject_path, sep=r"\s+", header=None)
        .to_numpy(dtype=np.int64)
        .ravel()
    )

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"{X_path} has {X.shape[1]} features but features.txt lists "
            f"{len(feature_names)}."
        )
    if not (X.shape[0] == len(y_codes) == len(subjects)):
        raise ValueError(
            f"Inconsistent HAR {split} shapes: X={X.shape[0]}, y={len(y_codes)}, "
            f"subjects={len(subjects)}."
        )

    unknown_codes = sorted(set(y_codes.tolist()).difference(activity_labels))
    if unknown_codes:
        raise ValueError(
            f"Unknown activity codes {unknown_codes} in {y_path}; expected "
            f"{sorted(activity_labels)}."
        )

    y = np.array([activity_labels[int(code)] for code in y_codes], dtype=object)

    return DatasetSplit(
        name=f"har_{split}",
        X=X,
        y=y,
        feature_names=feature_names,
        target_names=[activity_labels[code] for code in sorted(activity_labels)],
        subjects=subjects,
        metadata={
            "source": DATASET_SOURCES["har"]["source"],
            "source_dir": str(root),
            "activity_codes": y_codes,
            "subject_ids": sorted(set(subjects.tolist())),
        },
    )


def _load_har_csv_split(path: Path, split: str) -> DatasetSplit:
    """Load one HAR split from the UCI-owned Kaggle CSV layout."""
    frame = pd.read_csv(path)

    required = {"subject", "Activity"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"HAR file {path} is missing required columns: {sorted(missing_columns)}"
        )
    if frame["subject"].isna().any():
        raise ValueError(f"HAR file {path} contains missing subject identifiers.")
    if frame["Activity"].isna().any():
        raise ValueError(f"HAR file {path} contains missing activity labels.")

    feature_names = [column for column in frame.columns if column not in required]
    if len(feature_names) != HAR_N_FEATURES:
        raise ValueError(
            f"Expected {HAR_N_FEATURES} HAR features in {path}, got "
            f"{len(feature_names)}."
        )

    subjects = pd.to_numeric(frame["subject"], errors="raise").to_numpy(dtype=np.int64)
    activities = frame["Activity"].astype(str).str.strip()
    unknown_activities = sorted(set(activities.unique()).difference(HAR_ACTIVITIES))
    if unknown_activities:
        raise ValueError(
            f"Unexpected HAR activities {unknown_activities} in {path}; expected "
            f"{list(HAR_ACTIVITIES)}."
        )

    X = frame[feature_names].to_numpy(dtype=np.float64)
    y = activities.to_numpy(dtype=object)
    code_by_name = {name: index + 1 for index, name in enumerate(HAR_ACTIVITIES)}
    activity_codes = np.array(
        [code_by_name[str(activity)] for activity in y], dtype=np.int64
    )

    return DatasetSplit(
        name=f"har_{split}",
        X=X,
        y=y,
        feature_names=feature_names,
        target_names=list(HAR_ACTIVITIES),
        subjects=subjects,
        metadata={
            "source": DATASET_SOURCES["har"]["source"],
            "source_file": str(path),
            "activity_codes": activity_codes,
            "subject_ids": sorted(set(subjects.tolist())),
        },
    )


def load_har(raw_dir: str | Path | None = None) -> HARData:
    """Load the UCI HAR train and test splits with subject identifiers.

    Two layouts are supported:

    * Kaggle CSV layout: ``train.csv`` and ``test.csv`` with 561 feature
      columns plus ``subject`` and ``Activity``.
    * Original UCI layout: ``features.txt``, ``activity_labels.txt``, and
      ``train``/``test`` directories containing ``X_*.txt``, ``y_*.txt``, and
      ``subject_*.txt``.

    Parameters
    ----------
    raw_dir : str or Path, optional
        Directory containing the HAR files. Defaults to ``data/raw/har``.
        The original UCI layout may be either directly in this directory or
        inside a ``UCI HAR Dataset`` subdirectory.

    Returns
    -------
    HARData
        Train and test splits with 561 features, six activity labels, and
        preserved subject identifiers (``DatasetSplit.subjects``).

    Raises
    ------
    FileNotFoundError
        If neither supported layout is present.
    ValueError
        If feature counts, label codes, or split lengths are inconsistent.
    """
    csv_files = _find_har_csv_files(raw_dir)
    if csv_files is not None:
        train_path, test_path = csv_files
        train = _load_har_csv_split(train_path, "train")
        test = _load_har_csv_split(test_path, "test")

        observed_activities = set(train.y.tolist()).union(test.y.tolist())
        if observed_activities != set(HAR_ACTIVITIES):
            raise ValueError(
                "HAR train/test splits must cover all six activities; observed "
                f"{sorted(observed_activities)}."
            )

        activity_labels = {
            index + 1: name for index, name in enumerate(HAR_ACTIVITIES)
        }
        return HARData(train=train, test=test, activity_labels=activity_labels)

    root = _resolve_har_root(raw_dir)
    feature_names = _read_har_features(root)
    activity_labels = _read_har_activity_labels(root)

    train = _load_har_split(root, "train", feature_names, activity_labels)
    test = _load_har_split(root, "test", feature_names, activity_labels)

    return HARData(train=train, test=test, activity_labels=activity_labels)


def load_dataset(
    name: str,
    raw_dir: str | Path | None = None,
) -> DatasetSplit | HARData:
    """Load a supported dataset by registry key.

    Parameters
    ----------
    name : str
        One of ``"breast_cancer"``, ``"mnist"``, or ``"har"``.
    raw_dir : str or Path, optional
        Override the dataset's raw-data directory.

    Returns
    -------
    DatasetSplit or HARData
        ``HARData`` for ``"har"`` (train/test with subjects); a single
        ``DatasetSplit`` otherwise.
    """
    key = name.strip().lower()
    if key == "breast_cancer":
        return load_breast_cancer(raw_dir)
    if key == "mnist":
        return load_mnist(raw_dir)
    if key == "har":
        return load_har(raw_dir)
    raise ValueError(f"Unknown dataset {name!r}. Supported: {', '.join(list_datasets())}.")


def validate_dataset(split: DatasetSplit) -> ValidationReport:
    """Check a split for missing values, duplicates, and class structure.

    Parameters
    ----------
    split : DatasetSplit
        Dataset split to inspect.

    Returns
    -------
    ValidationReport
        Dimensions, class counts, missing-value counts, duplicate-row count,
        infinite-value count, and usability verdict.
    """
    X = np.asarray(split.X)
    y = np.asarray(split.y)

    feature_frame = pd.DataFrame(X, columns=split.feature_names)
    missing_features = int(feature_frame.isna().sum().sum())
    duplicate_rows = int(feature_frame.duplicated().sum())

    is_numeric = np.issubdtype(X.dtype, np.number)
    infinite_values = int(np.count_nonzero(np.isinf(X))) if is_numeric else 0

    label_series = pd.Series(y, dtype=object)
    missing_labels = int(label_series.isna().sum())
    class_counts_series = label_series.dropna().value_counts().sort_index()
    class_counts = {
        str(label): int(count) for label, count in class_counts_series.items()
    }

    warnings: list[str] = []
    if missing_features:
        warnings.append(f"{missing_features} missing feature values.")
    if missing_labels:
        warnings.append(f"{missing_labels} missing labels.")
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate feature rows.")
    if infinite_values:
        warnings.append(f"{infinite_values} infinite feature values.")
    if len(class_counts) < 2:
        warnings.append("Fewer than two classes; classification is not possible.")

    return ValidationReport(
        name=split.name,
        n_samples=split.n_samples,
        n_features=split.n_features,
        n_classes=len(class_counts),
        class_counts=class_counts,
        missing_feature_values=missing_features,
        missing_labels=missing_labels,
        duplicate_rows=duplicate_rows,
        infinite_values=infinite_values,
        warnings=warnings,
    )


def summarize_dataset(split: DatasetSplit) -> dict[str, Any]:
    """Return a compact summary dictionary for reporting and dashboards."""
    report = validate_dataset(split)
    return {
        "name": split.name,
        "n_samples": report.n_samples,
        "n_features": report.n_features,
        "n_classes": report.n_classes,
        "class_counts": report.class_counts,
        "missing_feature_values": report.missing_feature_values,
        "missing_labels": report.missing_labels,
        "duplicate_rows": report.duplicate_rows,
        "infinite_values": report.infinite_values,
        "feature_dtype": str(split.X.dtype),
        "subject_count": (
            None if split.subjects is None else len(np.unique(split.subjects))
        ),
        "is_usable": report.is_usable,
        "metadata": dict(split.metadata),
    }


def format_validation_report(report: ValidationReport) -> str:
    """Format a validation report as human-readable text."""
    class_distribution = ", ".join(
        f"{label}: {count}" for label, count in report.class_counts.items()
    )
    lines = [
        f"Dataset split : {report.name}",
        f"Dimensions    : {report.n_samples:,} samples x {report.n_features:,} features",
        f"Classes       : {report.n_classes} ({class_distribution})",
        f"Missing values: {report.missing_feature_values:,} feature, "
        f"{report.missing_labels:,} label",
        f"Duplicate rows: {report.duplicate_rows:,}",
        f"Infinite values: {report.infinite_values:,}",
        f"Usable        : {'YES' if report.is_usable else 'NO'}",
    ]
    if report.warnings:
        lines.append("Warnings      : " + "; ".join(report.warnings))
    return "\n".join(lines)
