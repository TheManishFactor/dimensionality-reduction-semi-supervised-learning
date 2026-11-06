"""Preprocessing, splitting, scaling, and leakage checks.

Pipeline per dataset
--------------------
1. Load raw data through :mod:`src.data_loader` (this module never downloads).
2. Separate features from non-feature columns and preserve feature names.
3. Create a held-out test split (Breast Cancer, MNIST) or keep the official
   HAR train/test split, which is already separated by subject.
4. Fit ``StandardScaler`` on the training partition only and transform the
   training and test partitions with those same statistics.
5. Draw reproducible stratified labelled/unlabelled subsets at 10%, 25%, and
   50% of the training partition. Unlabelled true labels are kept in a
   separate research-only dictionary and are never exposed through the
   model-facing split objects.
6. Validate dimensions, finiteness, proportions, class coverage, subject
   separation, and scaler scope, then record split metadata.

Leakage rules enforced here
---------------------------
* Test data never contributes to scaling statistics or labelled sampling.
* The scaler is fit on the training partition only. Using the complete
  training pool (labelled + unlabelled) for scaling is permitted in the
  semi-supervised setting; using test rows is not.
* Labelled and unlabelled indices are disjoint and together cover the
  training partition exactly.
* HAR subjects never appear in both the train and test partitions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_loader import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    DatasetSplit,
    HARData,
    load_dataset,
)

EXPECTED_FEATURE_COUNTS: dict[str, int] = {
    "breast_cancer": 30,
    "mnist": 784,
    "har": 561,
}
DEFAULT_PROPORTIONS: tuple[float, ...] = (0.10, 0.25, 0.50)
DEFAULT_SPLIT_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "preprocessing_splits.json"
)

# Columns that must never enter the feature matrix. The loaders already
# exclude them; this set is a second line of defence.
NON_FEATURE_COLUMNS = frozenset(
    {"id", "diagnosis", "label", "subject", "activity", "unnamed: 32"}
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainTestIndices:
    """Indices of a complete, disjoint train/test partition.

    ``train_indices`` and ``test_indices`` refer to rows of the full dataset
    for Breast Cancer and MNIST. For HAR the official split already exists, so
    both arrays are simply ``0 .. n-1`` within their own partition.
    """

    train_indices: np.ndarray
    test_indices: np.ndarray
    n_samples: int

    def __post_init__(self) -> None:
        train = _as_indices(self.train_indices, "train indices")
        test = _as_indices(self.test_indices, "test indices")
        if self.n_samples != train.size + test.size:
            raise ValueError(
                "Train/test indices must partition the dataset: "
                f"{train.size} + {test.size} != {self.n_samples}."
            )
        if np.unique(train).size != train.size or np.unique(test).size != test.size:
            raise ValueError("Train/test indices must not contain duplicates.")
        if np.intersect1d(train, test).size:
            raise ValueError("Train and test indices overlap.")
        combined = np.concatenate([train, test])
        if combined.min() < 0 or combined.max() >= self.n_samples:
            raise ValueError("Train/test indices are outside the dataset bounds.")
        if np.unique(combined).size != self.n_samples:
            raise ValueError("Train/test indices do not cover every dataset row.")
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "test_indices", test)

    @property
    def n_train(self) -> int:
        return int(self.train_indices.size)

    @property
    def n_test(self) -> int:
        return int(self.test_indices.size)


@dataclass
class SemiSupervisedSplit:
    """Index-based labelled/unlabelled division of one training partition.

    The object deliberately stores only the labelled targets. The unlabelled
    true labels live outside this container (see
    :attr:`PreprocessedDataset.research_unlabelled_labels`), which prevents
    accidental use during self-training.
    """

    proportion: float
    labelled_indices: np.ndarray
    unlabelled_indices: np.ndarray
    y_labelled: np.ndarray
    n_train: int
    random_state: int

    def __post_init__(self) -> None:
        proportion = validate_labelled_proportion(self.proportion)
        labelled = _as_indices(self.labelled_indices, "labelled indices")
        unlabelled = _as_indices(self.unlabelled_indices, "unlabelled indices")
        if labelled.size == 0 or unlabelled.size == 0:
            raise ValueError("Both the labelled and unlabelled subsets must be non-empty.")
        if int(self.n_train) != labelled.size + unlabelled.size:
            raise ValueError(
                "Labelled and unlabelled subsets must partition the training data: "
                f"{labelled.size} + {unlabelled.size} != {self.n_train}."
            )
        if np.unique(labelled).size != labelled.size:
            raise ValueError("Labelled indices must not contain duplicates.")
        if np.unique(unlabelled).size != unlabelled.size:
            raise ValueError("Unlabelled indices must not contain duplicates.")
        if np.intersect1d(labelled, unlabelled).size:
            raise ValueError("Labelled and unlabelled indices overlap.")
        combined = np.concatenate([labelled, unlabelled])
        if combined.min() < 0 or combined.max() >= self.n_train:
            raise ValueError("Indices fall outside the training partition.")
        if np.unique(combined).size != self.n_train:
            raise ValueError("Labelled/unlabelled indices do not cover the training rows.")

        y_labelled = np.asarray(self.y_labelled)
        if y_labelled.ndim != 1 or y_labelled.size != labelled.size:
            raise ValueError(
                "y_labelled must contain exactly one target for every labelled row: "
                f"{y_labelled.size} targets for {labelled.size} rows."
            )

        object.__setattr__(self, "proportion", proportion)
        object.__setattr__(self, "labelled_indices", labelled)
        object.__setattr__(self, "unlabelled_indices", unlabelled)
        object.__setattr__(self, "y_labelled", y_labelled)
        object.__setattr__(self, "n_train", int(self.n_train))
        object.__setattr__(self, "random_state", int(self.random_state))

    @property
    def n_labelled(self) -> int:
        return int(self.labelled_indices.size)

    @property
    def n_unlabelled(self) -> int:
        return int(self.unlabelled_indices.size)

    @property
    def actual_proportion(self) -> float:
        return self.n_labelled / self.n_train


@dataclass
class PreprocessedDataset:
    """Scaled features, fixed splits, and semi-supervised index settings."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    scaler: StandardScaler
    semi_supervised: dict[float, SemiSupervisedSplit]
    research_unlabelled_labels: dict[float, np.ndarray]
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_subjects: np.ndarray | None = None
    test_subjects: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attribute in ("X_train", "X_test", "y_train", "y_test"):
            object.__setattr__(self, attribute, np.asarray(getattr(self, attribute)))

        if self.X_train.ndim != 2 or self.X_test.ndim != 2:
            raise ValueError(f"{self.name}: scaled features must be two-dimensional.")
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError(f"{self.name}: train and test feature counts differ.")
        if self.X_train.shape[1] != len(self.feature_names):
            raise ValueError(f"{self.name}: feature name count does not match the data.")
        if self.X_train.shape[0] != len(self.y_train):
            raise ValueError(f"{self.name}: X_train and y_train lengths differ.")
        if self.X_test.shape[0] != len(self.y_test):
            raise ValueError(f"{self.name}: X_test and y_test lengths differ.")
        if self.X_train.shape[0] != len(self.train_indices):
            raise ValueError(f"{self.name}: train index count does not match X_train.")
        if self.X_test.shape[0] != len(self.test_indices):
            raise ValueError(f"{self.name}: test index count does not match X_test.")
        if not hasattr(self.scaler, "mean_"):
            raise ValueError(f"{self.name}: the scaler has not been fitted.")
        if self.train_subjects is not None:
            object.__setattr__(
                self, "train_subjects", np.asarray(self.train_subjects)
            )
            if len(self.train_subjects) != self.X_train.shape[0]:
                raise ValueError(f"{self.name}: train subject count does not match X_train.")
        if self.test_subjects is not None:
            object.__setattr__(self, "test_subjects", np.asarray(self.test_subjects))
            if len(self.test_subjects) != self.X_test.shape[0]:
                raise ValueError(f"{self.name}: test subject count does not match X_test.")

        if set(self.semi_supervised) != set(self.research_unlabelled_labels):
            raise ValueError(
                f"{self.name}: semi-supervised settings and research-label settings differ."
            )
        for proportion, split in self.semi_supervised.items():
            if split.n_train != self.X_train.shape[0]:
                raise ValueError(
                    f"{self.name}: split {proportion} does not match the training size."
                )
            research = np.asarray(self.research_unlabelled_labels[proportion])
            expected = self.y_train[split.unlabelled_indices]
            if research.ndim != 1 or research.size != expected.size:
                raise ValueError(
                    f"{self.name}: research labels for {proportion} have the wrong length."
                )
            if not np.array_equal(research, expected):
                raise ValueError(
                    f"{self.name}: research labels for {proportion} do not match y_train."
                )

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    def get_labelled_features(self, proportion: float) -> np.ndarray:
        """Return the scaled features for the labelled subset of ``proportion``."""
        split = self._split_for(proportion)
        return self.X_train[split.labelled_indices]

    def get_labelled_targets(self, proportion: float) -> np.ndarray:
        """Return the targets for the labelled subset of ``proportion``."""
        return self._split_for(proportion).y_labelled

    def get_unlabelled_features(self, proportion: float) -> np.ndarray:
        """Return unlabelled features only; no unlabelled targets are exposed."""
        split = self._split_for(proportion)
        return self.X_train[split.unlabelled_indices]

    def get_research_unlabelled_labels(self, proportion: float) -> np.ndarray:
        """Return unlabelled true labels for diagnostics only.

        These labels must never be given to a self-training loop or a model.
        """
        return self.research_unlabelled_labels[self._key(proportion)]

    def _key(self, proportion: float) -> float:
        key = float(proportion)
        if key not in self.semi_supervised:
            raise KeyError(
                f"{self.name}: no semi-supervised setting for proportion {proportion!r}. "
                f"Available: {sorted(self.semi_supervised)}."
            )
        return key

    def _split_for(self, proportion: float) -> SemiSupervisedSplit:
        return self.semi_supervised[self._key(proportion)]


@dataclass(frozen=True)
class CheckResult:
    """One PASS/FAIL verification check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class PreprocessingReport:
    """Verification result for one preprocessed dataset."""

    name: str
    checks: list[CheckResult]
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def _as_indices(values: Iterable[int], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{label} must not be empty.")
    return array


def _class_counts(y: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(np.asarray(y), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}


def _class_shares(y: np.ndarray) -> dict[str, float]:
    counts = _class_counts(y)
    total = sum(counts.values())
    return {label: count / total for label, count in counts.items()}


def _dataset_key(name: str) -> str:
    key = name.strip().lower()
    for dataset in EXPECTED_FEATURE_COUNTS:
        if key == dataset or key.startswith(f"{dataset}_"):
            return dataset
    raise ValueError(
        f"Unknown dataset {name!r}. Supported: {', '.join(EXPECTED_FEATURE_COUNTS)}."
    )


def _configured_proportions(config: dict[str, Any]) -> tuple[float, ...]:
    raw = config.get("data", {}).get("labelled_proportions", DEFAULT_PROPORTIONS)
    if not raw:
        raise ValueError("data.labelled_proportions must contain at least one value.")
    proportions = [validate_labelled_proportion(value) for value in raw]
    if len(set(proportions)) != len(proportions):
        raise ValueError("data.labelled_proportions contains duplicate values.")
    return tuple(sorted(proportions))


def validate_labelled_proportion(proportion: float) -> float:
    """Validate that a labelled-data proportion lies strictly in (0, 1).

    Parameters
    ----------
    proportion : float
        Fraction of non-test data treated as labelled.

    Returns
    -------
    float
        The validated proportion.

    Raises
    ------
    TypeError
        If ``proportion`` is not a real number.
    ValueError
        If ``proportion`` is not strictly between 0 and 1.
    """
    if isinstance(proportion, bool) or not isinstance(proportion, (int, float)):
        raise TypeError("Labelled proportion must be a float between 0 and 1.")
    if not 0.0 < float(proportion) < 1.0:
        raise ValueError("Labelled proportion must be strictly between 0 and 1.")
    return float(proportion)


def split_features_target(
    frame: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into a feature matrix and a target series.

    The target column is removed from the features. No other column is
    dropped implicitly, so unexpected columns surface as errors later rather
    than being silently ignored.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} is not present in the frame.")

    features = frame.drop(columns=[target_column])
    target = frame[target_column]
    if features.shape[1] == 0:
        raise ValueError("The frame contains no feature columns after removing the target.")
    return features, target


def separate_features_target(
    split: DatasetSplit,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """Separate a loaded :class:`DatasetSplit` into features and targets.

    Returns
    -------
    X, y, feature_names, subjects
        ``X`` is two-dimensional, ``y`` is one-dimensional, ``subjects`` is
        ``None`` unless the loader preserved subject identifiers (HAR).
    """
    if not isinstance(split, DatasetSplit):
        raise TypeError("split must be a DatasetSplit instance.")

    X = np.asarray(split.X)
    y = np.asarray(split.y)
    feature_names = [str(name) for name in split.feature_names]

    if X.ndim != 2:
        raise ValueError(f"{split.name}: X must be two-dimensional, got {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"{split.name}: y must be one-dimensional, got {y.shape}.")
    if X.shape[0] != y.size:
        raise ValueError(
            f"{split.name}: X has {X.shape[0]} rows but y has {y.size} labels."
        )
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"{split.name}: X has {X.shape[1]} columns but {len(feature_names)} "
            "feature names were provided."
        )

    non_feature = sorted(
        {name.strip().lower() for name in feature_names}.intersection(NON_FEATURE_COLUMNS)
    )
    if non_feature:
        raise ValueError(
            f"{split.name}: non-feature columns leaked into the feature matrix: "
            f"{non_feature}."
        )

    subjects = None
    if split.subjects is not None:
        subjects = np.asarray(split.subjects)
        if subjects.ndim != 1 or subjects.size != y.size:
            raise ValueError(f"{split.name}: subject identifiers do not match the rows.")

    return X, y, feature_names, subjects


# ---------------------------------------------------------------------------
# Splitting and scaling
# ---------------------------------------------------------------------------
def create_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> TrainTestIndices:
    """Create the held-out test split used by every experiment.

    The split is created once per dataset and seed and then reused across all
    DR, classifier, and labelled-proportion conditions. Indices are returned
    (not duplicated arrays) so large datasets such as MNIST stay memory
    efficient.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got {X.shape}.")
    if y.ndim != 1 or y.size != X.shape[0]:
        raise ValueError("y must contain exactly one label per row of X.")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")
    if X.shape[0] < 2:
        raise ValueError("At least two samples are required to create a split.")

    indices = np.arange(X.shape[0])
    train_indices, test_indices = train_test_split(
        indices,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y if stratify else None,
    )
    return TrainTestIndices(
        train_indices=np.sort(train_indices),
        test_indices=np.sort(test_indices),
        n_samples=int(X.shape[0]),
    )


def fit_scaler(X_train: np.ndarray, method: str = "standard") -> StandardScaler:
    """Fit a feature scaler on training data only.

    Parameters
    ----------
    X_train : ndarray
        Training features. This must never include test rows.
    method : str, default="standard"
        Currently only ``"standard"`` (``StandardScaler``) is supported.
    """
    if method != "standard":
        raise ValueError(
            f"Unsupported scaling method {method!r}; Phase 3 uses 'standard'."
        )

    X = np.asarray(X_train)
    if X.ndim != 2:
        raise ValueError(f"X_train must be two-dimensional, got {X.shape}.")
    if X.shape[0] < 2:
        raise ValueError("At least two training rows are required to fit a scaler.")
    if not np.isfinite(X).all():
        raise ValueError("X_train contains missing or infinite values.")

    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def transform_features(
    scaler: StandardScaler,
    X: np.ndarray,
    name: str = "features",
) -> np.ndarray:
    """Transform features with an already fitted scaler."""
    if not hasattr(scaler, "mean_"):
        raise ValueError(f"{name}: the scaler has not been fitted.")
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {array.shape}.")
    if array.shape[1] != len(scaler.mean_):
        raise ValueError(
            f"{name} has {array.shape[1]} features but the scaler was fitted on "
            f"{len(scaler.mean_)}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains missing or infinite values.")
    return scaler.transform(array)


def scale_features(
    X_fit: np.ndarray,
    *others: np.ndarray,
    method: str = "standard",
    return_scaler: bool = False,
) -> tuple[np.ndarray, ...]:
    """Fit a scaler on ``X_fit`` and transform it plus ``others``.

    Returns the transformed arrays in the same order as the inputs. When
    ``return_scaler`` is true the fitted scaler is appended as the final
    element, which is useful when the same statistics must be reused later.
    """
    scaler = fit_scaler(X_fit, method=method)
    transformed = [transform_features(scaler, X_fit, "X_fit")]
    transformed.extend(
        transform_features(scaler, array, f"array {index}")
        for index, array in enumerate(others, start=1)
    )
    if return_scaler:
        transformed.append(scaler)
    return tuple(transformed)


def create_labelled_unlabelled_split(
    X: np.ndarray,
    y: np.ndarray,
    labelled_proportion: float,
    random_state: int = 42,
    stratify: bool = True,
) -> SemiSupervisedSplit:
    """Split training data into labelled and unlabelled index subsets.

    ``X`` is used only for its row count and shape validation; the returned
    object stores indices rather than copied arrays. The unlabelled true
    labels are intentionally not returned here. Recover them from ``y`` with
    ``y[split.unlabelled_indices]`` and keep them in a research-only variable.

    Every class is guaranteed to appear in the labelled subset when the
    labelled budget allows it (``n_labelled >= n_classes``).
    """
    proportion = validate_labelled_proportion(labelled_proportion)
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got {X.shape}.")
    if y.ndim != 1 or y.size != X.shape[0]:
        raise ValueError("y must contain exactly one label per row of X.")

    n_train = int(X.shape[0])
    n_labelled = int(round(proportion * n_train))
    if n_labelled <= 0 or n_labelled >= n_train:
        raise ValueError(
            f"Proportion {proportion} yields {n_labelled} labelled rows out of "
            f"{n_train}; the labelled subset must be non-empty and smaller than "
            "the training partition."
        )

    classes = np.unique(y)
    if n_labelled < classes.size:
        raise ValueError(
            f"A labelled budget of {n_labelled} rows cannot cover all "
            f"{classes.size} classes. Increase the proportion."
        )

    indices = np.arange(n_train)
    if stratify:
        labelled_indices, unlabelled_indices = train_test_split(
            indices,
            train_size=n_labelled,
            random_state=int(random_state),
            stratify=y,
        )
    else:
        rng = np.random.default_rng(int(random_state))
        labelled_indices = rng.choice(indices, size=n_labelled, replace=False)
        mask = np.ones(n_train, dtype=bool)
        mask[labelled_indices] = False
        unlabelled_indices = indices[mask]

    labelled_indices = np.sort(labelled_indices)
    unlabelled_indices = np.sort(unlabelled_indices)

    missing_classes = sorted(set(classes.tolist()) - set(y[labelled_indices].tolist()))
    if missing_classes:
        raise ValueError(
            "The labelled subset is missing classes "
            f"{missing_classes}; every class must be represented when feasible."
        )

    return SemiSupervisedSplit(
        proportion=proportion,
        labelled_indices=labelled_indices,
        unlabelled_indices=unlabelled_indices,
        y_labelled=y[labelled_indices].copy(),
        n_train=n_train,
        random_state=random_state,
    )


def create_semi_supervised_settings(
    X: np.ndarray,
    y: np.ndarray,
    proportions: Sequence[float],
    random_state: int = 42,
    stratify: bool = True,
) -> dict[float, SemiSupervisedSplit]:
    """Create one labelled/unlabelled split per configured proportion."""
    if not proportions:
        raise ValueError("At least one labelled proportion is required.")
    settings: dict[float, SemiSupervisedSplit] = {}
    for proportion in sorted(validate_labelled_proportion(value) for value in proportions):
        if proportion in settings:
            raise ValueError(f"Duplicate labelled proportion: {proportion}.")
        settings[proportion] = create_labelled_unlabelled_split(
            X,
            y,
            labelled_proportion=proportion,
            random_state=random_state,
            stratify=stratify,
        )
    return settings


# ---------------------------------------------------------------------------
# Dataset pipeline
# ---------------------------------------------------------------------------
def _holdout_partitions(
    key: str,
    config: dict[str, Any],
    raw_dir: str | Path | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    data_config = config.get("data", {})
    split_config = data_config.get("splitting", {}).get(key, {})
    strategy = str(split_config.get("strategy", "stratified_holdout"))
    if strategy != "stratified_holdout":
        raise ValueError(
            f"{key}: configured strategy {strategy!r} does not match the "
            "stratified-holdout datasets."
        )

    test_size = float(split_config.get("test_size", data_config.get("test_size", 0.20)))
    stratify = bool(split_config.get("stratify", data_config.get("stratify", True)))

    loaded = load_dataset(key, raw_dir)
    if not isinstance(loaded, DatasetSplit):
        raise ValueError(f"{key}: expected a single DatasetSplit from the loader.")

    X_full, y_full, feature_names, _ = separate_features_target(loaded)
    expected = EXPECTED_FEATURE_COUNTS[key]
    if X_full.shape[1] != expected:
        raise ValueError(
            f"{key}: expected {expected} features, found {X_full.shape[1]}."
        )

    competition_test = None
    if key == "mnist":
        if not bool(split_config.get("ignore_competition_test", True)):
            raise ValueError(
                "mnist: the competition test.csv has no labels and must be ignored. "
                "Set data.splitting.mnist.ignore_competition_test: true."
            )
        base_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR / "mnist"
        competition_test = base_dir / "test.csv"

    indices = create_train_test_split(
        X_full,
        y_full,
        test_size=test_size,
        random_state=int(config["project"]["random_seed"]),
        stratify=stratify,
    )
    X_train = X_full[indices.train_indices]
    y_train = y_full[indices.train_indices]
    X_test = X_full[indices.test_indices]
    y_test = y_full[indices.test_indices]

    metadata = {
        "strategy": strategy,
        "random_seed": int(config["project"]["random_seed"]),
        "test_size": test_size,
        "stratify": stratify,
        "n_original": int(X_full.shape[0]),
        "official_test": False,
        "subject_separation": False,
        "source": loaded.metadata.get("source"),
        "source_file": loaded.metadata.get("source_file"),
    }
    if key == "mnist":
        metadata["competition_test_ignored"] = True
        metadata["competition_test_present"] = bool(
            competition_test is not None and competition_test.is_file()
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        list(loaded.target_names),
        indices.train_indices,
        indices.test_indices,
        metadata,
    )


def _official_har_partitions(
    config: dict[str, Any],
    raw_dir: str | Path | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
]:
    split_config = config.get("data", {}).get("splitting", {}).get("har", {})
    strategy = str(split_config.get("strategy", "official_train_test"))
    if strategy != "official_train_test":
        raise ValueError(
            f"har: configured strategy {strategy!r} does not match the official split."
        )
    if not bool(split_config.get("preserve_subject_separation", True)):
        raise ValueError(
            "har: preserve_subject_separation must remain true; mixing subjects "
            "between train and test invalidates the evaluation."
        )

    loaded = load_dataset("har", raw_dir)
    if not isinstance(loaded, HARData):
        raise ValueError("har: expected HARData from the loader.")

    X_train, y_train, feature_names, train_subjects = separate_features_target(loaded.train)
    X_test, y_test, test_feature_names, test_subjects = separate_features_target(loaded.test)

    if feature_names != test_feature_names:
        raise ValueError("har: train and test feature names do not match.")
    expected = EXPECTED_FEATURE_COUNTS["har"]
    if X_train.shape[1] != expected or X_test.shape[1] != expected:
        raise ValueError(
            f"har: expected {expected} features, found "
            f"{X_train.shape[1]} (train) and {X_test.shape[1]} (test)."
        )
    if train_subjects is None or test_subjects is None:
        raise ValueError("har: subject identifiers are required for separation checks.")

    overlap = np.intersect1d(np.unique(train_subjects), np.unique(test_subjects))
    if overlap.size:
        raise ValueError(
            "har: subject leakage detected; these subjects appear in both "
            f"partitions: {overlap.tolist()}."
        )

    metadata = {
        "strategy": strategy,
        "random_seed": int(config["project"]["random_seed"]),
        "test_size": None,
        "stratify": False,
        "n_original": int(X_train.shape[0] + X_test.shape[0]),
        "official_test": True,
        "subject_separation": True,
        "source": loaded.train.metadata.get("source"),
        "source_file": loaded.train.metadata.get("source_file")
        or loaded.train.metadata.get("source_dir"),
        "activity_labels": dict(loaded.activity_labels),
    }

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        list(loaded.train.target_names),
        np.arange(X_train.shape[0], dtype=np.int64),
        np.arange(X_test.shape[0], dtype=np.int64),
        train_subjects,
        test_subjects,
        metadata,
    )


def preprocess_dataset(
    name: str,
    config: dict[str, Any],
    raw_dir: str | Path | None = None,
) -> PreprocessedDataset:
    """Run the complete preprocessing pipeline for one dataset.

    Scaling is fitted on the full training partition (labelled + unlabelled
    pool), which is permitted for this semi-supervised study, and never on the
    test partition. The labelled/unlabelled index settings are then derived
    from the training partition only.
    """
    key = _dataset_key(name)
    proportions = _configured_proportions(config)
    scaling_config = config.get("data", {}).get("scaling", {})
    method = str(scaling_config.get("method", "standard"))

    if key == "har":
        (
            X_train_raw,
            X_test_raw,
            y_train,
            y_test,
            feature_names,
            target_names,
            train_indices,
            test_indices,
            train_subjects,
            test_subjects,
            metadata,
        ) = _official_har_partitions(config, raw_dir)
    else:
        (
            X_train_raw,
            X_test_raw,
            y_train,
            y_test,
            feature_names,
            target_names,
            train_indices,
            test_indices,
            metadata,
        ) = _holdout_partitions(key, config, raw_dir)
        train_subjects = None
        test_subjects = None

    scaler = fit_scaler(X_train_raw, method=method)
    X_train = transform_features(scaler, X_train_raw, f"{key} training features")
    X_test = transform_features(scaler, X_test_raw, f"{key} test features")

    stratify_settings = bool(metadata.get("stratify", True))
    semi_supervised = create_semi_supervised_settings(
        X_train_raw,
        y_train,
        proportions,
        random_state=int(config["project"]["random_seed"]),
        stratify=stratify_settings,
    )
    research_unlabelled_labels = {
        proportion: y_train[split.unlabelled_indices].copy()
        for proportion, split in semi_supervised.items()
    }

    metadata.update(
        {
            "n_samples_fit_scaler": int(X_train_raw.shape[0]),
            "scaling": {
                "method": method,
                "fit_scope": "training_partition_only",
                "n_samples_fit": int(X_train_raw.shape[0]),
            },
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "labelled_proportions": list(proportions),
        }
    )

    return PreprocessedDataset(
        name=key,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
        scaler=scaler,
        semi_supervised=semi_supervised,
        research_unlabelled_labels=research_unlabelled_labels,
        train_indices=np.asarray(train_indices),
        test_indices=np.asarray(test_indices),
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        metadata=metadata,
    )


def preprocess_all_datasets(
    config: dict[str, Any],
    raw_dir: str | Path | None = None,
) -> dict[str, PreprocessedDataset]:
    """Preprocess every enabled dataset in configuration order."""
    datasets = config.get("data", {}).get("datasets", {})
    keys = [
        key
        for key, settings in datasets.items()
        if isinstance(settings, dict) and settings.get("enabled", True)
    ]
    if not keys:
        keys = list(EXPECTED_FEATURE_COUNTS)
    return {
        key: preprocess_dataset(key, config, raw_dir=raw_dir)
        for key in keys
        if key in EXPECTED_FEATURE_COUNTS
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _check(
    checks: list[CheckResult],
    name: str,
    passed: bool,
    detail: str = "",
) -> None:
    checks.append(CheckResult(name=name, passed=bool(passed), detail=detail))


def _drift(first: dict[str, float], second: dict[str, float]) -> float:
    labels = set(first).union(second)
    return max(abs(first.get(label, 0.0) - second.get(label, 0.0)) for label in labels)


def validate_preprocessing(
    result: PreprocessedDataset,
    config: dict[str, Any],
) -> PreprocessingReport:
    """Verify dimensions, leakage rules, scaling, and split integrity."""
    checks: list[CheckResult] = []
    warnings: list[str] = []
    key = _dataset_key(result.name)
    expected = EXPECTED_FEATURE_COUNTS[key]
    proportions = _configured_proportions(config)

    n_features = result.n_features
    _check(
        checks,
        "feature_count",
        n_features == expected and len(result.feature_names) == expected,
        f"{n_features} features (expected {expected}).",
    )
    _check(
        checks,
        "feature_names",
        len(set(result.feature_names)) == len(result.feature_names)
        and all(name.strip() for name in result.feature_names),
        f"{len(result.feature_names)} unique non-empty names; "
        f"first={result.feature_names[0]!r}, last={result.feature_names[-1]!r}.",
    )
    _check(
        checks,
        "shapes",
        result.X_train.ndim == 2
        and result.X_test.ndim == 2
        and result.X_train.shape[1] == result.X_test.shape[1]
        and result.X_train.shape[0] == result.y_train.size
        and result.X_test.shape[0] == result.y_test.size,
        f"train={result.X_train.shape}, test={result.X_test.shape}.",
    )
    _check(
        checks,
        "finite_values",
        bool(np.isfinite(result.X_train).all())
        and bool(np.isfinite(result.X_test).all()),
        "No missing or infinite values in the scaled train/test matrices.",
    )

    n_seen = result.scaler.n_samples_seen_
    n_seen_int = int(np.max(n_seen)) if np.ndim(n_seen) else int(n_seen)
    _check(
        checks,
        "scaler_fit_scope",
        n_seen_int == result.n_train
        and int(result.scaler.mean_.size) == n_features
        and int(result.scaler.mean_.size) == int(result.scaler.scale_.size),
        f"scaler saw {n_seen_int:,} training rows; test rows "
        f"({result.n_test:,}) were excluded.",
    )

    variance = np.asarray(result.scaler.var_, dtype=np.float64)
    nonconstant = variance > 0.0
    train_mean = np.asarray(
        np.mean(result.X_train, axis=0, dtype=np.float64), dtype=np.float64
    )
    train_std = np.asarray(
        np.std(result.X_train, axis=0, dtype=np.float64), dtype=np.float64
    )
    max_abs_mean = (
        float(np.max(np.abs(train_mean[nonconstant]))) if nonconstant.any() else 0.0
    )
    max_std_error = (
        float(np.max(np.abs(train_std[nonconstant] - 1.0)))
        if nonconstant.any()
        else 0.0
    )
    constant_abs_max = (
        float(np.max(np.abs(result.X_train[:, ~nonconstant])))
        if (~nonconstant).any()
        else 0.0
    )
    _check(
        checks,
        "standardization",
        max_abs_mean <= 1e-6 and max_std_error <= 5e-3 and constant_abs_max <= 1e-6,
        f"max |train mean|={max_abs_mean:.2e}; max |train std - 1|="
        f"{max_std_error:.2e}; constant features={int((~nonconstant).sum())}.",
    )

    train_counts = _class_counts(result.y_train)
    test_counts = _class_counts(result.y_test)
    train_shares = _class_shares(result.y_train)
    test_shares = _class_shares(result.y_test)
    if set(train_counts) != set(test_counts):
        warnings.append("Train and test class sets differ.")

    if result.metadata.get("official_test"):
        train_indices = np.asarray(result.train_indices)
        test_indices = np.asarray(result.test_indices)
        partition_ok = (
            np.array_equal(train_indices, np.arange(result.n_train))
            and np.array_equal(test_indices, np.arange(result.n_test))
        )
        _check(
            checks,
            "official_partition",
            partition_ok,
            "Official HAR train/test rows are kept in their original order.",
        )
        drift_limit = 0.10
        drift_label = "official split drift limit 10%"
    else:
        try:
            TrainTestIndices(
                train_indices=np.asarray(result.train_indices),
                test_indices=np.asarray(result.test_indices),
                n_samples=int(result.metadata["n_original"]),
            )
            partition_ok = True
            detail = (
                f"{result.n_train:,} train + {result.n_test:,} test rows "
                f"cover all {result.metadata['n_original']:,} rows."
            )
        except (ValueError, KeyError) as exc:
            partition_ok = False
            detail = str(exc)
        _check(checks, "train_test_partition", partition_ok, detail)
        drift_limit = 0.05
        drift_label = "stratified holdout drift limit 5%"

    train_test_drift = _drift(train_shares, test_shares)
    _check(
        checks,
        "train_test_stratification",
        train_test_drift <= drift_limit,
        f"max class-share drift={train_test_drift:.4f} ({drift_label}).",
    )

    settings_ok = set(result.semi_supervised) == set(proportions)
    setting_details: list[str] = []
    for proportion in proportions:
        split = result.semi_supervised.get(proportion)
        if split is None:
            setting_details.append(f"{proportion:.2f}: missing")
            continue
        labelled_counts = _class_counts(split.y_labelled)
        covered = set(labelled_counts) == set(train_counts)
        actual = split.actual_proportion
        proportion_ok = abs(actual - proportion) <= (1.0 / split.n_train + 1e-9)
        research = result.get_research_unlabelled_labels(proportion)
        research_ok = np.array_equal(research, result.y_train[split.unlabelled_indices])
        no_hidden_targets = not hasattr(split, "y_unlabelled")
        drift = _drift(
            {label: count / split.n_labelled for label, count in labelled_counts.items()},
            train_shares,
        )
        ok = (
            covered
            and proportion_ok
            and research_ok
            and no_hidden_targets
            and drift <= 0.05
            and split.n_train == result.n_train
        )
        settings_ok = settings_ok and ok
        setting_details.append(
            f"{proportion:.2f}: {split.n_labelled:,} labelled / "
            f"{split.n_unlabelled:,} unlabelled (actual {actual:.4f}, "
            f"classes {len(labelled_counts)}/{len(train_counts)}, drift {drift:.4f})"
        )
    _check(
        checks,
        "semi_supervised_settings",
        settings_ok,
        "; ".join(setting_details),
    )

    bounds_ok = all(
        int(split.labelled_indices.max()) < result.n_train
        and int(split.unlabelled_indices.max()) < result.n_train
        and split.n_train == result.n_train
        for split in result.semi_supervised.values()
    )
    _check(
        checks,
        "no_test_leakage",
        bounds_ok,
        "All labelled/unlabelled indices refer to training rows only; "
        "test rows are never sampled.",
    )

    if key == "har":
        train_subjects = np.asarray(result.train_subjects)
        test_subjects = np.asarray(result.test_subjects)
        train_unique = np.unique(train_subjects)
        test_unique = np.unique(test_subjects)
        overlap = np.intersect1d(train_unique, test_unique)
        _check(
            checks,
            "har_subject_separation",
            overlap.size == 0 and len(train_subjects) == result.n_train,
            f"{len(train_unique)} train subjects, {len(test_unique)} test subjects, "
            f"overlap={overlap.tolist()}.",
        )
        _check(
            checks,
            "har_class_coverage",
            set(train_counts) == set(test_counts) == set(result.target_names),
            f"train classes={sorted(train_counts)}; test classes={sorted(test_counts)}.",
        )
    else:
        _check(
            checks,
            "no_subject_mixing",
            result.train_subjects is None and result.test_subjects is None,
            "Not applicable (non-subject dataset); both subject arrays are None.",
        )

    _check(
        checks,
        "seed_recorded",
        int(result.metadata.get("random_seed", -1))
        == int(config["project"]["random_seed"])
        and result.metadata.get("scaling", {}).get("fit_scope")
        == "training_partition_only",
        f"seed={result.metadata.get('random_seed')}; "
        f"scaler fit scope={result.metadata.get('scaling', {}).get('fit_scope')!r}.",
    )

    details = {
        "dataset": key,
        "strategy": result.metadata.get("strategy"),
        "n_original": result.metadata.get("n_original"),
        "n_train": result.n_train,
        "n_test": result.n_test,
        "n_features": n_features,
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "train_test_drift": train_test_drift,
        "labelled_proportions": list(proportions),
        "scaling": result.metadata.get("scaling"),
        "train_subjects": (
            None
            if result.train_subjects is None
            else sorted(int(value) for value in np.unique(result.train_subjects))
        ),
        "test_subjects": (
            None
            if result.test_subjects is None
            else sorted(int(value) for value in np.unique(result.test_subjects))
        ),
    }

    return PreprocessingReport(
        name=key,
        checks=checks,
        details=details,
        warnings=warnings,
    )


def format_preprocessing_report(report: PreprocessingReport) -> str:
    """Format a preprocessing verification report as readable text."""
    status = "PASS" if report.passed else "FAIL"
    details = report.details
    lines = [
        f"Dataset : {report.name}",
        f"Status  : {status}",
        f"Strategy: {details.get('strategy')}",
        f"Shape   : {details.get('n_train'):,} train x {details.get('n_test'):,} test "
        f"x {details.get('n_features'):,} features",
    ]
    lines.append("Checks  :")
    for check in report.checks:
        label = "PASS" if check.passed else "FAIL"
        lines.append(f"  [{label}] {check.name}: {check.detail}")
    if report.warnings:
        lines.append("Warnings: " + "; ".join(report.warnings))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Split metadata persistence
# ---------------------------------------------------------------------------
def _sha256_indices(indices: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_split_metadata(
    result: PreprocessedDataset,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serialisable record of splits, seeds, and scaling scope.

    Unlabelled true labels are deliberately excluded; only index hashes are
    stored, which is sufficient for reproducibility without leaking labels.
    """
    settings: dict[str, Any] = {}
    for proportion, split in sorted(result.semi_supervised.items()):
        settings[f"{proportion:.2f}"] = {
            "requested_proportion": float(proportion),
            "actual_proportion": split.actual_proportion,
            "n_labelled": split.n_labelled,
            "n_unlabelled": split.n_unlabelled,
            "labelled_class_counts": _class_counts(split.y_labelled),
            "labelled_indices_sha256": _sha256_indices(split.labelled_indices),
            "unlabelled_indices_sha256": _sha256_indices(split.unlabelled_indices),
        }

    metadata: dict[str, Any] = {
        "dataset": result.name,
        "strategy": result.metadata.get("strategy"),
        "random_seed": int(result.metadata.get("random_seed")),
        "test_size": result.metadata.get("test_size"),
        "scaling": result.metadata.get("scaling"),
        "n_original": int(result.metadata.get("n_original")),
        "n_train": result.n_train,
        "n_test": result.n_test,
        "n_features": result.n_features,
        "train_class_counts": _class_counts(result.y_train),
        "test_class_counts": _class_counts(result.y_test),
        "train_indices_sha256": _sha256_indices(result.train_indices),
        "test_indices_sha256": _sha256_indices(result.test_indices),
        "labelled_settings": settings,
    }
    if result.train_subjects is not None:
        metadata["subject_info"] = {
            "train_subjects": sorted(
                int(value) for value in np.unique(result.train_subjects)
            ),
            "test_subjects": sorted(
                int(value) for value in np.unique(result.test_subjects)
            ),
        }
    if "competition_test_ignored" in result.metadata:
        metadata["competition_test_ignored"] = result.metadata[
            "competition_test_ignored"
        ]
    if "competition_test_present" in result.metadata:
        metadata["competition_test_present"] = result.metadata[
            "competition_test_present"
        ]

    metadata["requested_proportions"] = list(_configured_proportions(config))
    return metadata


def save_split_metadata(
    metadata_by_dataset: dict[str, dict[str, Any]],
    config: dict[str, Any],
    output_path: str | Path = DEFAULT_SPLIT_METADATA_PATH,
) -> Path:
    """Persist split metadata, seed, and scaling scope as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": config.get("project", {}).get("name"),
        "random_seed": int(config["project"]["random_seed"]),
        "test_size": float(config.get("data", {}).get("test_size", 0.20)),
        "labelled_proportions": list(_configured_proportions(config)),
        "scaling": config.get("data", {}).get("scaling", {"method": "standard"}),
        "datasets": metadata_by_dataset,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path
