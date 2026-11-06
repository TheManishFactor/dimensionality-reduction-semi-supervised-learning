"""Supervised classifier construction from configuration.

Phase 5 classifiers
-------------------
* **Logistic Regression** — ``LogisticRegression`` with the lbfgs solver,
  ``max_iter=2000``, and a fixed random seed. The solver handles binary and
  multiclass targets.
* **Support Vector Machine** — ``LinearSVC`` (a *linear* SVM). It is chosen over
  the RBF-kernel ``SVC`` because it scales to the 33,600 x 784 MNIST training
  partition. LinearSVC does not expose ``predict_proba``; Phase 5 only needs
  hard predictions. A probability-capable SVM would be required for the
  confidence-based self-training planned for Phase 6.
* **K-Nearest Neighbors** — ``KNeighborsClassifier`` with 5 neighbours,
  distance weighting, and Euclidean (Minkowski p=2) distance. No tuning is
  performed in Phase 5.

Each estimator is constructed by its own builder so that configuration,
defaults, and any estimator-specific rules stay in one place.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

CLASSIFIER_NAMES: tuple[str, ...] = (
    "logistic_regression",
    "svm",
    "knn",
)
CLASSIFIER_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM (linear)",
    "knn": "KNN",
}


def list_classifiers() -> list[str]:
    """Return the configuration keys of all supported classifiers."""
    return list(CLASSIFIER_NAMES)


def build_logistic_regression(
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> LogisticRegression:
    """Construct an unfitted multinomial-capable logistic regression model."""
    options = dict(params or {})
    options.setdefault("solver", "lbfgs")
    options.setdefault("max_iter", 2000)
    options["random_state"] = int(random_state)
    return LogisticRegression(**options)


def build_svm(
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> LinearSVC:
    """Construct an unfitted linear support vector classifier.

    This study deliberately uses a **linear** SVM. ``LinearSVC`` is less
    expensive than an RBF-kernel ``SVC`` on the MNIST feature space and keeps
    the 81-experiment grid tractable. It produces hard labels only.

    The dual liblinear solver is used because it was about five times faster
    than the primal solver on the MNIST fit (21.5s vs 111.6s for the 10% subset),
    with a 2,000-iteration convergence budget.
    """
    options = dict(params or {})
    options.setdefault("loss", "squared_hinge")
    options.setdefault("dual", True)
    options.setdefault("tol", 1e-3)
    options.setdefault("max_iter", 2000)
    options["random_state"] = int(random_state)
    return LinearSVC(**options)


def build_knn(params: dict[str, Any] | None = None) -> KNeighborsClassifier:
    """Construct an unfitted K-nearest-neighbours classifier."""
    options = dict(params or {})
    options.setdefault("n_neighbors", 5)
    options.setdefault("weights", "distance")
    options.setdefault("metric", "minkowski")
    options.setdefault("p", 2)
    # Brute force is the predictable choice in high-dimensional feature spaces.
    options.setdefault("algorithm", "brute")
    options.setdefault("n_jobs", -1)
    return KNeighborsClassifier(**options)


class CalibratedLinearSVM(ClassifierMixin, BaseEstimator):
    """Linear SVM with Platt-scaled probabilities for self-training.

    Phase 5 uses ``LinearSVC`` with hard labels only. Self-training needs a
    confidence score, so this wrapper keeps the same underlying linear SVM and
    adds sigmoid (Platt) calibration:

    1. A class-stratified calibration subset is held out from the **original
       labelled rows** (the first ``n_original_labelled`` rows of the fitting
       matrix). Pseudo-labelled rows are always appended after those rows and
       are therefore never used for calibration.
    2. The base ``LinearSVC`` is fitted on the remaining labelled rows plus any
       pseudo-labelled rows.
    3. ``CalibratedClassifierCV`` with a frozen base estimator fits sigmoid
       calibrators on the calibration subset using ``StratifiedKFold`` folds,
       so every fold preserves class representation.

    The calibration subset is selected with a fixed seed from rows that never
    change between self-training iterations, which keeps the calibration
    reference stable. See ``docs``-level notes in the Phase 6 report for the
    caveats of using a held-out subset and calibrated probabilities.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        n_original_labelled: int | None = None,
        calibration_fraction: float = 0.20,
        min_calibration_per_class: int = 2,
        max_calibration_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        if calibration_fraction <= 0.0 or calibration_fraction >= 1.0:
            raise ValueError("calibration_fraction must be in (0, 1).")
        if min_calibration_per_class < 2:
            raise ValueError("min_calibration_per_class must be at least 2.")
        if max_calibration_splits < 2:
            raise ValueError("max_calibration_splits must be at least 2.")
        self.params = dict(params or {})
        self.n_original_labelled = n_original_labelled
        self.calibration_fraction = float(calibration_fraction)
        self.min_calibration_per_class = int(min_calibration_per_class)
        self.max_calibration_splits = int(max_calibration_splits)
        self.random_state = int(random_state)

    def _calibration_indices(self, y: np.ndarray) -> np.ndarray:
        """Choose a stratified calibration subset from the original labelled rows."""
        n_available = int(y.size) if self.n_original_labelled is None else int(
            self.n_original_labelled
        )
        if n_available < 2 or n_available > int(y.size):
            raise ValueError(
                "n_original_labelled must be between 2 and the number of fitting "
                f"rows; got {n_available} for {y.size} rows."
            )
        y_original = np.asarray(y[:n_available])
        classes = np.unique(y_original)
        if classes.size < 2:
            raise ValueError(
                "Calibration requires at least two classes in the original "
                f"labelled rows; found {classes.tolist()}."
            )

        rng = np.random.RandomState(self.random_state)
        selected: list[int] = []
        for label in classes:
            class_rows = np.flatnonzero(y_original == label)
            if class_rows.size < self.min_calibration_per_class + 1:
                raise ValueError(
                    f"Class {label!r} has {class_rows.size} labelled row(s); at "
                    "least three are required so that two rows can be used for "
                    "calibration and one row remains for SVM fitting."
                )
            n_calibration = max(
                self.min_calibration_per_class,
                int(round(class_rows.size * self.calibration_fraction)),
            )
            n_calibration = min(n_calibration, class_rows.size - 1)
            selected.extend(
                int(index)
                for index in rng.choice(
                    class_rows, size=n_calibration, replace=False
                )
            )
        return np.sort(np.asarray(selected, dtype=np.int64))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibratedLinearSVM":
        features = np.asarray(X)
        labels = np.asarray(y)
        if features.ndim != 2 or labels.ndim != 1:
            raise ValueError("X must be 2-D and y must be 1-D.")
        if features.shape[0] != labels.size:
            raise ValueError(
                f"X has {features.shape[0]} rows but y has {labels.size} labels."
            )

        calibration_indices = self._calibration_indices(labels)
        fit_mask = np.ones(labels.size, dtype=bool)
        fit_mask[calibration_indices] = False

        self.base_estimator_ = build_svm(self.params, self.random_state)
        self.base_estimator_.fit(features[fit_mask], labels[fit_mask])

        calibration_labels = labels[calibration_indices]
        class_counts = np.unique(calibration_labels, return_counts=True)[1]
        n_splits = min(self.max_calibration_splits, int(class_counts.min()))
        if n_splits < 2:
            raise ValueError(
                "At least two calibration rows per class are required for "
                "stratified sigmoid calibration."
            )
        folds = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        self.calibrated_estimator_ = CalibratedClassifierCV(
            estimator=FrozenEstimator(self.base_estimator_),
            method="sigmoid",
            cv=folds,
        )
        self.calibrated_estimator_.fit(
            features[calibration_indices], calibration_labels
        )

        self.classes_ = self.calibrated_estimator_.classes_
        self.n_features_in_ = features.shape[1]
        self.calibration_indices_ = calibration_indices
        self.n_calibration_ = int(calibration_indices.size)
        self.n_calibration_splits_ = n_splits
        self.n_fit_rows_ = int(fit_mask.sum())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.calibrated_estimator_.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.calibrated_estimator_.predict_proba(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.calibrated_estimator_.decision_function(X))


def build_calibrated_svm(
    params: dict[str, Any] | None = None,
    n_original_labelled: int | None = None,
    calibration_fraction: float = 0.20,
    min_calibration_per_class: int = 2,
    max_calibration_splits: int = 5,
    random_state: int = 42,
) -> CalibratedLinearSVM:
    """Construct an unfitted probability-capable linear SVM."""
    return CalibratedLinearSVM(
        params=params,
        n_original_labelled=n_original_labelled,
        calibration_fraction=calibration_fraction,
        min_calibration_per_class=min_calibration_per_class,
        max_calibration_splits=max_calibration_splits,
        random_state=random_state,
    )


def build_classifier(
    name: str,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> Any:
    """Construct an unfitted classifier from its configuration key.

    Parameters
    ----------
    name : str
        One of :data:`CLASSIFIER_NAMES`.
    params : dict, optional
        Estimator parameters from ``config.yaml``.
    random_state : int, default=42
        Seed applied where the estimator supports it.
    """
    if name == "logistic_regression":
        return build_logistic_regression(params, random_state=random_state)
    if name == "svm":
        return build_svm(params, random_state=random_state)
    if name == "knn":
        return build_knn(params)
    supported = ", ".join(CLASSIFIER_NAMES)
    raise ValueError(f"Unknown classifier {name!r}. Supported: {supported}.")


def build_all_classifiers(
    classifiers_config: dict[str, dict[str, Any]],
    random_state: int = 42,
) -> dict[str, Any]:
    """Construct every supported classifier from a configuration mapping."""
    return {
        name: build_classifier(
            name,
            classifiers_config.get(name, {}),
            random_state=random_state,
        )
        for name in CLASSIFIER_NAMES
    }
