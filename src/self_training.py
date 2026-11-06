"""Confidence-based self-training for semi-supervised classification.

Self-training alternates between fitting a classifier on labelled data and
adding confidently predicted unlabelled samples to the training set:

1. Fit the classifier on the original labelled rows.
2. Predict class probabilities for the remaining unlabelled pool.
3. Accept samples whose maximum predicted probability is at least the
   configured confidence threshold.
4. Assign the predicted classes as pseudo-labels, append the accepted rows to
   the training set, and remove them from the unlabelled pool.
5. Refit on the original labelled rows plus all accumulated pseudo-labelled
   rows and repeat.

The stopping conditions are: no samples meet the threshold, fewer than the
minimum number of new pseudo-labels are accepted, the iteration limit is
reached, or the unlabelled pool is exhausted.

Leakage guarantees
------------------
* This module never receives the true labels of unlabelled samples. It only
  receives ``X_unlabelled``.
* Accepted positions always refer to the supplied unlabelled pool, are unique,
  and are removed from the pool once accepted.
* Pseudo-labels are stored separately from the original labels.
* No test data is accepted by any function in this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

PSEUDO_LABEL_SOURCE = "classifier_prediction"


@dataclass(frozen=True)
class SelfTrainingConfig:
    """Settings shared by every Phase 6 self-training experiment."""

    confidence_threshold: float = 0.90
    max_iterations: int = 10
    minimum_new_pseudo_labels: int = 1
    random_state: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < float(self.confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1].")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be at least 1.")
        if int(self.minimum_new_pseudo_labels) < 1:
            raise ValueError("minimum_new_pseudo_labels must be at least 1.")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "SelfTrainingConfig":
        """Build a configuration from the ``self_training`` YAML section."""
        values = dict(mapping or {})
        minimum = values.get(
            "minimum_new_pseudo_labels",
            values.get("min_new_samples", 1),
        )
        return cls(
            confidence_threshold=float(values.get("confidence_threshold", 0.90)),
            max_iterations=int(values.get("max_iterations", 10)),
            minimum_new_pseudo_labels=int(minimum),
            random_state=int(values.get("random_state", 42)),
        )


@dataclass
class IterationRecord:
    """One completed accept-and-retrain round."""

    iteration: int
    n_training_samples: int
    n_unlabelled_considered: int
    n_accepted: int
    cumulative_pseudo_labels: int
    remaining_unlabelled: int
    mean_accepted_confidence: float | None
    min_accepted_confidence: float | None
    max_accepted_confidence: float | None

    def to_row(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "pseudo_labels_added": self.n_accepted,
            "cumulative_pseudo_labels": self.cumulative_pseudo_labels,
            "remaining_unlabelled": self.remaining_unlabelled,
            "mean_accepted_confidence": self.mean_accepted_confidence,
            "min_accepted_confidence": self.min_accepted_confidence,
            "max_accepted_confidence": self.max_accepted_confidence,
            "training_set_size": self.n_training_samples,
            "unlabelled_considered": self.n_unlabelled_considered,
        }


@dataclass
class SelfTrainingResult:
    """Full record of one self-training run."""

    final_classifier: Any
    iterations: list[IterationRecord] = field(default_factory=list)
    pseudo_positions: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    pseudo_labels: np.ndarray = field(default_factory=lambda: np.empty(0))
    pseudo_confidences: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    pseudo_iterations: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    remaining_positions: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    stopping_reason: str = ""
    confidence_threshold: float = 0.90
    confidence_method: str = ""
    n_labelled_original: int = 0
    n_unlabelled_original: int = 0
    training_time_seconds: float = 0.0
    final_model_description: str = ""

    @property
    def n_pseudo_labelled(self) -> int:
        return int(self.pseudo_positions.size)

    @property
    def n_remaining(self) -> int:
        return int(self.remaining_positions.size)

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    @property
    def acceptance_rate(self) -> float:
        if self.n_unlabelled_original == 0:
            return 0.0
        return self.n_pseudo_labelled / self.n_unlabelled_original


def _validate_inputs(
    X_labelled: np.ndarray,
    y_labelled: np.ndarray,
    X_unlabelled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labelled = np.asarray(X_labelled)
    targets = np.asarray(y_labelled)
    unlabelled = np.asarray(X_unlabelled)

    if labelled.ndim != 2 or unlabelled.ndim != 2:
        raise ValueError("Labelled and unlabelled features must be 2-D.")
    if labelled.shape[0] != targets.size:
        raise ValueError(
            f"Labelled features have {labelled.shape[0]} rows but targets have "
            f"{targets.size} entries."
        )
    if labelled.shape[1] != unlabelled.shape[1]:
        raise ValueError(
            "Labelled and unlabelled feature widths differ: "
            f"{labelled.shape[1]} vs {unlabelled.shape[1]}."
        )
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("Labelled targets must be a non-empty 1-D array.")
    if unlabelled.shape[0] == 0:
        raise ValueError("The unlabelled pool must contain at least one row.")
    if np.unique(targets).size < 2:
        raise ValueError("At least two classes are required for self-training.")
    if not np.isfinite(labelled).all() or not np.isfinite(unlabelled).all():
        raise ValueError("Feature matrices contain non-finite values.")
    return labelled, targets, unlabelled


def _probability_matrix(model: Any, features: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"Classifier {type(model).__name__} does not provide predict_proba; "
            "confidence-based self-training requires class probabilities."
        )
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != features.shape[0]:
        raise RuntimeError(
            "predict_proba returned an unexpected shape: "
            f"{probabilities.shape} for {features.shape[0]} rows."
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("predict_proba returned non-finite probabilities.")
    if probabilities.min() < -1e-9 or probabilities.max() > 1.0 + 1e-9:
        raise RuntimeError("predict_proba returned values outside [0, 1].")
    return probabilities


def run_self_training(
    model_factory: Callable[[], Any],
    X_labelled: np.ndarray,
    y_labelled: np.ndarray,
    X_unlabelled: np.ndarray,
    config: SelfTrainingConfig | None = None,
    confidence_method: str = "predict_proba (maximum class probability)",
    final_model_factory: Callable[[], Any] | None = None,
    final_model_description: str = "",
) -> SelfTrainingResult:
    """Run iterative confidence-based self-training.

    Parameters
    ----------
    model_factory : callable
        Returns an unfitted classifier implementing ``fit``, ``predict``,
        ``predict_proba``, and ``classes_``. A fresh instance is created for
        the initial fit, every retraining round, and (optionally) the final
        model.
    X_labelled, y_labelled : ndarray
        Original labelled training rows and their true labels only.
    X_unlabelled : ndarray
        Unlabelled training features. Their true labels are never accepted by
        this function.
    config : SelfTrainingConfig, optional
        Threshold, iteration limit, minimum acceptance, and seed.
    confidence_method : str
        Human-readable description stored with the result.
    final_model_factory : callable, optional
        When supplied, a fresh model is fitted on the original labelled rows
        plus all accepted pseudo-labelled rows after the loop. This is used for
        the linear SVM so test predictions come from the standard Phase 5
        ``LinearSVC`` fitted on the complete augmented training set.
    final_model_description : str
        Description of the final estimator stored with the result.
    """
    settings = config or SelfTrainingConfig()
    labelled, targets, unlabelled = _validate_inputs(
        X_labelled, y_labelled, X_unlabelled
    )
    n_unlabelled = int(unlabelled.shape[0])

    remaining = np.arange(n_unlabelled, dtype=np.int64)
    accepted_positions = np.empty(0, dtype=np.int64)
    accepted_labels = np.empty(0, dtype=targets.dtype)
    accepted_confidences = np.empty(0, dtype=float)
    accepted_iterations = np.empty(0, dtype=np.int64)
    iterations: list[IterationRecord] = []

    start = time.perf_counter()
    model = model_factory()
    model.fit(labelled, targets)

    stopping_reason = "max_iterations_reached"
    for iteration in range(1, settings.max_iterations + 1):
        if remaining.size == 0:
            stopping_reason = "no_unlabelled_samples_remaining"
            break

        considered = int(remaining.size)
        probabilities = _probability_matrix(model, unlabelled[remaining])
        best_columns = np.argmax(probabilities, axis=1)
        confidences = probabilities[np.arange(considered), best_columns]
        classes = np.asarray(getattr(model, "classes_", np.unique(targets)))
        predicted = classes[best_columns]

        accepted_mask = confidences >= float(settings.confidence_threshold)
        n_accepted = int(accepted_mask.sum())
        if n_accepted < settings.minimum_new_pseudo_labels:
            stopping_reason = (
                "no_samples_meeting_confidence_threshold"
                if n_accepted == 0
                else "minimum_new_pseudo_labels_not_met"
            )
            break

        new_positions = remaining[accepted_mask]
        if np.intersect1d(new_positions, accepted_positions).size:
            raise RuntimeError(
                "A sample was selected for pseudo-labelling more than once."
            )
        new_labels = predicted[accepted_mask]
        new_confidences = confidences[accepted_mask]

        accepted_positions = np.concatenate([accepted_positions, new_positions])
        accepted_labels = np.concatenate([accepted_labels, new_labels])
        accepted_confidences = np.concatenate(
            [accepted_confidences, new_confidences]
        )
        accepted_iterations = np.concatenate(
            [
                accepted_iterations,
                np.full(n_accepted, iteration, dtype=np.int64),
            ]
        )
        remaining = remaining[~accepted_mask]

        augmented_features = np.vstack(
            [labelled, unlabelled[accepted_positions]]
        )
        augmented_labels = np.concatenate([targets, accepted_labels])
        # The calibrated SVM model is only a confidence scorer; when it will
        # never score again it is not refitted, because the final estimator is
        # fitted separately on the same augmented rows below.
        if final_model_factory is None or (
            iteration < settings.max_iterations and remaining.size > 0
        ):
            model = model_factory()
            model.fit(augmented_features, augmented_labels)

        iterations.append(
            IterationRecord(
                iteration=iteration,
                n_training_samples=int(augmented_features.shape[0]),
                n_unlabelled_considered=considered,
                n_accepted=n_accepted,
                cumulative_pseudo_labels=int(accepted_positions.size),
                remaining_unlabelled=int(remaining.size),
                mean_accepted_confidence=float(np.mean(new_confidences)),
                min_accepted_confidence=float(np.min(new_confidences)),
                max_accepted_confidence=float(np.max(new_confidences)),
            )
        )
    else:
        if remaining.size == 0:
            stopping_reason = "no_unlabelled_samples_remaining"

    if final_model_factory is not None:
        final_features = np.vstack(
            [labelled, unlabelled[accepted_positions]]
        )
        final_labels = np.concatenate([targets, accepted_labels])
        final_classifier = final_model_factory()
        final_classifier.fit(final_features, final_labels)
    else:
        final_classifier = model
    training_time = time.perf_counter() - start

    return SelfTrainingResult(
        final_classifier=final_classifier,
        iterations=iterations,
        pseudo_positions=accepted_positions,
        pseudo_labels=accepted_labels,
        pseudo_confidences=accepted_confidences,
        pseudo_iterations=accepted_iterations,
        remaining_positions=remaining,
        stopping_reason=stopping_reason,
        confidence_threshold=float(settings.confidence_threshold),
        confidence_method=confidence_method,
        n_labelled_original=int(labelled.shape[0]),
        n_unlabelled_original=n_unlabelled,
        training_time_seconds=float(training_time),
        final_model_description=final_model_description,
    )


class SelfTrainer:
    """Object-oriented wrapper around :func:`run_self_training`.

    ``classifier`` may be an unfitted estimator (cloned for every fit) or a
    zero-argument factory returning a fresh estimator.
    """

    def __init__(
        self,
        classifier: Any,
        confidence_threshold: float = 0.90,
        max_iterations: int = 10,
        minimum_new_pseudo_labels: int = 1,
        random_state: int = 42,
    ) -> None:
        self.config = SelfTrainingConfig(
            confidence_threshold=confidence_threshold,
            max_iterations=max_iterations,
            minimum_new_pseudo_labels=minimum_new_pseudo_labels,
            random_state=random_state,
        )
        if callable(classifier) and not hasattr(classifier, "fit"):
            self._factory = classifier
        else:
            from sklearn.base import clone

            self._factory = lambda: clone(classifier)
        self.result_: SelfTrainingResult | None = None

    def fit(
        self,
        X_labelled: np.ndarray,
        y_labelled: np.ndarray,
        X_unlabelled: np.ndarray,
    ) -> "SelfTrainer":
        self.result_ = run_self_training(
            self._factory,
            X_labelled,
            y_labelled,
            X_unlabelled,
            config=self.config,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("SelfTrainer must be fitted before prediction.")
        return np.asarray(self.result_.final_classifier.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("SelfTrainer must be fitted before prediction.")
        return np.asarray(self.result_.final_classifier.predict_proba(X))
