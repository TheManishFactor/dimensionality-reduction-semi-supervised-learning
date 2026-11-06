"""Dimensionality reduction: baseline, PCA, and LDA.

Fitting rules
-------------
* **Baseline** returns the scaled Phase 3 feature matrices unchanged.
* **PCA** is unsupervised and is fitted on the complete scaled training pool
  (labelled + unlabelled training rows). It never sees test rows.
* **LDA** is supervised and is fitted separately for each labelled proportion
  on the labelled training rows and their labels only. It never sees
  unlabelled true labels or test rows/labels.

Results carry the transformed training and test matrices plus the fitted
transformer, so later phases can reuse the same projection without refitting.
"""

from __future__ import annotations

import numbers
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .preprocessing import PreprocessedDataset, SemiSupervisedSplit

SUPPORTED_METHODS: tuple[str, ...] = ("none", "pca", "lda")
METHOD_LABELS: dict[str, str] = {"none": "baseline", "pca": "pca", "lda": "lda"}
PCA_SOLVERS: tuple[str, ...] = ("auto", "full", "randomized", "arpack")
LDA_SOLVERS: tuple[str, ...] = ("svd", "lsqr", "eigen")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ReductionResult:
    """One fitted projection and its transformed matrices."""

    dataset: str
    method: str
    X_train: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    n_original_features: int
    n_components: int
    transformer: Any | None
    explained_variance_ratio: np.ndarray | None
    cumulative_explained_variance: np.ndarray | None
    fit_time_seconds: float
    transform_time_seconds: float
    fitted_on: str
    n_samples_fit: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method {self.method!r}; choose from {SUPPORTED_METHODS}."
            )
        self.X_train = np.asarray(self.X_train)
        self.X_test = np.asarray(self.X_test)
        if self.X_train.ndim != 2 or self.X_test.ndim != 2:
            raise ValueError(f"{self.dataset}/{self.method}: reduced matrices must be 2-D.")
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError(
                f"{self.dataset}/{self.method}: train and test feature widths differ."
            )
        if self.X_train.shape[1] != int(self.n_components):
            raise ValueError(
                f"{self.dataset}/{self.method}: reduced width does not match "
                f"n_components={self.n_components}."
            )
        if len(self.feature_names) != int(self.n_components):
            raise ValueError(
                f"{self.dataset}/{self.method}: feature-name count does not match "
                "the reduced width."
            )
        if int(self.n_original_features) < int(self.n_components):
            raise ValueError(
                f"{self.dataset}/{self.method}: reduced width exceeds the original "
                "feature count."
            )
        if not 0.0 <= float(self.fit_time_seconds):
            raise ValueError("fit_time_seconds must be non-negative.")
        if not 0.0 <= float(self.transform_time_seconds):
            raise ValueError("transform_time_seconds must be non-negative.")
        if int(self.n_samples_fit) < 0:
            raise ValueError("n_samples_fit must be non-negative.")

        if self.explained_variance_ratio is not None:
            self.explained_variance_ratio = np.asarray(
                self.explained_variance_ratio, dtype=np.float64
            )
            if self.explained_variance_ratio.ndim != 1:
                raise ValueError("explained_variance_ratio must be one-dimensional.")
            if self.explained_variance_ratio.size > int(self.n_components):
                raise ValueError(
                    "explained_variance_ratio is longer than the reduced width."
                )
        if self.cumulative_explained_variance is not None:
            self.cumulative_explained_variance = np.asarray(
                self.cumulative_explained_variance, dtype=np.float64
            )
            if self.cumulative_explained_variance.ndim != 1:
                raise ValueError(
                    "cumulative_explained_variance must be one-dimensional."
                )

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])

    @property
    def reduction_percent(self) -> float:
        """Percentage reduction in feature count relative to the original space."""
        if self.n_original_features == 0:
            return 0.0
        return 100.0 * (1.0 - self.n_components / self.n_original_features)

    @property
    def explained_variance_retained(self) -> float | None:
        """Variance retained (PCA) or discriminant variance (LDA).

        The baseline preserves the complete feature space, so it reports 1.0.
        LDA's value is the sum of scikit-learn's discriminant
        ``explained_variance_ratio_``; it is not a reconstruction target.
        """
        if self.method == "none":
            return 1.0
        if self.explained_variance_ratio is None:
            return None
        return float(np.sum(self.explained_variance_ratio))


@dataclass
class ReductionConfiguration:
    """One ``(dataset, method, labelled proportion)`` experiment condition."""

    dataset: str
    method: str
    labelled_proportion: float
    result: ReductionResult
    split: "SemiSupervisedSplit"

    def __post_init__(self) -> None:
        method = _normalise_method(self.method)
        proportion = float(self.labelled_proportion)
        if not 0.0 < proportion < 1.0:
            raise ValueError("labelled_proportion must be strictly between 0 and 1.")
        if self.result.method != method:
            raise ValueError("Configuration method and result method disagree.")
        if self.split.n_train != self.result.n_train:
            raise ValueError(
                "Configuration split and reduced training matrix have different sizes."
            )
        self.method = method
        self.labelled_proportion = proportion

    @property
    def method_label(self) -> str:
        return METHOD_LABELS[self.method]

    def get_labelled_features(self) -> np.ndarray:
        """Reduced features for the labelled rows of this condition."""
        return self.result.X_train[self.split.labelled_indices]

    def get_unlabelled_features(self) -> np.ndarray:
        """Reduced features for the unlabelled rows of this condition."""
        return self.result.X_train[self.split.unlabelled_indices]

    def statistics(self) -> dict[str, Any]:
        return calculate_reduction_statistics(self.result, self.labelled_proportion)


@dataclass
class DimensionalityReductionSuite:
    """All fitted projections for one dataset."""

    dataset: str
    baseline: ReductionResult | None
    pca: ReductionResult | None
    lda: dict[float, ReductionResult]
    configurations: list[ReductionConfiguration]

    def __post_init__(self) -> None:
        if not self.configurations:
            raise ValueError(f"{self.dataset}: no reduction configurations were built.")
        if any(config.dataset != self.dataset for config in self.configurations):
            raise ValueError(f"{self.dataset}: suite contains another dataset's results.")

    def configuration(
        self, method: str, labelled_proportion: float
    ) -> ReductionConfiguration:
        """Return the configuration for ``method`` and ``labelled_proportion``."""
        key = _normalise_method(method)
        proportion = float(labelled_proportion)
        for config in self.configurations:
            if config.method == key and config.labelled_proportion == proportion:
                return config
        raise KeyError(
            f"{self.dataset}: no configuration for method={method!r}, "
            f"proportion={proportion!r}."
        )


# ---------------------------------------------------------------------------
# Validation and helpers
# ---------------------------------------------------------------------------
def _normalise_method(method: str) -> str:
    key = str(method).strip().lower()
    if key == "baseline":
        key = "none"
    if key not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported reduction method {method!r}; choose from {SUPPORTED_METHODS}."
        )
    return key


def _validate_matrix(X: np.ndarray, name: str, min_rows: int = 2) -> np.ndarray:
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}.")
    if array.shape[0] < min_rows:
        raise ValueError(
            f"{name} must contain at least {min_rows} row(s), got {array.shape[0]}."
        )
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains missing or infinite values.")
    return array


def _validate_labels(y: np.ndarray, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(y)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if array.size != n_rows:
        raise ValueError(f"{name} has {array.size} labels for {n_rows} rows.")
    if array.dtype == object and any(value is None for value in array):
        raise ValueError(f"{name} contains missing labels.")
    return array


def _resolve_pca_settings(
    n_components: float | int,
    n_samples: int,
    n_features: int,
    svd_solver: str,
) -> tuple[float | int, str, str | None]:
    """Validate PCA settings and resolve a solver that supports the target."""
    solver = str(svd_solver).strip().lower()
    if solver not in PCA_SOLVERS:
        raise ValueError(
            f"Unsupported PCA svd_solver {svd_solver!r}; choose from {PCA_SOLVERS}."
        )
    if isinstance(n_components, bool) or not isinstance(
        n_components, numbers.Real
    ):
        raise TypeError(
            "PCA explained variance must be a float in (0, 1) or an integer "
            "component count."
        )

    minimum_dimension = min(int(n_samples), int(n_features))
    if isinstance(n_components, numbers.Integral):
        target = int(n_components)
        if not 1 <= target <= minimum_dimension:
            raise ValueError(
                f"PCA n_components={target} must be between 1 and "
                f"min(n_samples, n_features)={minimum_dimension}."
            )
        if solver in ("randomized", "arpack") and target >= minimum_dimension:
            raise ValueError(
                f"PCA n_components={target} must be strictly smaller than "
                f"min(n_samples, n_features)={minimum_dimension} for "
                f"svd_solver={solver!r}."
            )
        return target, solver, None

    target = float(n_components)
    if not 0.0 < target < 1.0:
        raise ValueError(
            "PCA explained variance must be strictly between 0 and 1. "
            "Use an integer n_components to request every component."
        )
    if solver in ("randomized", "arpack"):
        raise ValueError(
            f"PCA explained-variance target {target} is fractional and cannot be "
            f"supported by svd_solver={solver!r}. Use 'full' or 'auto' instead of "
            "silently changing the target."
        )
    note = None
    if solver == "auto":
        note = (
            "Resolved svd_solver='auto' to 'full' because fractional variance "
            "targets require the full solver; the target itself was unchanged."
        )
    return target, "full", note


def _transform_reducer(
    reducer: Any,
    X: np.ndarray,
    name: str,
    fitted_attribute: str,
) -> tuple[np.ndarray, float]:
    if not hasattr(reducer, fitted_attribute):
        raise ValueError(f"{name}: the reducer has not been fitted.")
    array = _validate_matrix(X, name, min_rows=1)
    expected_features = int(getattr(reducer, "n_features_in_", array.shape[1]))
    if array.shape[1] != expected_features:
        raise ValueError(
            f"{name} has {array.shape[1]} features but the reducer was fitted on "
            f"{expected_features}."
        )
    start = time.perf_counter()
    try:
        transformed = reducer.transform(array)
    except Exception as exc:  # Add context while preserving the cause.
        raise RuntimeError(f"{name}: transformation failed: {exc}") from exc
    elapsed = time.perf_counter() - start
    transformed = np.asarray(transformed)
    if transformed.ndim != 2 or not np.isfinite(transformed).all():
        raise RuntimeError(f"{name}: transformation produced invalid values.")
    return transformed, elapsed


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def apply_baseline(preprocessed: "PreprocessedDataset") -> ReductionResult:
    """Return the scaled Phase 3 features without any projection.

    The original feature names are preserved. The returned matrices are the
    Phase 3 arrays themselves (read-only by convention); callers must not
    mutate them.
    """
    X_train = _validate_matrix(preprocessed.X_train, f"{preprocessed.name} training features")
    X_test = _validate_matrix(preprocessed.X_test, f"{preprocessed.name} test features", min_rows=1)
    if X_train.shape[1] != preprocessed.n_features:
        raise ValueError(f"{preprocessed.name}: training feature count is inconsistent.")
    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError(f"{preprocessed.name}: train and test feature counts differ.")

    start = time.perf_counter()
    # Baseline performs no numerical transformation; the elapsed time measures
    # the validation and reference hand-off only.
    elapsed = time.perf_counter() - start
    return ReductionResult(
        dataset=preprocessed.name,
        method="none",
        X_train=X_train,
        X_test=X_test,
        feature_names=list(preprocessed.feature_names),
        n_original_features=preprocessed.n_features,
        n_components=preprocessed.n_features,
        transformer=None,
        explained_variance_ratio=None,
        cumulative_explained_variance=None,
        fit_time_seconds=0.0,
        transform_time_seconds=elapsed,
        fitted_on="not_applicable",
        n_samples_fit=0,
        metadata={
            "note": "Original scaled feature space; no reducer was fitted.",
            "n_train_rows": int(X_train.shape[0]),
            "n_test_rows": int(X_test.shape[0]),
        },
    )


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def fit_pca(
    X_train: np.ndarray,
    n_components: float | int = 0.95,
    whiten: bool = False,
    svd_solver: str = "auto",
    random_state: int = 42,
) -> tuple[PCA, float, dict[str, Any]]:
    """Fit PCA on the complete scaled training pool.

    Parameters
    ----------
    X_train : ndarray
        Complete training features (labelled + unlabelled). Test rows must not
        be included.
    n_components : float or int, default=0.95
        Fraction of variance to retain, or an exact component count.
    whiten : bool, default=False
        Passed to :class:`sklearn.decomposition.PCA`.
    svd_solver : str, default="auto"
        PCA solver. Fractional variance targets require ``"full"`` (or
        ``"auto"``, which is resolved to ``"full"`` and recorded).
    random_state : int, default=42
        Seed for randomized solvers.

    Returns
    -------
    pca, fit_time_seconds, info
        The fitted PCA object, the measured fit time, and settings metadata.
    """
    X = _validate_matrix(X_train, "PCA fit data", min_rows=2)
    target, resolved_solver, note = _resolve_pca_settings(
        n_components, X.shape[0], X.shape[1], svd_solver
    )

    pca = PCA(
        n_components=target,
        whiten=bool(whiten),
        svd_solver=resolved_solver,
        random_state=int(random_state),
    )
    start = time.perf_counter()
    try:
        pca.fit(X)
    except Exception as exc:
        raise RuntimeError(
            f"PCA fit failed for n_components={target!r}, "
            f"svd_solver={resolved_solver!r}, shape={X.shape}: {exc}"
        ) from exc
    fit_time = time.perf_counter() - start

    info = {
        "configured_n_components": target,
        "configured_solver": str(svd_solver).lower(),
        "resolved_solver": resolved_solver,
        "selected_components": int(pca.n_components_),
        "n_samples_fit": int(X.shape[0]),
        "n_features_fit": int(X.shape[1]),
        "whiten": bool(whiten),
        "random_state": int(random_state),
        "solver_note": note,
    }
    return pca, fit_time, info


def transform_pca(
    pca: PCA,
    X: np.ndarray,
    name: str = "features",
) -> tuple[np.ndarray, float]:
    """Transform features with an already fitted PCA object."""
    return _transform_reducer(pca, X, name, fitted_attribute="components_")


# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------
def fit_lda(
    X_labelled: np.ndarray,
    y_labelled: np.ndarray,
    n_components: int | None = None,
    solver: str = "svd",
    automatic_components: bool = True,
) -> tuple[LinearDiscriminantAnalysis, float, dict[str, Any]]:
    """Fit LDA on labelled training rows and labels only.

    The number of components is capped at ``n_classes - 1`` and at the number
    of features. Automatic selection uses that full cap. An explicit request
    above the cap raises an error instead of being silently truncated.
    """
    X = _validate_matrix(X_labelled, "LDA fit data", min_rows=2)
    y = _validate_labels(y_labelled, X.shape[0], "LDA labels")

    classes, counts = np.unique(y, return_counts=True)
    n_classes = int(classes.size)
    if n_classes < 2:
        raise ValueError("LDA requires at least two classes.")
    if X.shape[0] < n_classes:
        raise ValueError(
            f"LDA has {X.shape[0]} labelled rows for {n_classes} classes; at least "
            "one row per class is required."
        )

    max_components = min(X.shape[1], n_classes - 1)
    if max_components < 1:
        raise ValueError("LDA cannot produce a component from this data.")

    if automatic_components:
        if n_components is not None:
            raise ValueError(
                "automatic_components=true requires n_components=null; do not "
                "provide both."
            )
        selected = max_components
        selection = "automatic: min(n_features, n_classes - 1)"
    else:
        if n_components is None:
            raise ValueError(
                "automatic_components=false requires an explicit integer "
                "n_components."
            )
        if isinstance(n_components, bool) or not isinstance(
            n_components, numbers.Integral
        ):
            raise TypeError("LDA n_components must be an integer.")
        selected = int(n_components)
        if not 1 <= selected <= max_components:
            raise ValueError(
                f"LDA requested {selected} components but the maximum supported "
                f"here is {max_components} (n_classes={n_classes}, "
                f"n_features={X.shape[1]})."
            )
        selection = "explicit"

    solver_key = str(solver).strip().lower()
    if solver_key not in LDA_SOLVERS:
        raise ValueError(
            f"Unsupported LDA solver {solver!r}; choose from {LDA_SOLVERS}."
        )

    lda = LinearDiscriminantAnalysis(n_components=selected, solver=solver_key)
    start = time.perf_counter()
    try:
        lda.fit(X, y)
    except Exception as exc:
        raise RuntimeError(
            f"LDA fit failed for n_components={selected}, solver={solver_key!r}, "
            f"shape={X.shape}, n_classes={n_classes}: {exc}"
        ) from exc
    fit_time = time.perf_counter() - start

    actual_components = int(lda.transform(X[:1]).shape[1])
    if actual_components != selected:
        raise RuntimeError(
            f"LDA produced {actual_components} components but {selected} were "
            "requested."
        )

    explained_ratio = getattr(lda, "explained_variance_ratio_", None)
    if explained_ratio is not None:
        explained_ratio = np.asarray(explained_ratio, dtype=np.float64)
        if explained_ratio.size != actual_components:
            raise RuntimeError(
                "LDA explained_variance_ratio_ length does not match the number "
                "of transformed components."
            )

    info = {
        "n_classes": n_classes,
        "class_counts": {
            str(label): int(count) for label, count in zip(classes, counts)
        },
        "max_components": int(max_components),
        "selected_components": int(actual_components),
        "component_selection": selection,
        "solver": solver_key,
        "n_samples_fit": int(X.shape[0]),
        "n_features_fit": int(X.shape[1]),
    }
    return lda, fit_time, info


def transform_lda(
    lda: LinearDiscriminantAnalysis,
    X: np.ndarray,
    name: str = "features",
) -> tuple[np.ndarray, float]:
    """Transform features with an already fitted LDA object."""
    return _transform_reducer(lda, X, name, fitted_attribute="scalings_")


# ---------------------------------------------------------------------------
# Statistics and orchestration
# ---------------------------------------------------------------------------
def calculate_reduction_statistics(
    result: ReductionResult,
    labelled_proportion: float,
) -> dict[str, Any]:
    """Return a machine-readable statistics record for one configuration."""
    proportion = float(labelled_proportion)
    if not 0.0 < proportion < 1.0:
        raise ValueError("labelled_proportion must be strictly between 0 and 1.")

    variance_kind = {
        "none": "not_applicable",
        "pca": "pca_reconstruction",
        "lda": "lda_discriminant",
    }[result.method]
    return {
        "dataset": result.dataset,
        "method": result.method,
        "method_label": METHOD_LABELS[result.method],
        "labelled_proportion": proportion,
        "original_features": int(result.n_original_features),
        "reduced_features": int(result.n_components),
        "dimensionality_reduction_percent": float(result.reduction_percent),
        "explained_variance_retained": result.explained_variance_retained,
        "explained_variance_kind": variance_kind,
        "fit_time_seconds": float(result.fit_time_seconds),
        "transform_time_seconds": float(result.transform_time_seconds),
        "fitted_on": result.fitted_on,
        "n_samples_fit": int(result.n_samples_fit),
        "n_train_rows": int(result.n_train),
        "n_test_rows": int(result.n_test),
        "projection_reused_across_proportions": result.method in ("none", "pca"),
        "solver": result.metadata.get("resolved_solver")
        or result.metadata.get("solver"),
        "selected_components": result.metadata.get("selected_components"),
        "max_components": result.metadata.get("max_components"),
    }


def _enabled_methods(config: dict[str, Any]) -> list[str]:
    dr_config = config.get("dimensionality_reduction", {})
    configured = dr_config.get("methods", SUPPORTED_METHODS)
    methods: list[str] = []
    for method in configured:
        key = _normalise_method(method)
        if key not in methods:
            methods.append(key)
    if not methods:
        raise ValueError("dimensionality_reduction.methods is empty.")

    enabled: list[str] = []
    for method in methods:
        section = "baseline" if method == "none" else method
        settings = dr_config.get(section, {}) or {}
        if bool(settings.get("enabled", True)):
            enabled.append(method)
    if not enabled:
        raise ValueError("All configured dimensionality-reduction methods are disabled.")
    return enabled


def run_dimensionality_reduction(
    preprocessed: "PreprocessedDataset",
    config: dict[str, Any],
) -> DimensionalityReductionSuite:
    """Fit baseline/PCA/LDA and build all 9 configurations for one dataset.

    PCA is fitted once on the complete training pool and reused across
    labelled proportions. LDA is fitted separately for every proportion using
    only that proportion's labelled rows and labels.
    """
    dataset = preprocessed.name
    methods = _enabled_methods(config)
    dr_config = config.get("dimensionality_reduction", {})
    proportions = sorted(float(value) for value in preprocessed.semi_supervised)
    if not proportions:
        raise ValueError(f"{dataset}: no labelled proportions are available.")

    baseline_result: ReductionResult | None = None
    pca_result: ReductionResult | None = None
    lda_results: dict[float, ReductionResult] = {}

    if "none" in methods:
        baseline_result = apply_baseline(preprocessed)

    if "pca" in methods:
        pca_config = dr_config.get("pca", {}) or {}
        target = pca_config.get(
            "explained_variance", pca_config.get("n_components", 0.95)
        )
        random_state = int(
            pca_config.get("random_state", config["project"]["random_seed"])
        )
        try:
            pca, fit_time, info = fit_pca(
                preprocessed.X_train,
                n_components=target,
                whiten=bool(pca_config.get("whiten", False)),
                svd_solver=str(pca_config.get("svd_solver", "auto")),
                random_state=random_state,
            )
        except Exception as exc:
            raise RuntimeError(f"PCA failed for dataset {dataset!r}: {exc}") from exc

        try:
            X_train_pca, train_transform_time = transform_pca(
                pca, preprocessed.X_train, f"{dataset} complete training pool"
            )
            X_test_pca, test_transform_time = transform_pca(
                pca, preprocessed.X_test, f"{dataset} test features"
            )
        except Exception as exc:
            raise RuntimeError(
                f"PCA transformation failed for dataset {dataset!r}: {exc}"
            ) from exc

        ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        pca_result = ReductionResult(
            dataset=dataset,
            method="pca",
            X_train=X_train_pca,
            X_test=X_test_pca,
            feature_names=[f"PC{index + 1}" for index in range(X_train_pca.shape[1])],
            n_original_features=preprocessed.n_features,
            n_components=int(X_train_pca.shape[1]),
            transformer=pca,
            explained_variance_ratio=ratios,
            cumulative_explained_variance=np.cumsum(ratios),
            fit_time_seconds=fit_time,
            transform_time_seconds=train_transform_time + test_transform_time,
            fitted_on="complete_training_pool",
            n_samples_fit=preprocessed.n_train,
            metadata={
                **info,
                "fit_scope": "complete_training_pool",
                "n_train_rows": preprocessed.n_train,
                "n_test_rows": preprocessed.n_test,
                "train_transform_time_seconds": train_transform_time,
                "test_transform_time_seconds": test_transform_time,
            },
        )

    if "lda" in methods:
        lda_config = dr_config.get("lda", {}) or {}
        for proportion in proportions:
            split = preprocessed.semi_supervised[proportion]
            X_labelled = preprocessed.X_train[split.labelled_indices]
            n_classes = int(np.unique(split.y_labelled).size)
            try:
                lda, fit_time, info = fit_lda(
                    X_labelled,
                    split.y_labelled,
                    n_components=lda_config.get("n_components"),
                    solver=str(lda_config.get("solver", "svd")),
                    automatic_components=bool(
                        lda_config.get("automatic_components", True)
                    ),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"LDA failed for dataset {dataset!r}, labelled proportion "
                    f"{proportion:.2f} ({split.n_labelled:,} labelled rows, "
                    f"{n_classes} classes): {exc}"
                ) from exc

            try:
                X_labelled_reduced, labelled_time = transform_lda(
                    lda, X_labelled, f"{dataset} labelled training rows"
                )
                X_unlabelled_reduced, unlabelled_time = transform_lda(
                    lda,
                    preprocessed.X_train[split.unlabelled_indices],
                    f"{dataset} unlabelled training rows",
                )
                X_test_reduced, test_time = transform_lda(
                    lda, preprocessed.X_test, f"{dataset} test features"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"LDA transformation failed for dataset {dataset!r}, "
                    f"proportion {proportion:.2f}: {exc}"
                ) from exc

            n_components = int(X_labelled_reduced.shape[1])
            X_train_reduced = np.empty(
                (preprocessed.n_train, n_components), dtype=X_labelled_reduced.dtype
            )
            X_train_reduced[split.labelled_indices] = X_labelled_reduced
            X_train_reduced[split.unlabelled_indices] = X_unlabelled_reduced

            ratios = getattr(lda, "explained_variance_ratio_", None)
            if ratios is not None:
                ratios = np.asarray(ratios, dtype=np.float64)

            lda_results[proportion] = ReductionResult(
                dataset=dataset,
                method="lda",
                X_train=X_train_reduced,
                X_test=X_test_reduced,
                feature_names=[f"LD{index + 1}" for index in range(n_components)],
                n_original_features=preprocessed.n_features,
                n_components=n_components,
                transformer=lda,
                explained_variance_ratio=ratios,
                cumulative_explained_variance=(
                    np.cumsum(ratios) if ratios is not None else None
                ),
                fit_time_seconds=fit_time,
                transform_time_seconds=(
                    labelled_time + unlabelled_time + test_time
                ),
                fitted_on=f"labelled_training_subset_{proportion:.2f}",
                n_samples_fit=split.n_labelled,
                metadata={
                    **info,
                    "fit_scope": "labelled_training_subset",
                    "labelled_proportion": proportion,
                    "n_train_rows": preprocessed.n_train,
                    "n_test_rows": preprocessed.n_test,
                    "n_labelled": split.n_labelled,
                    "n_unlabelled": split.n_unlabelled,
                    "labelled_transform_time_seconds": labelled_time,
                    "unlabelled_transform_time_seconds": unlabelled_time,
                    "test_transform_time_seconds": test_time,
                },
            )

    configurations: list[ReductionConfiguration] = []
    for proportion in proportions:
        split = preprocessed.semi_supervised[proportion]
        if baseline_result is not None:
            configurations.append(
                ReductionConfiguration(
                    dataset=dataset,
                    method="none",
                    labelled_proportion=proportion,
                    result=baseline_result,
                    split=split,
                )
            )
        if pca_result is not None:
            configurations.append(
                ReductionConfiguration(
                    dataset=dataset,
                    method="pca",
                    labelled_proportion=proportion,
                    result=pca_result,
                    split=split,
                )
            )
        if proportion in lda_results:
            configurations.append(
                ReductionConfiguration(
                    dataset=dataset,
                    method="lda",
                    labelled_proportion=proportion,
                    result=lda_results[proportion],
                    split=split,
                )
            )

    return DimensionalityReductionSuite(
        dataset=dataset,
        baseline=baseline_result,
        pca=pca_result,
        lda=lda_results,
        configurations=configurations,
    )
