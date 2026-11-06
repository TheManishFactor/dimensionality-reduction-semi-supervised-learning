"""Visualization helpers for dimensionality-reduction and experiment analysis.

Phase 4 implements the figures required to inspect baseline, PCA, and LDA
configurations:

1. Original versus reduced feature dimensions.
2. PCA cumulative explained variance.
3. PCA component comparison across datasets.
4. LDA component comparison across datasets and labelled proportions.
5. Dimensionality-reduction percentages.

Every value shown is read from measured reduction statistics; no metrics are
invented. Figures are written under ``results/dimensionality_reduction/figures``
by the Phase 4 verification script.

Phase 5 adds classifier-result figures (accuracy, macro F1, classifier
comparison, labelled-percentage trends, and training time) generated from the
measured experiment table.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.patches import Patch

if TYPE_CHECKING:  # pragma: no cover - typing only
    import matplotlib.figure
    import pandas as pd

    from .dimensionality_reduction import ReductionResult

DEFAULT_DPI: int = 300
DEFAULT_FIGSIZE: tuple[float, float] = (8.0, 6.0)

DATASET_ORDER: tuple[str, ...] = ("breast_cancer", "mnist", "har")
DATASET_LABELS: dict[str, str] = {
    "breast_cancer": "Breast Cancer",
    "mnist": "MNIST",
    "har": "HAR",
}
METHOD_ORDER: tuple[str, ...] = ("none", "pca", "lda")
METHOD_LABELS: dict[str, str] = {"none": "Baseline", "pca": "PCA", "lda": "LDA"}
METHOD_COLORS: dict[str, str] = {
    "none": "#4C72B0",
    "pca": "#DD8452",
    "lda": "#55A868",
}
ORIGINAL_COLOR: str = "#8C8C8C"

STATISTICS_COLUMNS: tuple[str, ...] = (
    "dataset",
    "method",
    "labelled_proportion",
    "original_features",
    "reduced_features",
    "dimensionality_reduction_percent",
)


def _dataset_label(name: str) -> str:
    return DATASET_LABELS.get(name, str(name).replace("_", " ").title())


def _require_columns(statistics: "pd.DataFrame", columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in statistics.columns]
    if missing:
        raise ValueError(
            "Reduction statistics are missing required column(s): "
            + ", ".join(missing)
        )
    if statistics.empty:
        raise ValueError("Reduction statistics are empty.")


def _ordered_datasets(values: Sequence[str]) -> list[str]:
    present = set(values)
    ordered = [name for name in DATASET_ORDER if name in present]
    ordered.extend(sorted(present.difference(DATASET_ORDER)))
    return ordered


def _annotate_bars(axis: Any, bars: Any, values: Sequence[int], suffix: str = "") -> None:
    """Write integer values next to horizontal or vertical bars."""
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{int(value):,}{suffix}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )


def save_figure(
    figure: "matplotlib.figure.Figure",
    output_path: str | Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Save a figure to disk and close it to release memory."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Phase 4 reduction figures
# ---------------------------------------------------------------------------
def plot_original_vs_reduced_dimensions(
    statistics: "pd.DataFrame",
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Horizontal bars comparing original and reduced feature counts.

    ``statistics`` is the table produced by one
    :func:`src.dimensionality_reduction.calculate_reduction_statistics` record
    per configuration. Proportions are collapsed because the feature width is
    identical across them for a given projection.
    """
    _require_columns(statistics, STATISTICS_COLUMNS)
    records = statistics.drop_duplicates(subset=["dataset", "method"]).copy()
    records["_dataset_order"] = records["dataset"].map(
        {name: index for index, name in enumerate(DATASET_ORDER)}
    )
    records["_method_order"] = records["method"].map(
        {name: index for index, name in enumerate(METHOD_ORDER)}
    )
    records = records.sort_values(["_dataset_order", "_method_order"])
    if records.empty:
        raise ValueError("No reduction statistics are available to plot.")

    labels = [
        f"{_dataset_label(row.dataset)} \u00b7 {METHOD_LABELS.get(row.method, row.method)}"
        for row in records.itertuples()
    ]
    original = records["original_features"].to_numpy(dtype=float)
    reduced = records["reduced_features"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    positions = np.arange(len(records))
    height = 0.38
    original_bars = axis.barh(
        positions - height / 2,
        original,
        height=height,
        color=ORIGINAL_COLOR,
        label="Original features",
    )
    reduced_bars = axis.barh(
        positions + height / 2,
        np.maximum(reduced, 0.8),
        height=height,
        color=[METHOD_COLORS.get(method, "#333333") for method in records["method"]],
        label="Reduced features",
    )
    _annotate_bars(axis, original_bars, original)
    _annotate_bars(axis, reduced_bars, reduced)

    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlim(left=0.8)
    axis.set_xlabel("Number of features (log scale)")
    axis.set_title("Original vs reduced feature dimensions")
    axis.grid(axis="x", alpha=0.25, which="both")
    legend_handles = [Patch(facecolor=ORIGINAL_COLOR, label="Original features")]
    legend_handles.extend(
        Patch(
            facecolor=METHOD_COLORS.get(method, "#333333"),
            label=f"Reduced ({METHOD_LABELS.get(method, method)})",
        )
        for method in records["method"].unique()
    )
    axis.legend(handles=legend_handles, loc="lower right", fontsize=8)
    figure.text(
        0.01,
        0.01,
        "Reduced bars are coloured by method; the baseline bar equals the original count.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))

    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_pca_cumulative_variance(
    pca_results: Mapping[str, "ReductionResult"],
    output_path: str | Path | None = None,
    target_variance: float | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Line chart of cumulative PCA explained variance per dataset."""
    if not pca_results:
        raise ValueError("No PCA results were supplied.")

    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    colors = colormaps["tab10"]
    datasets = _ordered_datasets(list(pca_results))

    for index, dataset in enumerate(datasets):
        result = pca_results[dataset]
        if result.cumulative_explained_variance is None:
            raise ValueError(f"{dataset}: PCA cumulative explained variance is missing.")
        cumulative = np.asarray(result.cumulative_explained_variance, dtype=np.float64)
        components = np.arange(1, cumulative.size + 1)
        color = colors(index % 10)
        axis.plot(
            components,
            cumulative,
            color=color,
            linewidth=1.8,
            label=f"{_dataset_label(dataset)} ({result.n_components} PCs)",
        )
        selected = int(result.n_components)
        if 1 <= selected <= cumulative.size:
            axis.scatter(
                [selected],
                [cumulative[selected - 1]],
                color=color,
                edgecolor="black",
                zorder=5,
                s=45,
            )
            near_right_edge = selected > 0.85 * cumulative.size
            axis.annotate(
                f"{cumulative[selected - 1]:.4f}",
                xy=(selected, cumulative[selected - 1]),
                xytext=(-8 if near_right_edge else 8, -12),
                textcoords="offset points",
                ha="right" if near_right_edge else "left",
                fontsize=8,
                color=color,
            )

    if target_variance is not None:
        target = float(target_variance)
        if not 0.0 < target < 1.0:
            raise ValueError("target_variance must be strictly between 0 and 1.")
        axis.axhline(
            target,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=f"Target variance ({target:.0%})",
        )

    axis.set_xlabel("Number of principal components retained")
    axis.set_ylabel("Cumulative explained variance ratio")
    axis.set_ylim(0.0, 1.02)
    axis.set_xlim(left=1)
    axis.set_title("PCA cumulative explained variance (fit on complete training pool)")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()

    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_pca_component_comparison(
    statistics: "pd.DataFrame",
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Grouped bars of original versus PCA component counts per dataset."""
    _require_columns(
        statistics,
        STATISTICS_COLUMNS + ("explained_variance_retained",),
    )
    records = statistics[statistics["method"] == "pca"].drop_duplicates(
        subset=["dataset"]
    )
    if records.empty:
        raise ValueError("No PCA statistics are available to plot.")
    datasets = _ordered_datasets(records["dataset"].tolist())
    records = records.set_index("dataset").loc[datasets]

    positions = np.arange(len(datasets))
    width = 0.36
    original = records["original_features"].to_numpy(dtype=float)
    reduced = records["reduced_features"].to_numpy(dtype=float)
    retained = records["explained_variance_retained"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    original_bars = axis.bar(
        positions - width / 2,
        original,
        width=width,
        color=ORIGINAL_COLOR,
        label="Original features",
    )
    reduced_bars = axis.bar(
        positions + width / 2,
        reduced,
        width=width,
        color=METHOD_COLORS["pca"],
        label="PCA components",
    )
    for bar, value, retained_value in zip(reduced_bars, reduced, retained):
        axis.annotate(
            f"{int(value):,} PCs\n{retained_value:.2%} variance",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, value in zip(original_bars, original):
        axis.annotate(
            f"{int(value):,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis.set_xticks(positions, [_dataset_label(name) for name in datasets])
    axis.set_ylabel("Number of features")
    axis.set_title("PCA components versus original feature count")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()

    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_lda_component_comparison(
    statistics: "pd.DataFrame",
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Grouped bars of LDA component counts per dataset and proportion."""
    _require_columns(
        statistics,
        STATISTICS_COLUMNS + ("max_components", "labelled_proportion"),
    )
    records = statistics[statistics["method"] == "lda"].copy()
    if records.empty:
        raise ValueError("No LDA statistics are available to plot.")
    records = records.sort_values(["dataset", "labelled_proportion"])
    datasets = _ordered_datasets(records["dataset"].tolist())
    proportions = sorted(records["labelled_proportion"].unique())

    positions = np.arange(len(datasets))
    width = 0.8 / max(len(proportions), 1)
    colors = colormaps["viridis"]

    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    maximum = float(records["reduced_features"].max())
    for index, proportion in enumerate(proportions):
        subset = (
            records[records["labelled_proportion"] == proportion]
            .set_index("dataset")
            .reindex(datasets)
        )
        counts = subset["reduced_features"].to_numpy(dtype=float)
        offset = (index - (len(proportions) - 1) / 2) * width
        bars = axis.bar(
            positions + offset,
            counts,
            width=width,
            color=colors(index / max(len(proportions) - 1, 1)),
            label=f"{proportion:.0%} labelled",
        )
        for bar, value in zip(bars, counts):
            if np.isfinite(value):
                axis.annotate(
                    f"{int(value)}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    caps = records.groupby("dataset")["max_components"].max().reindex(datasets)
    for dataset_index, (dataset, cap) in enumerate(zip(datasets, caps.to_numpy(dtype=float))):
        if np.isfinite(cap):
            axis.annotate(
                f"max {int(cap)}",
                xy=(
                    positions[dataset_index]
                    + (len(proportions) - 1) * width / 2
                    + 0.18,
                    max(cap, 0.4),
                ),
                ha="left",
                va="bottom",
                fontsize=8,
                color="#555555",
            )

    axis.set_xticks(positions, [_dataset_label(name) for name in datasets])
    axis.set_ylabel("LDA components")
    axis.set_ylim(0, maximum + max(2.0, maximum * 0.25))
    axis.set_title("LDA components per dataset and labelled proportion")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()

    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_reduction_percentages(
    statistics: "pd.DataFrame",
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Grouped bars of the percentage of features removed by each method."""
    _require_columns(statistics, STATISTICS_COLUMNS)
    records = statistics.drop_duplicates(
        subset=["dataset", "method"]
    ).sort_values(["dataset", "method"])
    if records.empty:
        raise ValueError("No reduction statistics are available to plot.")
    datasets = _ordered_datasets(records["dataset"].tolist())
    methods = [method for method in METHOD_ORDER if method in set(records["method"])]

    positions = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)

    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    for index, method in enumerate(methods):
        subset = (
            records[records["method"] == method]
            .set_index("dataset")
            .reindex(datasets)
        )
        values = subset["dimensionality_reduction_percent"].to_numpy(dtype=float)
        offset = (index - (len(methods) - 1) / 2) * width
        bars = axis.bar(
            positions + offset,
            values,
            width=width,
            color=METHOD_COLORS.get(method, "#333333"),
            label=METHOD_LABELS.get(method, method),
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.annotate(
                    f"{value:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axis.set_xticks(positions, [_dataset_label(name) for name in datasets])
    axis.set_ylabel("Features removed (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Dimensionality reduction achieved by method")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()

    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def build_reduction_figure_report(
    statistics: "pd.DataFrame",
    pca_results: Mapping[str, "ReductionResult"],
    figures_dir: str | Path = "results/dimensionality_reduction/figures",
    target_variance: float | None = None,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Generate and save the five Phase 4 reduction figures."""
    directory = Path(figures_dir)
    paths: list[Path] = []
    specifications = (
        (
            "feature_dimensions_baseline_vs_reduced.png",
            lambda path: plot_original_vs_reduced_dimensions(
                statistics, output_path=path, dpi=dpi
            ),
        ),
        (
            "pca_cumulative_explained_variance.png",
            lambda path: plot_pca_cumulative_variance(
                pca_results,
                output_path=path,
                target_variance=target_variance,
                dpi=dpi,
            ),
        ),
        (
            "pca_component_comparison.png",
            lambda path: plot_pca_component_comparison(
                statistics, output_path=path, dpi=dpi
            ),
        ),
        (
            "lda_component_comparison.png",
            lambda path: plot_lda_component_comparison(
                statistics, output_path=path, dpi=dpi
            ),
        ),
        (
            "dimensionality_reduction_percentages.png",
            lambda path: plot_reduction_percentages(
                statistics, output_path=path, dpi=dpi
            ),
        ),
    )
    for filename, plotter in specifications:
        target = directory / filename
        plotter(target)
        if not target.exists():
            raise RuntimeError(f"Plot function did not write {target}.")
        paths.append(target)
    return paths


# ---------------------------------------------------------------------------
# Phase 5 classifier figures
# ---------------------------------------------------------------------------
CLASSIFIER_ORDER: tuple[str, ...] = ("logistic_regression", "svm", "knn")
CLASSIFIER_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM (linear)",
    "knn": "KNN",
}
CLASSIFIER_COLORS: dict[str, str] = {
    "logistic_regression": "#4C72B0",
    "svm": "#DD8452",
    "knn": "#55A868",
}
REPRESENTATION_ORDER: tuple[str, ...] = ("baseline", "pca", "lda")
REPRESENTATION_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "pca": "PCA",
    "lda": "LDA",
}
REPRESENTATION_COLORS: dict[str, str] = {
    "baseline": METHOD_COLORS["none"],
    "pca": METHOD_COLORS["pca"],
    "lda": METHOD_COLORS["lda"],
}
METRIC_LABELS: dict[str, str] = {"accuracy": "Accuracy", "macro_f1": "Macro F1"}

CLASSIFIER_RESULT_COLUMNS: tuple[str, ...] = (
    "dataset",
    "labelled_percentage",
    "classifier",
    "feature_representation",
    "accuracy",
    "macro_f1",
    "training_time_seconds",
    "status",
)


def _completed_results(results: "pd.DataFrame") -> "pd.DataFrame":
    """Return validated, completed classifier experiments only."""
    if results is None or results.empty:
        raise ValueError("Classifier results are empty.")
    missing = [
        column for column in CLASSIFIER_RESULT_COLUMNS if column not in results.columns
    ]
    if missing:
        raise ValueError(
            "Classifier results are missing required column(s): " + ", ".join(missing)
        )
    completed = results[results["status"] == "completed"].copy()
    if completed.empty:
        raise ValueError("No completed classifier experiments are available to plot.")
    for metric in ("accuracy", "macro_f1"):
        values = completed[metric].astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Classifier results contain invalid {metric} values.")
    if (completed["training_time_seconds"].astype(float) < 0.0).any():
        raise ValueError("Training times must be non-negative.")
    return completed


def _dataset_results(results: "pd.DataFrame", dataset: str) -> "pd.DataFrame":
    completed = _completed_results(results)
    frame = completed[completed["dataset"] == dataset].copy()
    if frame.empty:
        raise ValueError(f"No completed experiments for dataset {dataset!r}.")
    return frame


def _present(values: Sequence[str], order: Sequence[str]) -> list[str]:
    present = set(values)
    ordered = [value for value in order if value in present]
    ordered.extend(sorted(present.difference(order)))
    return ordered


def _grouped_bar_panel(
    axis: Any,
    frame: "pd.DataFrame",
    metric: str,
    primary_column: str,
    primary_order: Sequence[str],
    primary_labels: Mapping[str, str],
    secondary_column: str,
    secondary_order: Sequence[str],
    secondary_labels: Mapping[str, str],
    secondary_colors: Mapping[str, str],
    axis_label: str | None = None,
    value_format: str = "{:.4f}",
) -> float:
    """Draw grouped mean bars and return the bar width."""
    positions = np.arange(len(primary_order), dtype=float)
    width = 0.8 / max(len(secondary_order), 1)
    for index, secondary in enumerate(secondary_order):
        means: list[float] = []
        for primary in primary_order:
            values = frame.loc[
                (frame[primary_column] == primary)
                & (frame[secondary_column] == secondary),
                metric,
            ].to_numpy(dtype=float)
            means.append(float(values.mean()) if values.size else float("nan"))
        offset = (index - (len(secondary_order) - 1) / 2.0) * width
        bars = axis.bar(
            positions + offset,
            means,
            width=width,
            color=secondary_colors.get(secondary, "#555555"),
            label=secondary_labels.get(secondary, secondary),
        )
        for bar, value in zip(bars, means):
            if np.isfinite(value):
                axis.annotate(
                    value_format.format(value),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    axis.set_xticks(
        positions, [primary_labels.get(value, value) for value in primary_order]
    )
    if axis_label is not None:
        axis.set_ylabel(axis_label)
    axis.grid(axis="y", alpha=0.25)
    return width


def _overlay_points(
    axis: Any,
    frame: "pd.DataFrame",
    metric: str,
    primary_column: str,
    primary_order: Sequence[str],
    secondary_column: str,
    secondary_order: Sequence[str],
    width: float,
) -> None:
    """Scatter the individual experiment values over grouped mean bars."""
    for index, secondary in enumerate(secondary_order):
        offset = (index - (len(secondary_order) - 1) / 2.0) * width
        for position, primary in enumerate(primary_order):
            values = frame.loc[
                (frame[primary_column] == primary)
                & (frame[secondary_column] == secondary),
                metric,
            ].to_numpy(dtype=float)
            if values.size == 0:
                continue
            if values.size == 1:
                jitter = np.zeros(1, dtype=float)
            else:
                jitter = np.linspace(-width * 0.18, width * 0.18, values.size)
            axis.scatter(
                np.full(values.size, position + offset) + jitter,
                values,
                s=12,
                color="#222222",
                alpha=0.55,
                linewidths=0,
                zorder=4,
            )


def _plot_metric_by_representation(
    results: "pd.DataFrame",
    dataset: str,
    metric: str,
    output_path: str | Path | None,
    dpi: int,
) -> "matplotlib.figure.Figure":
    frame = _dataset_results(results, dataset)
    representations = _present(
        frame["feature_representation"].unique().tolist(), REPRESENTATION_ORDER
    )
    classifiers = _present(frame["classifier"].unique().tolist(), CLASSIFIER_ORDER)
    metric_label = METRIC_LABELS.get(metric, metric)

    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    width = _grouped_bar_panel(
        axis,
        frame,
        metric,
        "feature_representation",
        representations,
        REPRESENTATION_LABELS,
        "classifier",
        classifiers,
        CLASSIFIER_LABELS,
        CLASSIFIER_COLORS,
        axis_label=metric_label,
    )
    _overlay_points(
        axis,
        frame,
        metric,
        "feature_representation",
        representations,
        "classifier",
        classifiers,
        width,
    )
    axis.set_xlabel("Feature representation")
    axis.set_ylim(0.0, 1.18)
    axis.set_title(
        f"{_dataset_label(dataset)}: {metric_label} across feature representations"
    )
    axis.legend(
        title="Classifier",
        loc="upper center",
        ncol=max(len(classifiers), 1),
        fontsize=8,
    )

    figure.text(
        0.01,
        0.01,
        "Bars show the mean over labelled percentages (10/25/50%); dots are "
        "individual experiments.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))

    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_accuracy_by_representation(
    results: "pd.DataFrame",
    dataset: str,
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Accuracy of each classifier across baseline, PCA, and LDA."""
    return _plot_metric_by_representation(
        results, dataset, "accuracy", output_path, dpi
    )


def plot_macro_f1_by_representation(
    results: "pd.DataFrame",
    dataset: str,
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Macro F1 of each classifier across baseline, PCA, and LDA."""
    return _plot_metric_by_representation(
        results, dataset, "macro_f1", output_path, dpi
    )


def plot_classifier_comparison(
    results: "pd.DataFrame",
    dataset: str,
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Compare logistic regression, linear SVM, and KNN on accuracy and macro F1."""
    frame = _dataset_results(results, dataset)
    classifiers = _present(frame["classifier"].unique().tolist(), CLASSIFIER_ORDER)
    representations = _present(
        frame["feature_representation"].unique().tolist(), REPRESENTATION_ORDER
    )

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)
    for axis, metric in zip(axes, ("accuracy", "macro_f1")):
        width = _grouped_bar_panel(
            axis,
            frame,
            metric,
            "classifier",
            classifiers,
            CLASSIFIER_LABELS,
            "feature_representation",
            representations,
            REPRESENTATION_LABELS,
            REPRESENTATION_COLORS,
            axis_label=METRIC_LABELS[metric],
        )
        _overlay_points(
            axis,
            frame,
            metric,
            "classifier",
            classifiers,
            "feature_representation",
            representations,
            width,
        )
        axis.set_xlabel("Classifier")
        axis.set_ylim(0.0, 1.05)
        axis.set_title(METRIC_LABELS[metric])

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(len(representations), 1),
        title="Feature representation",
        fontsize=8,
    )
    figure.suptitle(f"{_dataset_label(dataset)}: classifier comparison")
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))

    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_performance_by_labelled_proportion(
    results: "pd.DataFrame",
    dataset: str,
    metric: str = "macro_f1",
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """One line panel per classifier showing metric versus labelled percentage."""
    frame = _dataset_results(results, dataset)
    metric_label = METRIC_LABELS.get(metric, metric)
    classifiers = _present(frame["classifier"].unique().tolist(), CLASSIFIER_ORDER)
    representations = _present(
        frame["feature_representation"].unique().tolist(), REPRESENTATION_ORDER
    )
    percentages = sorted(frame["labelled_percentage"].unique().tolist())

    values = frame[metric].astype(float).to_numpy()
    low = max(0.0, float(values.min()) - 0.03)
    high = min(1.0, float(values.max()) + 0.03)
    if high - low < 0.15:
        high = min(1.0, low + 0.15)

    figure, axes = plt.subplots(
        1, len(classifiers), figsize=(5.2 * len(classifiers), 4.8), sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, classifier in zip(axes, classifiers):
        for representation in representations:
            subset = frame[
                (frame["classifier"] == classifier)
                & (frame["feature_representation"] == representation)
            ].sort_values("labelled_percentage")
            axis.plot(
                subset["labelled_percentage"].to_numpy(dtype=float),
                subset[metric].to_numpy(dtype=float),
                marker="o",
                linewidth=1.8,
                color=REPRESENTATION_COLORS.get(representation, "#333333"),
                label=REPRESENTATION_LABELS.get(representation, representation),
            )
        axis.set_title(CLASSIFIER_LABELS.get(classifier, classifier))
        axis.set_xlabel("Labelled training data (%)")
        axis.set_xticks([float(value) for value in percentages])
        axis.set_ylim(low, high)
        axis.grid(alpha=0.25)

    axes[0].set_ylabel(metric_label)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(len(representations), 1),
        title="Feature representation",
        fontsize=8,
    )
    figure.suptitle(
        f"{_dataset_label(dataset)}: {metric_label} versus labelled data"
    )
    figure.text(
        0.01,
        0.01,
        "The y-axis is scaled to the observed range so small differences remain visible.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))

    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_training_time_by_classifier(
    results: "pd.DataFrame",
    dataset: str,
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Mean model-fitting time per classifier and feature representation."""
    frame = _dataset_results(results, dataset)
    classifiers = _present(frame["classifier"].unique().tolist(), CLASSIFIER_ORDER)
    representations = _present(
        frame["feature_representation"].unique().tolist(), REPRESENTATION_ORDER
    )
    times = frame["training_time_seconds"].astype(float).to_numpy()
    positive = times[times > 0.0]
    linthresh = max(float(positive.min()) / 10.0, 1e-5) if positive.size else 1e-4

    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    width = _grouped_bar_panel(
        axis,
        frame,
        "training_time_seconds",
        "classifier",
        classifiers,
        CLASSIFIER_LABELS,
        "feature_representation",
        representations,
        REPRESENTATION_LABELS,
        REPRESENTATION_COLORS,
        axis_label="Training time (seconds, symlog scale)",
        value_format="{:.3f}s",
    )
    _overlay_points(
        axis,
        frame,
        "training_time_seconds",
        "classifier",
        classifiers,
        "feature_representation",
        representations,
        width,
    )
    axis.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
    axis.set_xlabel("Classifier")
    axis.set_title(
        f"{_dataset_label(dataset)}: model fitting time by classifier"
    )
    axis.legend(title="Feature representation", loc="upper left", fontsize=8)
    figure.text(
        0.01,
        0.01,
        "Times measure model fitting only. KNN defers neighbour search to "
        "prediction, so its fit time is close to zero.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))

    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def build_classifier_figure_report(
    results: "pd.DataFrame",
    figures_dir: str | Path = "results/classifier_evaluation/figures",
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Generate and save the five Phase 5 figures for every dataset."""
    directory = Path(figures_dir)
    completed = _completed_results(results)
    datasets = _ordered_datasets(completed["dataset"].unique().tolist())
    paths: list[Path] = []

    specifications = (
        (
            "accuracy_by_representation_{dataset}.png",
            plot_accuracy_by_representation,
        ),
        (
            "macro_f1_by_representation_{dataset}.png",
            plot_macro_f1_by_representation,
        ),
        ("classifier_comparison_{dataset}.png", plot_classifier_comparison),
        (
            "performance_by_labelled_proportion_{dataset}.png",
            plot_performance_by_labelled_proportion,
        ),
        (
            "training_time_by_classifier_{dataset}.png",
            plot_training_time_by_classifier,
        ),
    )

    for dataset in datasets:
        for filename_template, plotter in specifications:
            target = directory / filename_template.format(dataset=dataset)
            plotter(results, dataset, output_path=target, dpi=dpi)
            if not target.exists():
                raise RuntimeError(f"Plot function did not write {target}.")
            paths.append(target)
    return paths


# ---------------------------------------------------------------------------
# Phase 6 self-training figures
# ---------------------------------------------------------------------------
SUPERVISED_COLOR: str = "#8C8C8C"
SELF_TRAINING_COLOR: str = "#4C72B0"
PERCENTAGE_COLORS: dict[int, str] = {
    10: "#4C72B0",
    25: "#DD8452",
    50: "#55A868",
}
PERCENTAGE_MARKERS: dict[int, str] = {10: "o", 25: "s", 50: "^"}


def _comparison_values(comparison: "pd.DataFrame") -> "pd.DataFrame":
    required = (
        "dataset",
        "labelled_percentage",
        "supervised_accuracy",
        "self_training_accuracy",
        "supervised_macro_f1",
        "self_training_macro_f1",
        "macro_f1_improvement",
    )
    missing = [column for column in required if column not in comparison.columns]
    if missing:
        raise ValueError(
            "Comparison table is missing required column(s): " + ", ".join(missing)
        )
    valid = comparison.dropna(subset=["macro_f1_improvement"]).copy()
    if valid.empty:
        raise ValueError("No comparable supervised/self-training experiments.")
    return valid


def _configuration_labels(frame: "pd.DataFrame") -> list[str]:
    return [
        f"{REPRESENTATION_LABELS.get(row.feature_representation, row.feature_representation)}"
        f"\n{CLASSIFIER_LABELS.get(row.classifier, row.classifier)}"
        for row in frame.itertuples(index=False)
    ]


def plot_supervised_vs_self_training_metric(
    comparison: "pd.DataFrame",
    metric: str,
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Compare mean supervised and self-training performance per dataset."""
    if metric not in ("accuracy", "macro_f1"):
        raise ValueError(f"Unsupported metric {metric!r}.")
    valid = _comparison_values(comparison)
    metric_label = METRIC_LABELS[metric]
    datasets = _ordered_datasets(valid["dataset"].unique().tolist())
    percentages = sorted(valid["labelled_percentage"].unique().tolist())

    figure, axes = plt.subplots(
        1, len(datasets), figsize=(5.2 * len(datasets), 4.6), squeeze=False
    )
    lower = min(
        valid[f"supervised_{metric}"].min(),
        valid[f"self_training_{metric}"].min(),
    )
    lower = max(0.0, float(lower) - 0.08)
    positions = np.arange(len(percentages), dtype=float)
    width = 0.36

    for axis, dataset in zip(axes[0], datasets):
        subset = valid[valid["dataset"] == dataset]
        supervised = [
            subset[subset["labelled_percentage"] == percentage][
                f"supervised_{metric}"
            ].mean()
            for percentage in percentages
        ]
        self_training = [
            subset[subset["labelled_percentage"] == percentage][
                f"self_training_{metric}"
            ].mean()
            for percentage in percentages
        ]
        left_bars = axis.bar(
            positions - width / 2,
            supervised,
            width,
            color=SUPERVISED_COLOR,
            label="Supervised (Phase 5)",
        )
        right_bars = axis.bar(
            positions + width / 2,
            self_training,
            width,
            color=SELF_TRAINING_COLOR,
            label="Self-training (Phase 6)",
        )
        for bars in (left_bars, right_bars):
            for bar in bars:
                height = bar.get_height()
                axis.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        axis.set_xticks(positions)
        axis.set_xticklabels([f"{int(value)}%" for value in percentages])
        axis.set_xlabel("Labelled training proportion")
        axis.set_ylabel(metric_label)
        axis.set_ylim(lower, 1.03)
        axis.set_title(_dataset_label(dataset))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")

    figure.suptitle(
        f"Supervised vs self-training {metric_label} "
        "(mean over the 9 representation x classifier configurations)"
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    figure.text(
        0.01,
        0.01,
        "Bars average over baseline/PCA/LDA and LR/SVM/KNN. Test-set metrics; "
        "no threshold tuning was performed.",
        fontsize=8,
        color="#555555",
    )
    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_supervised_vs_self_training_detail(
    comparison: "pd.DataFrame",
    dataset: str,
    metric: str,
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Compare supervised and self-training results for one dataset in detail."""
    valid = _comparison_values(comparison)
    subset = valid[valid["dataset"] == dataset].copy()
    if subset.empty:
        raise ValueError(f"No comparable experiments for dataset {dataset!r}.")
    percentages = sorted(subset["labelled_percentage"].unique().tolist())
    metric_label = METRIC_LABELS[metric]

    figure, axes = plt.subplots(
        1, len(percentages), figsize=(5.4 * len(percentages), 5.0), squeeze=False
    )
    width = 0.38
    for axis, percentage in zip(axes[0], percentages):
        block = subset[subset["labelled_percentage"] == percentage].copy()
        block = block.sort_values(
            ["feature_representation", "classifier"], kind="stable"
        )
        positions = np.arange(len(block), dtype=float)
        axis.bar(
            positions - width / 2,
            block[f"supervised_{metric}"],
            width,
            color=SUPERVISED_COLOR,
            label="Supervised",
        )
        axis.bar(
            positions + width / 2,
            block[f"self_training_{metric}"],
            width,
            color=SELF_TRAINING_COLOR,
            label="Self-training",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(_configuration_labels(block), fontsize=7, rotation=45, ha="right")
        axis.set_ylim(
            max(0.0, float(block[f"supervised_{metric}"].min()) - 0.08), 1.03
        )
        axis.set_ylabel(metric_label)
        axis.set_title(f"{int(percentage)}% labelled")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    figure.suptitle(f"{_dataset_label(dataset)}: supervised vs self-training {metric_label}")
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_macro_f1_improvement_by_dataset(
    comparison: "pd.DataFrame",
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Show the distribution of macro-F1 improvements for each dataset."""
    valid = _comparison_values(comparison)
    datasets = _ordered_datasets(valid["dataset"].unique().tolist())
    representations = _present(
        valid["feature_representation"].unique().tolist(), REPRESENTATION_ORDER
    )
    rng = np.random.RandomState(42)

    figure, axis = plt.subplots(figsize=(8.6, 5.0))
    for offset, representation in enumerate(representations):
        color = REPRESENTATION_COLORS.get(representation, "#777777")
        for position, dataset in enumerate(datasets):
            values = valid[
                (valid["dataset"] == dataset)
                & (valid["feature_representation"] == representation)
            ]["macro_f1_improvement"].to_numpy(dtype=float)
            if values.size == 0:
                continue
            jitter = rng.uniform(-0.10, 0.10, size=values.size)
            axis.scatter(
                position + (offset - (len(representations) - 1) / 2) * 0.18 + jitter,
                values,
                s=26,
                alpha=0.75,
                color=color,
                label=REPRESENTATION_LABELS.get(representation, representation)
                if position == 0
                else None,
            )
    means = [
        valid[valid["dataset"] == dataset]["macro_f1_improvement"].mean()
        for dataset in datasets
    ]
    axis.plot(
        np.arange(len(datasets)),
        means,
        color="black",
        marker="D",
        markersize=7,
        linewidth=1.2,
        label="Dataset mean",
        zorder=5,
    )
    axis.axhline(0.0, color="#B22222", linewidth=1.0, linestyle="--")
    axis.set_xticks(np.arange(len(datasets)))
    axis.set_xticklabels([_dataset_label(name) for name in datasets])
    axis.set_ylabel("Macro-F1 improvement (self-training - supervised)")
    axis.set_title("Macro-F1 improvement by dataset")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_self_training_performance_by_percentage(
    results: "pd.DataFrame",
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Self-training accuracy and macro F1 across labelled proportions."""
    completed = _completed_results(results)
    datasets = _ordered_datasets(completed["dataset"].unique().tolist())
    representations = _present(
        completed["feature_representation"].unique().tolist(), REPRESENTATION_ORDER
    )
    percentages = sorted(completed["labelled_percentage"].unique().tolist())

    figure, axes = plt.subplots(
        2, len(datasets), figsize=(5.2 * len(datasets), 7.4), squeeze=False
    )
    for row, metric in enumerate(("accuracy", "macro_f1")):
        for column, dataset in enumerate(datasets):
            axis = axes[row][column]
            subset = completed[completed["dataset"] == dataset]
            for representation in representations:
                block = subset[subset["feature_representation"] == representation]
                values = [
                    block[block["labelled_percentage"] == percentage][metric].mean()
                    for percentage in percentages
                ]
                axis.plot(
                    percentages,
                    values,
                    marker="o",
                    linewidth=1.6,
                    color=REPRESENTATION_COLORS.get(representation, "#777777"),
                    label=REPRESENTATION_LABELS.get(
                        representation, representation
                    ),
                )
            axis.set_xticks(percentages)
            axis.set_xticklabels([f"{int(value)}%" for value in percentages])
            axis.set_xlabel("Labelled training proportion")
            axis.set_ylabel(METRIC_LABELS[metric])
            axis.set_title(f"{_dataset_label(dataset)}: {METRIC_LABELS[metric]}")
            axis.grid(alpha=0.25)
            if row == 0:
                axis.legend(fontsize=8, loc="lower right")
    figure.suptitle(
        "Self-training performance across labelled proportions "
        "(mean over LR/SVM/KNN per representation)"
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.95))
    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_pseudo_label_acceptance_counts(
    results: "pd.DataFrame",
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Pseudo-label acceptance counts by representation and proportion."""
    completed = _completed_results(results)
    datasets = _ordered_datasets(completed["dataset"].unique().tolist())
    representations = _present(
        completed["feature_representation"].unique().tolist(), REPRESENTATION_ORDER
    )
    percentages = sorted(completed["labelled_percentage"].unique().tolist())

    figure, axes = plt.subplots(
        1, len(datasets), figsize=(5.4 * len(datasets), 4.8), squeeze=False
    )
    positions = np.arange(len(percentages), dtype=float)
    width = 0.8 / max(1, len(representations))
    for axis, dataset in zip(axes[0], datasets):
        subset = completed[completed["dataset"] == dataset]
        for offset, representation in enumerate(representations):
            block = subset[subset["feature_representation"] == representation]
            values = [
                block[block["labelled_percentage"] == percentage][
                    "pseudo_label_count"
                ].mean()
                for percentage in percentages
            ]
            axis.bar(
                positions + (offset - (len(representations) - 1) / 2) * width,
                values,
                width,
                color=REPRESENTATION_COLORS.get(representation, "#777777"),
                label=REPRESENTATION_LABELS.get(representation, representation),
            )
        axis.set_xticks(positions)
        axis.set_xticklabels([f"{int(value)}%" for value in percentages])
        axis.set_xlabel("Labelled training proportion")
        axis.set_ylabel("Mean pseudo-labelled samples")
        axis.set_title(_dataset_label(dataset))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8, loc="upper right")
    figure.suptitle(
        "Pseudo-label acceptance counts (mean over classifiers; max = full pool)"
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_pseudo_label_acceptance_heatmap(
    results: "pd.DataFrame",
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Show pseudo-label acceptance rates as a configuration heatmap."""
    completed = _completed_results(results)
    datasets = _ordered_datasets(completed["dataset"].unique().tolist())
    percentages = sorted(completed["labelled_percentage"].unique().tolist())
    configurations = [
        (representation, classifier)
        for representation in REPRESENTATION_ORDER
        for classifier in CLASSIFIER_ORDER
    ]

    figure, axes = plt.subplots(
        1, len(datasets), figsize=(5.0 * len(datasets), 5.0), squeeze=False
    )
    for axis, dataset in zip(axes[0], datasets):
        subset = completed[completed["dataset"] == dataset]
        matrix = np.full((len(configurations), len(percentages)), np.nan)
        for row, (representation, classifier) in enumerate(configurations):
            for column, percentage in enumerate(percentages):
                values = subset[
                    (subset["feature_representation"] == representation)
                    & (subset["classifier"] == classifier)
                    & (subset["labelled_percentage"] == percentage)
                ]["pseudo_label_acceptance_rate"]
                if not values.empty:
                    matrix[row, column] = float(values.iloc[0])
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_xticks(np.arange(len(percentages)))
        axis.set_xticklabels([f"{int(value)}%" for value in percentages])
        axis.set_yticks(np.arange(len(configurations)))
        axis.set_yticklabels(
            [
                f"{REPRESENTATION_LABELS.get(representation, representation)} + "
                f"{CLASSIFIER_LABELS.get(classifier, classifier)}"
                for representation, classifier in configurations
            ],
            fontsize=7,
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isfinite(value):
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value < 0.55 else "black",
                    )
        axis.set_xlabel("Labelled training proportion")
        axis.set_title(_dataset_label(dataset))
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Acceptance rate")
    figure.suptitle("Pseudo-label acceptance rate by configuration")
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def _iteration_frame(iteration_logs: Mapping[str, "pd.DataFrame"]) -> "pd.DataFrame":
    if not iteration_logs:
        raise ValueError("No iteration logs are available to plot.")
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for identifier, log in iteration_logs.items():
        parts = str(identifier).split("__")
        if len(parts) != 4:
            continue
        dataset, labelled, representation, classifier = parts
        percentage = int(labelled.replace("labelled_", ""))
        for record in log.to_dict(orient="records"):
            rows.append(
                {
                    "experiment_id": identifier,
                    "dataset": dataset,
                    "labelled_percentage": percentage,
                    "feature_representation": representation,
                    "classifier": classifier,
                    "iteration": int(record["iteration"]),
                    "cumulative_pseudo_labels": int(
                        record["cumulative_pseudo_labels"]
                    ),
                    "remaining_unlabelled": int(record["remaining_unlabelled"]),
                    "pseudo_labels_added": int(record["pseudo_labels_added"]),
                }
            )
    if not rows:
        raise ValueError("Iteration logs contain no usable records.")
    return pd.DataFrame(rows)


def plot_pseudo_label_growth(
    iteration_logs: Mapping[str, "pd.DataFrame"],
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Mean cumulative pseudo-label count by iteration and dataset."""
    frame = _iteration_frame(iteration_logs)
    datasets = _ordered_datasets(frame["dataset"].unique().tolist())
    percentages = sorted(frame["labelled_percentage"].unique().tolist())

    figure, axes = plt.subplots(
        1, len(datasets), figsize=(5.2 * len(datasets), 4.6), squeeze=False
    )
    for axis, dataset in zip(axes[0], datasets):
        subset = frame[frame["dataset"] == dataset]
        for percentage in percentages:
            block = subset[subset["labelled_percentage"] == percentage]
            grouped = (
                block.groupby("iteration")["cumulative_pseudo_labels"]
                .agg(["mean", "min", "max"])
                .reset_index()
            )
            color = PERCENTAGE_COLORS.get(percentage, "#777777")
            axis.plot(
                grouped["iteration"],
                grouped["mean"],
                marker=PERCENTAGE_MARKERS.get(percentage, "o"),
                color=color,
                label=f"{percentage}% labelled",
            )
            axis.fill_between(
                grouped["iteration"],
                grouped["min"],
                grouped["max"],
                color=color,
                alpha=0.15,
            )
        axis.set_xlabel("Self-training iteration")
        axis.set_ylabel("Cumulative pseudo-labels (mean)")
        axis.set_title(_dataset_label(dataset))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    figure.suptitle(
        "Pseudo-label growth across iterations (shaded band = min-max over configurations)"
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_pseudo_label_growth_dataset(
    iteration_logs: Mapping[str, "pd.DataFrame"],
    dataset: str,
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Dataset-level pseudo-label growth with one line per representation."""
    frame = _iteration_frame(iteration_logs)
    subset = frame[frame["dataset"] == dataset]
    if subset.empty:
        raise ValueError(f"No iteration logs for dataset {dataset!r}.")
    percentages = sorted(subset["labelled_percentage"].unique().tolist())
    representations = _present(
        subset["feature_representation"].unique().tolist(), REPRESENTATION_ORDER
    )

    figure, axes = plt.subplots(
        1, len(percentages), figsize=(5.0 * len(percentages), 4.4), squeeze=False
    )
    for axis, percentage in zip(axes[0], percentages):
        block = subset[subset["labelled_percentage"] == percentage]
        for representation in representations:
            values = block[
                block["feature_representation"] == representation
            ]
            grouped = (
                values.groupby("iteration")["cumulative_pseudo_labels"]
                .mean()
                .reset_index()
            )
            if grouped.empty:
                continue
            axis.plot(
                grouped["iteration"],
                grouped["cumulative_pseudo_labels"],
                marker="o",
                color=REPRESENTATION_COLORS.get(representation, "#777777"),
                label=REPRESENTATION_LABELS.get(representation, representation),
            )
        axis.set_xticks(sorted(block["iteration"].unique().tolist()))
        axis.set_xlabel("Self-training iteration")
        axis.set_ylabel("Cumulative pseudo-labels (mean over classifiers)")
        axis.set_title(f"{int(percentage)}% labelled")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    figure.suptitle(f"{_dataset_label(dataset)}: pseudo-label growth by representation")
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_training_time_comparison(
    comparison: "pd.DataFrame",
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Compare mean supervised and self-training fit time per dataset."""
    valid = _comparison_values(comparison)
    datasets = _ordered_datasets(valid["dataset"].unique().tolist())
    percentages = sorted(valid["labelled_percentage"].unique().tolist())

    figure, axes = plt.subplots(
        1, len(datasets), figsize=(5.2 * len(datasets), 4.6), squeeze=False
    )
    positions = np.arange(len(percentages), dtype=float)
    width = 0.38
    for axis, dataset in zip(axes[0], datasets):
        subset = valid[valid["dataset"] == dataset]
        supervised = [
            subset[subset["labelled_percentage"] == percentage][
                "supervised_training_time_seconds"
            ].mean()
            for percentage in percentages
        ]
        self_training = [
            subset[subset["labelled_percentage"] == percentage][
                "self_training_training_time_seconds"
            ].mean()
            for percentage in percentages
        ]
        axis.bar(
            positions - width / 2,
            supervised,
            width,
            color=SUPERVISED_COLOR,
            label="Supervised",
        )
        axis.bar(
            positions + width / 2,
            self_training,
            width,
            color=SELF_TRAINING_COLOR,
            label="Self-training",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([f"{int(value)}%" for value in percentages])
        axis.set_yscale("symlog", linthresh=0.01)
        axis.set_xlabel("Labelled training proportion")
        axis.set_ylabel("Mean training time (s, symlog)")
        axis.set_title(_dataset_label(dataset))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8, loc="upper left")
    figure.suptitle(
        "Training-time comparison (self-training includes all iterative refits)"
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def plot_representation_comparison_self_training(
    results: "pd.DataFrame",
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> "matplotlib.figure.Figure":
    """Compare baseline, PCA, and LDA under self-training."""
    completed = _completed_results(results)
    datasets = _ordered_datasets(completed["dataset"].unique().tolist())
    representations = _present(
        completed["feature_representation"].unique().tolist(), REPRESENTATION_ORDER
    )

    figure, axes = plt.subplots(
        2, len(datasets), figsize=(5.2 * len(datasets), 7.4), squeeze=False
    )
    for row, metric in enumerate(("accuracy", "macro_f1")):
        for column, dataset in enumerate(datasets):
            axis = axes[row][column]
            subset = completed[completed["dataset"] == dataset]
            values = [
                subset[subset["feature_representation"] == representation][
                    metric
                ].mean()
                for representation in representations
            ]
            bars = axis.bar(
                np.arange(len(representations)),
                values,
                color=[
                    REPRESENTATION_COLORS.get(representation, "#777777")
                    for representation in representations
                ],
            )
            for bar, value in zip(bars, values):
                axis.annotate(
                    f"{value:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            axis.set_xticks(np.arange(len(representations)))
            axis.set_xticklabels(
                [
                    REPRESENTATION_LABELS.get(representation, representation)
                    for representation in representations
                ]
            )
            axis.set_ylim(max(0.0, float(min(values)) - 0.08), 1.03)
            axis.set_ylabel(METRIC_LABELS[metric])
            axis.set_title(f"{_dataset_label(dataset)}: {METRIC_LABELS[metric]}")
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Baseline vs PCA vs LDA under self-training "
        "(mean over classifiers and labelled proportions)"
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    if output_path is not None:
        save_figure(figure, output_path, dpi=dpi)
    return figure


def build_self_training_figure_report(
    results: "pd.DataFrame",
    comparison: "pd.DataFrame | None",
    iteration_logs: Mapping[str, "pd.DataFrame"],
    figures_dir: str | Path = "results/self_training/figures",
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Generate and save all Phase 6 figures."""
    if comparison is None:
        raise ValueError("A supervised/self-training comparison is required for figures.")
    directory = Path(figures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    completed = _completed_results(results)
    datasets = _ordered_datasets(completed["dataset"].unique().tolist())
    paths: list[Path] = []

    for metric in ("accuracy", "macro_f1"):
        target = directory / f"supervised_vs_self_training_{metric}.png"
        plot_supervised_vs_self_training_metric(
            comparison, metric, output_path=target, dpi=dpi
        )
        paths.append(target)
        for dataset in datasets:
            target = directory / f"supervised_vs_self_training_{metric}_{dataset}.png"
            plot_supervised_vs_self_training_detail(
                comparison, dataset, metric, output_path=target, dpi=dpi
            )
            paths.append(target)

    target = directory / "macro_f1_improvement_by_dataset.png"
    plot_macro_f1_improvement_by_dataset(comparison, output_path=target, dpi=dpi)
    paths.append(target)

    target = directory / "self_training_performance_by_labelled_percentage.png"
    plot_self_training_performance_by_percentage(
        results, output_path=target, dpi=dpi
    )
    paths.append(target)

    target = directory / "pseudo_label_acceptance_counts.png"
    plot_pseudo_label_acceptance_counts(results, output_path=target, dpi=dpi)
    paths.append(target)

    target = directory / "pseudo_label_acceptance_rate_heatmap.png"
    plot_pseudo_label_acceptance_heatmap(results, output_path=target, dpi=dpi)
    paths.append(target)

    target = directory / "pseudo_label_growth.png"
    plot_pseudo_label_growth(iteration_logs, output_path=target, dpi=dpi)
    paths.append(target)
    for dataset in datasets:
        target = directory / f"pseudo_label_growth_{dataset}.png"
        plot_pseudo_label_growth_dataset(
            iteration_logs, dataset, output_path=target, dpi=dpi
        )
        paths.append(target)

    target = directory / "training_time_comparison.png"
    plot_training_time_comparison(comparison, output_path=target, dpi=dpi)
    paths.append(target)

    target = directory / "representation_comparison_self_training.png"
    plot_representation_comparison_self_training(
        results, output_path=target, dpi=dpi
    )
    paths.append(target)

    return paths
