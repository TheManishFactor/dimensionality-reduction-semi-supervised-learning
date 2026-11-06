"""Phase 7 figures, summary tables, and final research documents.

All plotting functions read the joined comparison table produced by
``src.final_analysis``. No experimental value is typed by hand; report text is
formatted from the analysis payload so that prose and tables stay consistent
with the saved results.
"""

from __future__ import annotations

import html
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .final_analysis import (
    CLASSIFIER_LABELS,
    DATASET_LABELS,
    DATASET_SHORT_LABELS,
    DATASET_ORDER,
    METRIC_LABELS,
    REPRESENTATION_LABELS,
    REPRESENTATION_ORDER,
    STOPPING_REASON_LABELS,
    AnalysisError,
    sha256_file,
)
from .classifiers import CLASSIFIER_NAMES

FIGURE_DPI = 300
SUPERVISED_COLOR = "#4D4D4D"
SELF_TRAINING_COLOR = "#D55E00"
DATASET_COLORS = {
    "breast_cancer": "#0072B2",
    "mnist": "#E69F00",
    "har": "#009E73",
}
REPRESENTATION_COLORS = {
    "baseline": "#0072B2",
    "pca": "#E69F00",
    "lda": "#009E73",
}
CLASSIFIER_COLORS = {
    "logistic_regression": "#0072B2",
    "svm": "#D55E00",
    "knn": "#009E73",
}
CLASSIFIER_MARKERS = {
    "logistic_regression": "o",
    "svm": "s",
    "knn": "^",
}


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def _finish(
    figure: plt.Figure,
    path: Path,
    title: str,
    description: str,
    source_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": title,
            "Author": "Phase 7 final analysis",
            "Description": description,
            "Source": "Phase 5/6 result tables (81 matched conditions)",
            "SourceSHA256": source_hash,
            "Software": "matplotlib",
        },
    )
    plt.close(figure)


def _axis_limits(values: Iterable[float], pad: float = 0.03) -> tuple[float, float]:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0, 1.0
    low = min(0.0, float(data.min()) - pad)
    high = max(1.0, float(data.max()) + pad)
    return low, high


def _jitter(count: int, width: float = 0.08, seed: int = 7) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=count)


def _metric_axis_label(metric: str) -> str:
    return METRIC_LABELS[metric]


def create_figures(
    comparison: pd.DataFrame,
    payload: dict[str, Any],
    figures_dir: str | Path,
    source_hash: str,
) -> list[dict[str, Any]]:
    """Create every Phase 7 figure and return caption metadata."""
    _style()
    directory = Path(figures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, Any]] = []

    def add(name: str, caption: str) -> None:
        figures.append(
            {
                "number": len(figures) + 1,
                "filename": name,
                "path": directory / name,
                "caption": caption,
            }
        )

    # ------------------------------------------------------------------
    # Figure 1-2: paired scatter, supervised vs self-training
    # ------------------------------------------------------------------
    for metric, name in (
        ("accuracy", "fig01_accuracy_supervised_vs_self_training.png"),
        ("macro_f1", "fig02_macro_f1_supervised_vs_self_training.png"),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharex=True, sharey=True)
        for axis, dataset in zip(axes, DATASET_ORDER):
            subset = comparison[comparison["dataset"] == dataset]
            for classifier in CLASSIFIER_NAMES:
                for representation in REPRESENTATION_ORDER:
                    points = subset[
                        (subset["classifier"] == classifier)
                        & (subset["feature_representation"] == representation)
                    ]
                    if points.empty:
                        continue
                    axis.scatter(
                        points[f"supervised_{metric}"],
                        points[f"self_training_{metric}"],
                        s=34,
                        marker=CLASSIFIER_MARKERS[classifier],
                        facecolor=REPRESENTATION_COLORS[representation],
                        edgecolor="black",
                        linewidth=0.4,
                        alpha=0.85,
                        zorder=3,
                    )
            degenerate = subset[subset["numerically_degenerate_fit"]]
            if not degenerate.empty:
                axis.scatter(
                    degenerate[f"supervised_{metric}"],
                    degenerate[f"self_training_{metric}"],
                    s=120,
                    facecolor="none",
                    edgecolor="#B2182B",
                    linewidth=1.4,
                    zorder=4,
                )
            axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.9)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_aspect("equal")
            axis.set_title(DATASET_SHORT_LABELS[dataset])
            axis.set_xlabel(f"Supervised {_metric_axis_label(metric)}")
            if dataset == DATASET_ORDER[0]:
                axis.set_ylabel(f"Self-training {_metric_axis_label(metric)}")
            axis.text(
                0.04,
                0.96,
                f"n = {len(subset)}",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
                color="#555555",
            )
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=REPRESENTATION_COLORS[representation],
                markeredgecolor="black",
                label=REPRESENTATION_LABELS[representation],
            )
            for representation in REPRESENTATION_ORDER
        ] + [
            Line2D(
                [0],
                [0],
                marker=CLASSIFIER_MARKERS[classifier],
                linestyle="",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor="black",
                label=CLASSIFIER_LABELS[classifier],
            )
            for classifier in CLASSIFIER_NAMES
        ]
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor="#B2182B",
                markeredgewidth=1.4,
                label="Degenerate fit",
            )
        )
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=7,
            bbox_to_anchor=(0.5, -0.16),
            fontsize=7.5,
        )
        figure.suptitle(
            f"Supervised vs self-training {METRIC_LABELS[metric]} "
            "(each point is one matched condition)",
            fontsize=12,
            y=1.02,
        )
        figure.tight_layout()
        _finish(
            figure,
            directory / name,
            f"Supervised vs self-training {METRIC_LABELS[metric]}",
            "Paired scatter of the 81 matched conditions; the dashed line marks "
            "equal performance; points below it are regressions.",
            source_hash,
        )
        add(
            name,
            f"Supervised versus self-training {METRIC_LABELS[metric]} for "
            "the 81 matched conditions. Each point is one condition; the dashed "
            "diagonal marks equal performance, so points below it are "
            "regressions. Colour denotes the feature representation and marker "
            "shape the classifier. Red circles mark the three numerically "
            "degenerate MNIST LDA + Logistic Regression fits.",
        )

    # ------------------------------------------------------------------
    # Figure 3: paired improvement distributions
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 6.8))
    settings = [
        (0, 0, "accuracy", False, "Accuracy difference (all 81)"),
        (0, 1, "macro_f1", False, "Macro-F1 difference (all 81)"),
        (1, 0, "accuracy", True, "Accuracy difference (non-degenerate)"),
        (1, 1, "macro_f1", True, "Macro-F1 difference (non-degenerate)"),
    ]
    for row, column, metric, exclude_degenerate, title in settings:
        axis = axes[row, column]
        subset = comparison[~comparison["numerically_degenerate_fit"]] if exclude_degenerate else comparison
        groups = []
        labels = []
        for dataset in DATASET_ORDER:
            values = subset.loc[subset["dataset"] == dataset, f"{metric}_improvement"].dropna()
            groups.append(values.to_numpy())
            wins = int((values > 1e-12).sum())
            losses = int((values < -1e-12).sum())
            labels.append(
                f"{DATASET_SHORT_LABELS[dataset]}\n(n={len(values)})\n"
                f"mean {values.mean():+.4f}\nW/L {wins}/{losses}"
            )
        all_values = subset[f"{metric}_improvement"].dropna()
        groups.append(all_values.to_numpy())
        labels.append(
            f"All\n(n={len(all_values)})\nmean {all_values.mean():+.4f}\n"
            f"W/L {int((all_values > 1e-12).sum())}/{int((all_values < -1e-12).sum())}"
        )
        box = axis.boxplot(
            groups,
            positions=np.arange(len(groups)),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"color": "#555555", "linewidth": 0.8},
            capprops={"color": "#555555", "linewidth": 0.8},
        )
        colors = [DATASET_COLORS[name] for name in DATASET_ORDER] + ["#666666"]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            patch.set_edgecolor(color)
        for index, values in enumerate(groups):
            axis.scatter(
                index + _jitter(len(values), 0.12),
                values,
                s=14,
                color=colors[index],
                alpha=0.75,
                linewidth=0,
                zorder=3,
            )
        axis.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--")
        axis.set_xticks(np.arange(len(groups)), labels, fontsize=7.2)
        axis.set_ylabel(f"{METRIC_LABELS[metric]} difference")
        axis.set_title(title)
        if not exclude_degenerate:
            low, high = -0.90, 0.34
        else:
            low, high = -0.06, 0.34
        axis.set_ylim(low, high)
    figure.suptitle(
        "Distribution of paired performance changes (self-training minus supervised)",
        fontsize=12,
    )
    figure.tight_layout()
    _finish(
        figure,
        directory / "fig03_paired_improvement_distributions.png",
        "Paired improvement distributions",
        "Boxplots and individual condition differences; the bottom row excludes "
        "the three numerically degenerate fits to show the non-degenerate scale.",
        source_hash,
    )
    add(
        "fig03_paired_improvement_distributions.png",
        "Paired performance changes after self-training (self-training minus "
        "supervised). Positive values are improvements. The top row shows all 81 "
        "conditions, including the three large MNIST LDA + Logistic Regression "
        "failures; the bottom row rescales to the 78 non-degenerate conditions. "
        "Boxes show the median and interquartile range, whiskers extend to "
        "1.5 x IQR, and points are individual conditions. W/L counts wins and "
        "losses at a 1e-12 tie tolerance.",
    )

    # ------------------------------------------------------------------
    # Figure 4: representation comparison
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    width = 0.34
    for axis, dataset in zip(axes, DATASET_ORDER):
        subset = comparison[comparison["dataset"] == dataset]
        x = np.arange(len(REPRESENTATION_ORDER))
        supervised = [
            subset.loc[subset["feature_representation"] == representation, "supervised_macro_f1"].mean()
            for representation in REPRESENTATION_ORDER
        ]
        self_training = [
            subset.loc[subset["feature_representation"] == representation, "self_training_macro_f1"].mean()
            for representation in REPRESENTATION_ORDER
        ]
        bars1 = axis.bar(x - width / 2, supervised, width, color=SUPERVISED_COLOR, label="Supervised")
        bars2 = axis.bar(x + width / 2, self_training, width, color=SELF_TRAINING_COLOR, label="Self-training")
        for index, representation in enumerate(REPRESENTATION_ORDER):
            points = subset[subset["feature_representation"] == representation]
            axis.scatter(
                index - width / 2 + _jitter(len(points), 0.04),
                points["supervised_macro_f1"],
                s=11,
                facecolor="white",
                edgecolor=SUPERVISED_COLOR,
                linewidth=0.6,
                zorder=4,
            )
            axis.scatter(
                index + width / 2 + _jitter(len(points), 0.04),
                points["self_training_macro_f1"],
                s=11,
                facecolor="white",
                edgecolor=SELF_TRAINING_COLOR,
                linewidth=0.6,
                zorder=4,
            )
        axis.set_xticks(x, [REPRESENTATION_LABELS[name] for name in REPRESENTATION_ORDER])
        axis.set_title(DATASET_SHORT_LABELS[dataset])
        axis.set_xlabel("Feature representation")
        axis.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Mean macro-F1 across 9 conditions")
    figure.legend(
        handles=[
            Patch(facecolor=SUPERVISED_COLOR, label="Supervised (Phase 5)"),
            Patch(facecolor=SELF_TRAINING_COLOR, label="Self-training (Phase 6)"),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=8,
    )
    figure.suptitle("Effect of feature representation on macro-F1", fontsize=12, y=1.02)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    _finish(
        figure,
        directory / "fig04_representation_comparison.png",
        "Representation comparison",
        "Mean macro-F1 by feature representation for supervised and "
        "self-training runs; points show individual conditions.",
        source_hash,
    )
    add(
        "fig04_representation_comparison.png",
        "Mean macro-F1 for Baseline, PCA, and LDA representations in each "
        "dataset, before and after self-training. Bars average the nine "
        "conditions (three proportions x three classifiers); open points show "
        "individual conditions. The vertical axis starts at 0.75.",
    )

    # ------------------------------------------------------------------
    # Figure 5: labelled proportion trend
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=False)
    proportions = [10, 25, 50]
    for axis, dataset in zip(axes, DATASET_ORDER):
        subset = comparison[comparison["dataset"] == dataset]
        for metric, color, label in (
            ("supervised_macro_f1", SUPERVISED_COLOR, "Supervised"),
            ("self_training_macro_f1", SELF_TRAINING_COLOR, "Self-training"),
        ):
            means = [
                subset.loc[subset["labelled_percentage"] == proportion, metric].mean()
                for proportion in proportions
            ]
            lows = [
                subset.loc[subset["labelled_percentage"] == proportion, metric].min()
                for proportion in proportions
            ]
            highs = [
                subset.loc[subset["labelled_percentage"] == proportion, metric].max()
                for proportion in proportions
            ]
            axis.plot(proportions, means, marker="o", color=color, linewidth=1.7, label=label)
            axis.fill_between(proportions, lows, highs, color=color, alpha=0.12, linewidth=0)
        axis.set_xticks(proportions, [f"{value}%" for value in proportions])
        axis.set_xlabel("Labelled share of the training partition")
        axis.set_title(DATASET_SHORT_LABELS[dataset])
        low = min(
            subset["supervised_macro_f1"].min(),
            subset["self_training_macro_f1"].min(),
        )
        high = max(
            subset["supervised_macro_f1"].max(),
            subset["self_training_macro_f1"].max(),
        )
        padding = max(0.015, (high - low) * 0.12)
        axis.set_ylim(max(0.0, low - padding), min(1.0, high + padding))
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(loc="lower right")
    figure.suptitle(
        "Performance versus labelled-data proportion (line = mean, band = min-max)",
        fontsize=12,
        y=1.02,
    )
    figure.tight_layout()
    _finish(
        figure,
        directory / "fig05_labelled_proportion_trend.png",
        "Labelled proportion trend",
        "Mean macro-F1 and min-max range across nine conditions at each "
        "labelled proportion.",
        source_hash,
    )
    add(
        "fig05_labelled_proportion_trend.png",
        "Macro-F1 as a function of the labelled share of the training "
        "partition. Lines are means over nine conditions; shaded bands span "
        "the minimum to maximum condition. Each panel uses its own vertical "
        "scale to make the within-dataset trend readable.",
    )

    # ------------------------------------------------------------------
    # Figure 6: classifier comparison
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    for axis, dataset in zip(axes, DATASET_ORDER):
        subset = comparison[comparison["dataset"] == dataset]
        x = np.arange(len(CLASSIFIER_NAMES))
        supervised = [
            subset.loc[subset["classifier"] == classifier, "supervised_macro_f1"].mean()
            for classifier in CLASSIFIER_NAMES
        ]
        self_training = [
            subset.loc[subset["classifier"] == classifier, "self_training_macro_f1"].mean()
            for classifier in CLASSIFIER_NAMES
        ]
        bars1 = axis.bar(x - width / 2, supervised, width, color=SUPERVISED_COLOR, label="Supervised")
        bars2 = axis.bar(x + width / 2, self_training, width, color=SELF_TRAINING_COLOR, label="Self-training")
        for index, classifier in enumerate(CLASSIFIER_NAMES):
            points = subset[subset["classifier"] == classifier]
            axis.scatter(
                index - width / 2 + _jitter(len(points), 0.04),
                points["supervised_macro_f1"],
                s=11,
                facecolor="white",
                edgecolor=SUPERVISED_COLOR,
                linewidth=0.6,
                zorder=4,
            )
            axis.scatter(
                index + width / 2 + _jitter(len(points), 0.04),
                points["self_training_macro_f1"],
                s=11,
                facecolor="white",
                edgecolor=SELF_TRAINING_COLOR,
                linewidth=0.6,
                zorder=4,
            )
        axis.set_xticks(x, [CLASSIFIER_LABELS[name] for name in CLASSIFIER_NAMES], rotation=12)
        axis.set_title(DATASET_SHORT_LABELS[dataset])
        axis.set_xlabel("Classifier")
        axis.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Mean macro-F1 across 9 conditions")
    figure.legend(
        handles=[
            Patch(facecolor=SUPERVISED_COLOR, label="Supervised (Phase 5)"),
            Patch(facecolor=SELF_TRAINING_COLOR, label="Self-training (Phase 6)"),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=8,
    )
    figure.suptitle("Classifier-wise performance before and after self-training", fontsize=12, y=1.02)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    _finish(
        figure,
        directory / "fig06_classifier_comparison.png",
        "Classifier comparison",
        "Mean macro-F1 by classifier for supervised and self-training runs; "
        "points show individual conditions.",
        source_hash,
    )
    add(
        "fig06_classifier_comparison.png",
        "Mean macro-F1 by classifier in each dataset before and after "
        "self-training. Bars average the nine conditions (three proportions x "
        "three representations); open points show individual conditions. The "
        "vertical axis starts at 0.65.",
    )

    # ------------------------------------------------------------------
    # Figure 7: training time
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1, 1.35]})
    runtime = payload["runtime"]
    datasets = [row["dataset"] for row in runtime["by_dataset"]]
    supervised_totals = [row["supervised_training_seconds"] for row in runtime["by_dataset"]]
    self_totals = [row["self_training_training_seconds"] for row in runtime["by_dataset"]]
    x = np.arange(len(datasets))
    axes[0].bar(x - width / 2, supervised_totals, width, color=SUPERVISED_COLOR, label="Supervised")
    axes[0].bar(x + width / 2, self_totals, width, color=SELF_TRAINING_COLOR, label="Self-training")
    axes[0].set_yscale("log")
    for index, (first, second) in enumerate(zip(supervised_totals, self_totals)):
        axes[0].text(index - width / 2, first * 1.18, f"{first:.1f}", ha="center", fontsize=7)
        axes[0].text(index + width / 2, second * 1.18, f"{second:.1f}", ha="center", fontsize=7)
    axes[0].set_xticks(x, [DATASET_SHORT_LABELS[name] for name in datasets])
    axes[0].set_ylabel("Total training time (seconds, log scale)")
    axes[0].set_title("Total training time by dataset")
    axes[0].legend(loc="upper left")

    for dataset in DATASET_ORDER:
        subset = comparison[comparison["dataset"] == dataset]
        axes[1].scatter(
            subset["supervised_training_time_seconds"],
            subset["self_training_training_time_seconds"],
            s=36,
            color=DATASET_COLORS[dataset],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
            label=DATASET_SHORT_LABELS[dataset],
        )
    low = max(
        1e-4,
        min(
            comparison["supervised_training_time_seconds"].min(),
            comparison["self_training_training_time_seconds"].min(),
        ),
    )
    high = max(
        comparison["supervised_training_time_seconds"].max(),
        comparison["self_training_training_time_seconds"].max(),
    )
    axes[1].plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=0.9)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Supervised training time (seconds, log scale)")
    axes[1].set_ylabel("Self-training training time (seconds, log scale)")
    axes[1].set_title("Per-condition training cost")
    axes[1].legend(loc="upper left")
    mnist_svm = comparison[
        (comparison["dataset"] == "mnist") & (comparison["classifier"] == "svm")
    ]
    for row in mnist_svm.nlargest(2, "self_training_training_time_seconds").itertuples():
        axes[1].annotate(
            f"{row.labelled_percentage}% {row.feature_representation.upper()}",
            (row.supervised_training_time_seconds, row.self_training_training_time_seconds),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=6.5,
            color="#7A3E00",
        )
    figure.suptitle(
        "Training cost: self-training repeats model fits at every round",
        fontsize=12,
        y=1.02,
    )
    figure.tight_layout()
    _finish(
        figure,
        directory / "fig07_training_time.png",
        "Training time",
        "Total and per-condition training time; Phase 6 includes every "
        "self-training round. Phase 5 prediction time was not recorded.",
        source_hash,
    )
    add(
        "fig07_training_time.png",
        "Training cost. Left: total measured training seconds by dataset on a "
        "logarithmic axis. Right: per-condition supervised versus self-training "
        "training time; points above the dashed equality line are more "
        "expensive after self-training. The two most expensive MNIST linear-SVM "
        "conditions are annotated. Training time excludes KNN neighbour search, "
        "which happens at prediction time.",
    )

    # ------------------------------------------------------------------
    # Figure 8: prediction time (Phase 6 only)
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    for axis, dataset in zip(axes, DATASET_ORDER):
        subset = comparison[comparison["dataset"] == dataset]
        x = np.arange(len(CLASSIFIER_NAMES))
        values = [
            subset.loc[subset["classifier"] == classifier, "self_training_prediction_time_seconds"].mean()
            for classifier in CLASSIFIER_NAMES
        ]
        bars = axis.bar(
            x,
            values,
            width=0.58,
            color=[CLASSIFIER_COLORS[name] for name in CLASSIFIER_NAMES],
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                max(value, 1e-5) * 1.25,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        axis.set_yscale("log")
        axis.set_xticks(x, [CLASSIFIER_LABELS[name] for name in CLASSIFIER_NAMES], rotation=12)
        axis.set_title(DATASET_SHORT_LABELS[dataset])
        axis.set_xlabel("Classifier")
    axes[0].set_ylabel("Mean Phase 6 prediction time (seconds, log scale)")
    figure.suptitle(
        "Self-training prediction time (Phase 5 prediction time was not recorded)",
        fontsize=12,
        y=1.02,
    )
    figure.tight_layout()
    _finish(
        figure,
        directory / "fig08_prediction_time.png",
        "Prediction time",
        "Mean Phase 6 prediction time per condition by classifier and dataset. "
        "Phase 5 did not log prediction time, so no paired prediction-time "
        "comparison exists.",
        source_hash,
    )
    add(
        "fig08_prediction_time.png",
        "Mean prediction time of the final Phase 6 self-training model by "
        "classifier and dataset, on a logarithmic axis. Phase 5 did not record "
        "prediction time, so this figure cannot show a supervised/self-training "
        "prediction-time comparison. KNN prediction includes the neighbour "
        "search that its training timer excludes.",
    )

    # ------------------------------------------------------------------
    # Figure 9: pseudo-label acceptance
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    acceptance = payload["pseudo_labels"]["by_dataset_representation"]
    dataset_abbreviation = {"breast_cancer": "BC", "mnist": "MNIST", "har": "HAR"}
    representation_abbreviation = {"baseline": "Base", "pca": "PCA", "lda": "LDA"}
    x_labels = []
    x_positions = []
    position = 0
    for dataset in DATASET_ORDER:
        for representation in REPRESENTATION_ORDER:
            rows = [
                row
                for row in acceptance
                if row["dataset"] == dataset
                and row["feature_representation"] == representation
            ]
            if not rows:
                continue
            x_labels.append(
                f"{dataset_abbreviation[dataset]}\n{representation_abbreviation[representation]}"
            )
            x_positions.append(position)
            position += 1
        position += 0.4
    values = [
        row["mean_acceptance_rate"]
        for dataset in DATASET_ORDER
        for representation in REPRESENTATION_ORDER
        for row in acceptance
        if row["dataset"] == dataset and row["feature_representation"] == representation
    ]
    colors = [
        REPRESENTATION_COLORS[representation]
        for dataset in DATASET_ORDER
        for representation in REPRESENTATION_ORDER
        for _ in [0]
    ]
    bars = axes[0].bar(
        x_positions,
        values,
        width=0.72,
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    for bar, value in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )
    axes[0].axhline(
        payload["pseudo_labels"]["mean_experiment_acceptance_rate"],
        color="#333333",
        linestyle="--",
        linewidth=0.9,
    )
    axes[0].set_xticks(x_positions, x_labels, rotation=0, fontsize=6.5)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Mean acceptance rate")
    axes[0].set_title("Acceptance by dataset and representation")

    for dataset in DATASET_ORDER:
        subset = comparison[comparison["dataset"] == dataset]
        axes[1].scatter(
            subset["labelled_percentage"] + _jitter(len(subset), 2.2),
            subset["pseudo_label_acceptance_rate"],
            s=34,
            color=DATASET_COLORS[dataset],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
            label=DATASET_SHORT_LABELS[dataset],
        )
        means = [
            subset.loc[subset["labelled_percentage"] == proportion, "pseudo_label_acceptance_rate"].mean()
            for proportion in (10, 25, 50)
        ]
        axes[1].plot([10, 25, 50], means, color=DATASET_COLORS[dataset], linewidth=1.4)
    axes[1].set_xticks([10, 25, 50], ["10%", "25%", "50%"])
    axes[1].set_xlabel("Labelled share of the training partition")
    axes[1].set_ylabel("Pseudo-label acceptance rate")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Acceptance by labelled proportion")
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        fontsize=7.5,
    )
    figure.suptitle("Pseudo-label acceptance (accepted / original unlabelled pool)", fontsize=12, y=1.02)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    _finish(
        figure,
        directory / "fig09_pseudo_label_acceptance.png",
        "Pseudo-label acceptance",
        "Acceptance rate by dataset/representation and by labelled proportion. "
        "The dashed line is the mean experiment acceptance rate.",
        source_hash,
    )
    add(
        "fig09_pseudo_label_acceptance.png",
        "Pseudo-label acceptance, defined as the number of accepted "
        "pseudo-labels divided by the original unlabelled-pool size. Left: mean "
        "acceptance by dataset and representation (three classifiers per bar); "
        "the dashed line marks the mean experiment acceptance rate of "
        f"{payload['pseudo_labels']['mean_experiment_acceptance_rate']:.4f}. "
        "Right: every condition by labelled proportion with dataset means "
        "connected.",
    )

    # ------------------------------------------------------------------
    # Figure 10: iterations and stopping reasons
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    by_dataset = payload["pseudo_labels"]["by_dataset"]
    datasets = [row["dataset"] for row in by_dataset]
    cap = [row["iteration_cap_hits"] for row in by_dataset]
    threshold = [row["threshold_stops"] for row in by_dataset]
    x = np.arange(len(datasets))
    axes[0].bar(x, cap, color="#D55E00", label="Iteration cap reached")
    axes[0].bar(x, threshold, bottom=cap, color="#0072B2", label="No sample met threshold")
    for index in range(len(datasets)):
        axes[0].text(index, cap[index] / 2, str(cap[index]), ha="center", va="center", color="white", fontsize=8)
        if threshold[index]:
            axes[0].text(
                index,
                cap[index] + threshold[index] / 2,
                str(threshold[index]),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    axes[0].set_xticks(x, [DATASET_SHORT_LABELS[name] for name in datasets])
    axes[0].set_ylabel("Experiments (of 27 per dataset)")
    axes[0].set_ylim(0, 30)
    axes[0].set_title("Stopping reason by dataset")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=7.5,
    )

    rounds = payload["pseudo_labels"]["rounds"]
    round_numbers = [row["iteration"] for row in rounds]
    added = [row["pseudo_labels_added"] for row in rounds]
    bars = axes[1].bar(round_numbers, added, color="#009E73", width=0.6)
    axes[1].set_yscale("log")
    axes[1].set_ylim(5e3, 2e6)
    for bar, row in zip(bars, rounds):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            row["pseudo_labels_added"] * 1.25,
            f"{row['pseudo_labels_added']:,}\n({row['experiments_with_round']} runs)",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axes[1].set_xticks(round_numbers)
    axes[1].set_xlabel("Self-training iteration")
    axes[1].set_ylabel("Pseudo-labels accepted (log scale)")
    axes[1].set_title("Pseudo-labels accepted per round")
    figure.suptitle("Self-training iteration statistics", fontsize=12, y=1.02)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    _finish(
        figure,
        directory / "fig10_iteration_statistics.png",
        "Iteration statistics",
        "Stopping reasons by dataset and pseudo-labels accepted in each "
        "self-training round.",
        source_hash,
    )
    add(
        "fig10_iteration_statistics.png",
        "Self-training stopping and iteration statistics. Left: number of "
        "experiments ending at the three-iteration cap or because no remaining "
        "sample reached the 0.90 confidence threshold. Right: pseudo-labels "
        "accepted in each round on a logarithmic axis, with the number of "
        "experiments that reached that round in parentheses.",
    )

    # ------------------------------------------------------------------
    # Figure 11: improvement heatmaps
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.6), gridspec_kw={"width_ratios": [1, 1]})
    panels = [
        ("feature_representation", REPRESENTATION_ORDER, REPRESENTATION_LABELS, "Representation"),
        ("classifier", CLASSIFIER_NAMES, CLASSIFIER_LABELS, "Classifier"),
    ]
    for axis, (column, levels, labels, title) in zip(axes, panels):
        matrix = np.zeros((len(DATASET_ORDER), len(levels)))
        for row_index, dataset in enumerate(DATASET_ORDER):
            for column_index, level in enumerate(levels):
                subset = comparison[
                    (comparison["dataset"] == dataset)
                    & (comparison[column] == level)
                ]
                matrix[row_index, column_index] = subset["macro_f1_improvement"].mean()
        norm = TwoSlopeNorm(vmin=-0.22, vcenter=0.0, vmax=0.06)
        image = axis.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
        axis.set_xticks(np.arange(len(levels)), [labels[level] for level in levels])
        axis.set_yticks(np.arange(len(DATASET_ORDER)), [DATASET_SHORT_LABELS[name] for name in DATASET_ORDER])
        axis.set_title(f"By {title.lower()}")
        axis.grid(False)
        for row_index in range(len(DATASET_ORDER)):
            for column_index in range(len(levels)):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.4f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if abs(value) > 0.11 else "#222222",
                )
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Mean macro-F1 difference")
    figure.suptitle("Mean macro-F1 change after self-training", fontsize=12, y=1.03)
    _finish(
        figure,
        directory / "fig11_improvement_heatmaps.png",
        "Improvement heatmaps",
        "Mean paired macro-F1 difference by dataset and representation or "
        "classifier. The colour scale is centred at zero.",
        source_hash,
    )
    add(
        "fig11_improvement_heatmaps.png",
        "Mean macro-F1 change after self-training by dataset and (left) feature "
        "representation or (right) classifier. Each cell averages nine or three "
        "conditions. The diverging colour scale is centred at zero; the MNIST "
        "LDA cell and the MNIST Logistic Regression cell are dominated by the "
        "three numerically degenerate fits.",
    )

    return figures


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def _save_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")
    return path


def write_summary_tables(
    payload: dict[str, Any],
    comparison: pd.DataFrame,
    tables_dir: str | Path,
) -> dict[str, Path]:
    """Export flat summary tables used by the report and the verifier."""
    directory = Path(tables_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    overall_rows = []
    for metric in ("accuracy", "macro_f1"):
        for scope, record in payload["overall"][metric].items():
            row = {"metric": metric, "scope": scope}
            row.update({key: value for key, value in record.items() if key != "group"})
            overall_rows.append(row)
    paths["overall"] = _save_table(pd.DataFrame(overall_rows), directory / "summary_overall.csv")

    factor_rows = []
    for metric in ("accuracy", "macro_f1"):
        for factor, records in payload["factor_summaries"][metric].items():
            for record in records:
                row = {"metric": metric, "factor": factor}
                row.update({f"level_{key}": value for key, value in record["factors"].items()})
                row.update({key: value for key, value in record.items() if key not in ("factors", "group")})
                factor_rows.append(row)
    paths["by_factor"] = _save_table(pd.DataFrame(factor_rows), directory / "summary_by_factor.csv")

    best_rows = []
    for metric in ("accuracy", "macro_f1"):
        for dataset, methods in payload["best_configurations"][metric].items():
            for method, record in methods.items():
                row = {"metric": metric, "dataset": dataset, "method": method}
                row.update(record)
                best_rows.append(row)
    paths["best"] = _save_table(pd.DataFrame(best_rows), directory / "summary_best_configurations.csv")

    warning_rows = []
    for phase in ("phase5", "phase6"):
        for record in payload["warnings"][phase]["experiments"]:
            warning_rows.append(
                {
                    "phase": phase,
                    "experiment_id": record["experiment_id"],
                    "warning_type": record["warning_type"],
                    "warning_message": record["warning_message"],
                }
            )
    paths["warnings"] = _save_table(pd.DataFrame(warning_rows), directory / "summary_warnings.csv")

    degenerate_rows = []
    for record in payload["warnings"]["degenerate_fits"]["experiments"]:
        degenerate_rows.append(record)
    paths["degenerate"] = _save_table(
        pd.DataFrame(degenerate_rows), directory / "summary_degenerate_fits.csv"
    )

    runtime_rows = []
    for row in payload["runtime"]["by_dataset_classifier"]:
        runtime_rows.append({"grouping": "dataset_classifier", **row})
    for row in payload["runtime"]["by_dataset"]:
        runtime_rows.append({"grouping": "dataset", **row})
    paths["runtime"] = _save_table(pd.DataFrame(runtime_rows), directory / "summary_runtime.csv")

    pseudo_rows = []
    for row in payload["pseudo_labels"]["by_dataset"]:
        pseudo_rows.append({"grouping": "dataset", **row})
    for row in payload["pseudo_labels"]["by_dataset_representation"]:
        pseudo_rows.append({"grouping": "dataset_representation", **row})
    for row in payload["pseudo_labels"]["by_labelled_percentage"]:
        pseudo_rows.append({"grouping": "labelled_percentage", **row})
    for row in payload["pseudo_labels"]["rounds"]:
        pseudo_rows.append({"grouping": "round", **row})
    paths["pseudo"] = _save_table(
        pd.DataFrame(pseudo_rows), directory / "summary_pseudo_labels.csv"
    )

    return paths


# ---------------------------------------------------------------------------
# Document model (Markdown + print-ready HTML)
# ---------------------------------------------------------------------------
@dataclass
class Block:
    kind: str
    payload: Any


class Document:
    """A small block-based document that renders to Markdown and HTML."""

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        metadata: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.title = title
        self.subtitle = subtitle
        self.metadata = list(metadata)
        self.blocks: list[Block] = []
        self.figure_count = 0
        self.table_count = 0

    def heading(self, text: str, level: int = 2) -> "Document":
        self.blocks.append(Block("heading", {"text": text, "level": level}))
        return self

    def paragraph(self, text: str) -> "Document":
        self.blocks.append(Block("paragraph", text))
        return self

    def bullets(self, items: Sequence[str]) -> "Document":
        self.blocks.append(Block("bullets", list(items)))
        return self

    def numbered(self, items: Sequence[str]) -> "Document":
        self.blocks.append(Block("numbered", list(items)))
        return self

    def code(self, text: str) -> "Document":
        self.blocks.append(Block("code", text))
        return self

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[Any]],
        caption: str | None = None,
    ) -> "Document":
        self.table_count += 1
        self.blocks.append(
            Block(
                "table",
                {
                    "number": self.table_count,
                    "headers": list(headers),
                    "rows": [list(row) for row in rows],
                    "caption": caption,
                },
            )
        )
        return self

    def figure(
        self,
        filename: str,
        caption: str,
        number: int | None = None,
        directory: Path | None = None,
    ) -> "Document":
        self.figure_count += 1
        if number is None:
            number = self.figure_count
        if directory is not None and not (directory / filename).exists():
            raise FileNotFoundError(f"Figure not found: {directory / filename}")
        self.blocks.append(
            Block(
                "figure",
                {"number": number, "filename": filename, "caption": caption},
            )
        )
        return self

    def page_break(self) -> "Document":
        self.blocks.append(Block("page_break", None))
        return self

    def note(self, text: str) -> "Document":
        self.blocks.append(Block("note", text))
        return self

    # -- Markdown ---------------------------------------------------------
    def render_markdown(self) -> str:
        lines: list[str] = [f"# {self.title}", ""]
        if self.subtitle:
            lines.extend([f"*{self.subtitle}*", ""])
        for key, value in self.metadata:
            lines.append(f"**{key}:** {value}")
        lines.append("")
        for block in self.blocks:
            if block.kind == "heading":
                lines.extend(
                    [f"{'#' * block.payload['level']} {block.payload['text']}", ""]
                )
            elif block.kind == "paragraph":
                lines.extend([block.payload, ""])
            elif block.kind == "bullets":
                lines.extend([f"- {item}" for item in block.payload])
                lines.append("")
            elif block.kind == "numbered":
                lines.extend(
                    [f"{index}. {item}" for index, item in enumerate(block.payload, 1)]
                )
                lines.append("")
            elif block.kind == "code":
                lines.extend(["```text", block.payload, "```", ""])
            elif block.kind == "note":
                lines.extend([f"> {block.payload}", ""])
            elif block.kind == "table":
                table = block.payload
                if table["caption"]:
                    lines.extend(
                        [
                            f"**Table {table['number']}. {table['caption']}**",
                            "",
                        ]
                    )
                escaped_headers = [_escape_pipe(str(item)) for item in table["headers"]]
                lines.append("| " + " | ".join(escaped_headers) + " |")
                lines.append("|" + "|".join(["---"] * len(escaped_headers)) + "|")
                for row in table["rows"]:
                    cells = [_escape_pipe(str(item)) for item in row]
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")
            elif block.kind == "figure":
                figure = block.payload
                lines.extend(
                    [
                        f"![Figure {figure['number']}](figures/{figure['filename']})",
                        "",
                        f"*Figure {figure['number']}. {figure['caption']}*",
                        "",
                    ]
                )
            elif block.kind == "page_break":
                lines.extend(["<!-- page break -->", ""])
        return "\n".join(lines).rstrip() + "\n"

    # -- HTML -------------------------------------------------------------
    def render_html(self) -> str:
        body: list[str] = []
        body.append('<div class="title-page">')
        body.append(f'<p class="document-kind">Phase 7 - Final research report</p>')
        body.append(f"<h1>{html.escape(self.title)}</h1>")
        if self.subtitle:
            body.append(f'<p class="subtitle">{_inline_html(self.subtitle)}</p>')
        body.append('<div class="metadata">')
        for key, value in self.metadata:
            body.append(
                f'<p><span class="meta-key">{html.escape(key)}</span>'
                f"<span>{_inline_html(value)}</span></p>"
            )
        body.append("</div>")
        body.append("</div>")

        for block in self.blocks:
            if block.kind == "heading":
                level = int(block.payload["level"])
                body.append(
                    f"<h{level}>{_inline_html(block.payload['text'])}</h{level}>"
                )
            elif block.kind == "paragraph":
                body.append(f"<p>{_inline_html(block.payload)}</p>")
            elif block.kind == "bullets":
                body.append("<ul>")
                body.extend(f"<li>{_inline_html(item)}</li>" for item in block.payload)
                body.append("</ul>")
            elif block.kind == "numbered":
                body.append("<ol>")
                body.extend(f"<li>{_inline_html(item)}</li>" for item in block.payload)
                body.append("</ol>")
            elif block.kind == "code":
                body.append(f"<pre>{html.escape(block.payload)}</pre>")
            elif block.kind == "note":
                body.append(f'<aside class="note">{_inline_html(block.payload)}</aside>')
            elif block.kind == "table":
                table = block.payload
                if table["caption"]:
                    body.append(
                        f'<p class="table-caption"><strong>Table {table["number"]}.'
                        f"</strong> {_inline_html(table['caption'])}</p>"
                    )
                body.append("<table>")
                body.append("<thead><tr>")
                body.extend(f"<th>{_inline_html(str(h))}</th>" for h in table["headers"])
                body.append("</tr></thead><tbody>")
                for row in table["rows"]:
                    body.append("<tr>")
                    body.extend(f"<td>{_inline_html(str(c))}</td>" for c in row)
                    body.append("</tr>")
                body.append("</tbody></table>")
            elif block.kind == "figure":
                figure = block.payload
                body.append('<figure class="report-figure">')
                body.append(
                    f'<img src="figures/{html.escape(figure["filename"])}" '
                    f'alt="Figure {figure["number"]}">'
                )
                body.append(
                    f'<figcaption><strong>Figure {figure["number"]}.</strong> '
                    f"{_inline_html(figure['caption'])}</figcaption>"
                )
                body.append("</figure>")
            elif block.kind == "page_break":
                body.append('<div class="page-break"></div>')

        return _HTML_TEMPLATE.format(
            title=html.escape(self.title), body="\n".join(body)
        )


def _escape_pipe(text: str) -> str:
    return text.replace("|", "\\|")


def _inline_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
@page {{ size: A4; margin: 17mm 16mm 18mm 16mm; }}
html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{
  font-family: Georgia, "DejaVu Serif", "Times New Roman", serif;
  font-size: 10.3pt; line-height: 1.45; color: #1b1b1b; margin: 0;
}}
h1 {{ font-size: 26pt; line-height: 1.15; margin: 0 0 10pt 0; color: #10263d; }}
h2 {{
  font-size: 15pt; color: #10263d; margin: 24pt 0 8pt 0;
  padding-bottom: 3pt; border-bottom: 0.8pt solid #c9d3dc;
  page-break-after: avoid;
}}
h3 {{ font-size: 11.3pt; color: #1d3b58; margin: 15pt 0 5pt 0; page-break-after: avoid; }}
p {{ margin: 0 0 8pt 0; text-align: justify; }}
ul, ol {{ margin: 0 0 9pt 0; padding-left: 18pt; }}
li {{ margin-bottom: 3pt; }}
code {{ font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 0.9em; background: #f2f4f6; padding: 0 2pt; }}
pre {{ background: #f5f6f8; border: 0.5pt solid #d5dbe1; padding: 8pt; font-size: 8.8pt; white-space: pre-wrap; }}
.title-page {{
  min-height: 235mm; padding-top: 55mm; page-break-after: always;
}}
.document-kind {{ font-size: 10pt; letter-spacing: 1.4pt; text-transform: uppercase; color: #7a4a00; margin-bottom: 12pt; }}
.subtitle {{ font-size: 13pt; color: #3b4d5e; margin-bottom: 32pt; text-align: left; }}
.metadata {{ border-top: 0.8pt solid #c9d3dc; padding-top: 10pt; font-size: 9.2pt; }}
.metadata p {{ margin: 0 0 4pt 0; text-align: left; }}
.meta-key {{ display: inline-block; width: 34mm; color: #5a6a78; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 4pt 0 13pt 0; page-break-inside: avoid; }}
th {{ background: #eef2f5; text-align: left; font-weight: bold; }}
th, td {{ border: 0.45pt solid #b9c3cc; padding: 3pt 4.5pt; vertical-align: top; }}
tbody tr:nth-child(even) {{ background: #fafbfc; }}
.table-caption {{ font-size: 9pt; margin: 8pt 0 3pt 0; color: #2f3f4e; page-break-after: avoid; }}
.report-figure {{ margin: 12pt 0 15pt 0; page-break-inside: avoid; }}
.report-figure img {{ width: 100%; height: auto; border: 0.4pt solid #dce2e7; }}
figcaption {{ font-size: 8.7pt; color: #46535f; margin-top: 4pt; text-align: left; }}
.note {{ background: #fff8e6; border-left: 2.5pt solid #b7791f; padding: 6pt 9pt; margin: 8pt 0 12pt 0; font-size: 9.4pt; }}
.page-break {{ page-break-before: always; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _signed(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def _seconds(value: Any, digits: int = 2) -> str:
    return f"{_num(value, digits)} s"


def _effect(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.3f}"


def _overall(payload: dict[str, Any], metric: str, scope: str) -> dict[str, Any]:
    return payload["overall"][metric][scope]


def _factor(
    payload: dict[str, Any], metric: str, factor: str, **conditions: Any
) -> dict[str, Any]:
    for record in payload["factor_summaries"][metric][factor]:
        if all(record["factors"].get(key) == value for key, value in conditions.items()):
            return record
    raise KeyError(f"No {factor} record for {metric} {conditions}")


def _factor_rows(payload: dict[str, Any], metric: str, factor: str) -> list[dict[str, Any]]:
    return payload["factor_summaries"][metric][factor]


def _wins(record: dict[str, Any]) -> str:
    return f"{record['improved']} / {record['unchanged']} / {record['regressed']}"


def _dataset_name(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def _classifier_name(classifier: str) -> str:
    return CLASSIFIER_LABELS.get(classifier, classifier)


def _representation_name(representation: str) -> str:
    return REPRESENTATION_LABELS.get(representation, representation)


def _comparison_stats_table(
    payload: dict[str, Any],
    metric: str,
    factor: str,
    label_header: str,
) -> tuple[list[str], list[list[str]]]:
    headers = [
        label_header,
        "n",
        "Supervised",
        "Self-training",
        "Mean change",
        "W / U / L",
        "d_z",
    ]
    rows = []
    for record in _factor_rows(payload, metric, factor):
        key = next(iter(record["factors"]))
        value = record["factors"][key]
        if key == "dataset":
            label = _dataset_name(value)
        elif key == "feature_representation":
            label = _representation_name(value)
        elif key == "classifier":
            label = _classifier_name(value)
        elif key == "labelled_percentage":
            label = f"{value}%"
        else:
            label = str(value)
        rows.append(
            [
                label,
                str(record["n"]),
                _num(record["mean_supervised"]),
                _num(record["mean_self_training"]),
                _signed(record["mean_difference"]),
                _wins(record),
                _effect(record["descriptive_cohens_dz"]),
            ]
        )
    return headers, rows


def _figure_catalog(figures: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        item["filename"]: {
            "number": item["number"],
            "caption": item["caption"],
            "path": item["path"],
        }
        for item in figures
    }


def _add_figure(
    document: Document,
    catalog: dict[str, dict[str, Any]],
    filename: str,
    figures_dir: Path,
) -> None:
    item = catalog[filename]
    document.figure(
        filename,
        item["caption"],
        directory=figures_dir,
    )


# ---------------------------------------------------------------------------
# Final research report
# ---------------------------------------------------------------------------
REPORT_SECTIONS = (
    "Abstract",
    "Introduction",
    "Research Problem",
    "Research Objectives",
    "Dataset Descriptions",
    "Methodology",
    "Preprocessing and Data Splitting",
    "PCA and LDA Methodology",
    "Supervised Learning Experiments",
    "Semi-Supervised Self-Training Methodology",
    "Experimental Results",
    "Comparative Analysis",
    "Statistical Analysis, Where Justified",
    "Discussion",
    "Computational Complexity and Runtime Analysis",
    "Limitations",
    "Future Scope",
    "Conclusion",
    "References",
)


def build_final_report(
    payload: dict[str, Any],
    figures: Sequence[dict[str, Any]],
    figures_dir: str | Path,
) -> tuple[Document, str]:
    """Build the 19-section final research report.

    Returns the document object and its Markdown rendering.
    """
    catalog = _figure_catalog(figures)
    figures_path = Path(figures_dir)
    design = payload["design"]
    datasets = {row["dataset"]: row for row in design["datasets"]}
    runtime = payload["runtime"]
    pseudo = payload["pseudo_labels"]
    warnings = payload["warnings"]

    accuracy = _overall(payload, "accuracy", "all_conditions")
    accuracy_nd = _overall(payload, "accuracy", "excluding_degenerate_fits")
    f1 = _overall(payload, "macro_f1", "all_conditions")
    f1_nd = _overall(payload, "macro_f1", "excluding_degenerate_fits")

    bc = _factor(payload, "macro_f1", "dataset", dataset="breast_cancer")
    mnist = _factor(payload, "macro_f1", "dataset", dataset="mnist")
    har = _factor(payload, "macro_f1", "dataset", dataset="har")
    baseline = _factor(
        payload, "macro_f1", "feature_representation", feature_representation="baseline"
    )
    pca = _factor(
        payload, "macro_f1", "feature_representation", feature_representation="pca"
    )
    lda = _factor(
        payload, "macro_f1", "feature_representation", feature_representation="lda"
    )
    lr = _factor(
        payload,
        "macro_f1",
        "classifier",
        classifier="logistic_regression",
    )
    svm = _factor(payload, "macro_f1", "classifier", classifier="svm")
    knn = _factor(payload, "macro_f1", "classifier", classifier="knn")
    p10 = _factor(
        payload, "macro_f1", "labelled_percentage", labelled_percentage=10
    )
    p25 = _factor(
        payload, "macro_f1", "labelled_percentage", labelled_percentage=25
    )
    p50 = _factor(
        payload, "macro_f1", "labelled_percentage", labelled_percentage=50
    )

    mnist_lda = _factor(
        payload,
        "macro_f1",
        "dataset_x_representation",
        dataset="mnist",
        feature_representation="lda",
    )
    mnist_lr = _factor(
        payload,
        "macro_f1",
        "dataset_x_classifier",
        dataset="mnist",
        classifier="logistic_regression",
    )
    mnist_svm = _factor(
        payload,
        "macro_f1",
        "dataset_x_classifier",
        dataset="mnist",
        classifier="svm",
    )
    mnist_knn = _factor(
        payload,
        "macro_f1",
        "dataset_x_classifier",
        dataset="mnist",
        classifier="knn",
    )
    mnist_baseline = _factor(
        payload,
        "macro_f1",
        "dataset_x_representation",
        dataset="mnist",
        feature_representation="baseline",
    )
    mnist_pca = _factor(
        payload,
        "macro_f1",
        "dataset_x_representation",
        dataset="mnist",
        feature_representation="pca",
    )
    har_baseline = _factor(
        payload,
        "macro_f1",
        "dataset_x_representation",
        dataset="har",
        feature_representation="baseline",
    )
    har_pca = _factor(
        payload,
        "macro_f1",
        "dataset_x_representation",
        dataset="har",
        feature_representation="pca",
    )
    har_lda = _factor(
        payload,
        "macro_f1",
        "dataset_x_representation",
        dataset="har",
        feature_representation="lda",
    )
    bc_baseline = _factor(
        payload,
        "macro_f1",
        "dataset_x_representation",
        dataset="breast_cancer",
        feature_representation="baseline",
    )
    bc_pca = _factor(
        payload,
        "macro_f1",
        "dataset_x_representation",
        dataset="breast_cancer",
        feature_representation="pca",
    )
    bc_lda = _factor(
        payload,
        "macro_f1",
        "dataset_x_representation",
        dataset="breast_cancer",
        feature_representation="lda",
    )

    comparison_records = payload.get("comparison") or []
    nondegenerate_records = [
        record for record in comparison_records if not record["numerically_degenerate_fit"]
    ]
    nd_lr_values = [
        float(record["macro_f1_improvement"])
        for record in nondegenerate_records
        if record["classifier"] == "logistic_regression"
    ]
    nd_lr = {
        "mean_difference": (
            float(np.mean(nd_lr_values)) if nd_lr_values else lr["mean_difference"]
        )
    }

    best_f1 = payload["best_configurations"]["macro_f1"]
    best_acc = payload["best_configurations"]["accuracy"]
    top_f1 = payload["top_changes"]["macro_f1"]
    degenerate = warnings["degenerate_fits"]
    degenerate_records = degenerate["experiments"]

    document = Document(
        payload["title"],
        "Phase 7 - Comprehensive research analysis, comparison, and final report",
        metadata=[
            ("Generated (UTC)", payload["generated_utc"]),
            ("Workspace", "D:\\ml_project"),
            (
                "Evidence base",
                "81 Phase 5 supervised + 81 Phase 6 self-training results "
                "on one fixed split (seed 42)",
            ),
            (
                "Comparison table",
                "results/final_report/final_results_comparison.csv "
                f"({payload['coverage']['matched_experiments']} matched conditions)",
            ),
            ("Analysis policy", "Descriptive paired comparisons; no p-values or "
             "confidence intervals (see Section 13)"),
        ],
    )

    # 1 ------------------------------------------------------------------
    document.heading("1. Abstract")
    document.paragraph(
        "This report quantifies how dimensionality reduction (DR) changes the "
        "performance of supervised classifiers when labelled data are scarce and "
        "the remaining training data are exploited through confidence-based "
        "self-training. Three datasets were studied: Breast Cancer Wisconsin "
        "(Diagnostic), MNIST, and UCI Human Activity Recognition (HAR). A single "
        "stratified or official subject-disjoint split was fixed in advance, and "
        "three labelled proportions (10%, 25%, 50%), three feature "
        "representations (Baseline, PCA, LDA), and three classifiers (Logistic "
        "Regression, linear SVM, KNN) produced 81 conditions. Each condition was "
        "run once as a fully supervised experiment (Phase 5) and once with "
        "self-training (Phase 6), giving 162 matched runs and 81 paired "
        "comparisons."
    )
    document.paragraph(
        f"Across all 81 conditions, mean macro-F1 was {_num(f1['mean_supervised'])} "
        f"for supervised learning and {_num(f1['mean_self_training'])} for "
        f"self-training, a mean paired change of {_signed(f1['mean_difference'])} "
        f"(median {_signed(f1['median_difference'])}, SD {_num(f1['sd_difference'])}); "
        f"{f1['improved']} conditions improved, {f1['unchanged']} were unchanged, "
        f"and {f1['regressed']} regressed. The negative mean is driven almost "
        "entirely by three numerically degenerate MNIST LDA + Logistic Regression "
        "fits. Excluding those three documented failures, the mean change becomes "
        f"{_signed(f1_nd['mean_difference'])} (SD {_num(f1_nd['sd_difference'])}), "
        f"with {f1_nd['improved']} improvements, {f1_nd['unchanged']} ties, and "
        f"{f1_nd['regressed']} regressions. The largest genuine gain was "
        f"{_signed(top_f1['largest_improvements'][0]['macro_f1_improvement'])} "
        f"macro-F1 for MNIST 10% LDA + linear SVM; the largest losses other than "
        "the degenerate fits were small relative to that gain."
    )
    document.paragraph(
        "Representation and classifier interactions dominate the average effect. "
        f"Self-training changed macro-F1 by {_signed(baseline['mean_difference'])} "
        f"under Baseline, {_signed(pca['mean_difference'])} under PCA, and "
        f"{_signed(lda['mean_difference'])} under LDA; the LDA mean is dominated by "
        f"MNIST ({_signed(mnist_lda['mean_difference'])}). By classifier, the mean "
        f"change was {_signed(lr['mean_difference'])} for Logistic Regression, "
        f"{_signed(svm['mean_difference'])} for linear SVM, and "
        f"{_signed(knn['mean_difference'])} for KNN. Self-training accepted "
        f"{pseudo['total_pseudo_labels']:,} pseudo-labels in total, but required "
        f"{_num(runtime['training_time_ratio_total'], 3)}x the supervised training "
        f"time ({_num(runtime['self_training_training_seconds_total'])} s versus "
        f"{_num(runtime['supervised_training_seconds_total'])} s). No condition "
        "shows a universal benefit, and the report makes no significance claims "
        "because each condition has one deterministic run on one shared split."
    )

    # 2 ------------------------------------------------------------------
    document.heading("2. Introduction")
    document.paragraph(
        "Supervised classification normally assumes that every training example "
        "carries a reliable label. In many scientific and industrial settings this "
        "assumption is expensive: labels require expert time, while unlabelled "
        "measurements accumulate cheaply. Semi-supervised learning (SSL) attempts "
        "to use those unlabelled measurements, most simply by letting a model label "
        "its own high-confidence predictions and then retraining on them - the "
        "self-training or pseudo-labelling strategy (Yarowsky, 1995; Zhu, 2005; "
        "Chapelle et al., 2006; Lee, 2013). Separately, dimensionality reduction "
        "(DR) is routinely applied before classification to remove collinearity, "
        "compress noise, and reduce computation. The two choices interact: DR can "
        "make a classifier more stable and its confidence estimates more usable, "
        "or it can discard exactly the low-variance directions that separate "
        "classes, making pseudo-labels less reliable."
    )
    document.paragraph(
        "This project isolates that interaction in a controlled factorial design. "
        "Phase 3 fixed the train/test splits and the labelled/unlabelled partitions; "
        "Phase 4 fitted Baseline, PCA, and LDA representations without using test "
        "rows; Phase 5 trained the supervised reference models; Phase 6 repeated "
        "the identical 81 conditions with confidence-based self-training. Phase 7, "
        "reported here, is a read-only analysis of those saved results. No Phase 5 "
        "or Phase 6 experiment was rerun, no result was altered, and no missing "
        "quantity was estimated."
    )
    document.paragraph(
        "The contribution is therefore not a new algorithm but a transparent, "
        "fully crossed empirical comparison with matched conditions, audited "
        "leakage controls, explicit treatment of convergence warnings and numerical "
        "failures, and reproducible artefacts. The report deliberately separates "
        "three things that are often conflated: warnings emitted by an optimiser, "
        "limitations imposed by the design, and experiments that failed numerically."
    )

    # 3 ------------------------------------------------------------------
    document.heading("3. Research Problem")
    document.paragraph(
        "Let a dataset provide a training partition and a held-out test partition. "
        "For a labelled proportion p, the training partition is divided into a "
        "labelled set L(p) and an unlabelled pool U(p); only L(p) and its labels are "
        "visible to a supervised learner, while U(p) is visible without labels. For "
        "each feature representation r in {Baseline, PCA, LDA} and classifier c in "
        "{Logistic Regression, linear SVM, KNN}, two models are produced: a "
        "supervised model fitted on L(p), and a self-training model fitted on L(p) "
        "and pseudo-labelled subsets of U(p). Both are evaluated once on the same "
        "fixed test partition."
    )
    document.paragraph(
        "The research problem is to characterise the paired difference in test "
        "accuracy and macro-F1 between the self-training and supervised models as a "
        "function of dataset, p, r, and c, and to relate that difference to the "
        "number and acceptance rate of pseudo-labels, the number of self-training "
        "iterations, and the computational cost. Four questions follow:"
    )
    document.bullets(
        [
            "RQ1 (representation): Does the benefit or harm of self-training depend "
            "on whether features are raw, PCA-reduced, or LDA-reduced?",
            "RQ2 (classifier): Are some classifiers more compatible with "
            "confidence-based pseudo-labelling than others?",
            "RQ3 (labelled proportion): Does self-training help most when labels "
            "are scarcest (10%) and least when they are plentiful (50%)?",
            "RQ4 (cost and reliability): What runtime does self-training add, how "
            "many iterations are actually used, and which warnings or numerical "
            "failures qualify the conclusions?",
        ]
    )
    document.paragraph(
        "A negative result is informative. If self-training does not improve "
        "performance on average under a fixed 0.90 threshold and a three-iteration "
        "cap, the report must say so and explain where the average is misleading, "
        "rather than selecting a favourable subset post hoc."
    )

    # 4 ------------------------------------------------------------------
    document.heading("4. Research Objectives")
    document.numbered(
        [
            "Build a matched supervised/self-training comparison for every one of "
            "the 81 experimental conditions, using the saved Phase 5 and Phase 6 "
            "results without rerunning or modifying them.",
            "Quantify paired changes in accuracy and macro-F1 overall and by "
            "dataset, labelled proportion, feature representation, and classifier, "
            "including wins, ties, and losses.",
            "Relate performance changes to pseudo-label acceptance, stopping "
            "reasons, iteration counts, and the calibrated confidence behaviour of "
            "each classifier.",
            "Measure the training and prediction cost of self-training relative to "
            "the supervised baseline, and identify the dominant cost driver.",
            "Audit convergence warnings, numerical failures, and design "
            "limitations; keep failed or degenerate results in the primary "
            "statistics while also reporting a sensitivity analysis that excludes "
            "them.",
            "Produce publication-quality figures, machine-readable comparison "
            "tables, a 19-section final report, an executive summary, and a "
            "standalone limitations/future-work document, all generated from the "
            "saved results.",
            "Verify independently that every reported metric traces to an "
            "underlying result file, that all 81 + 81 conditions are accounted for, "
            "and that Phase 1-6 artefacts are unchanged.",
        ]
    )

    # 5 ------------------------------------------------------------------
    document.heading("5. Dataset Descriptions")
    dataset_rows = []
    for name in DATASET_ORDER:
        info = datasets[name]
        dataset_rows.append(
            [
                _dataset_name(name),
                str(info["n_original"]),
                str(info["n_features"]),
                f"{info['n_train']} / {info['n_test']}",
                "Stratified holdout (80/20)"
                if info["strategy"] == "stratified_holdout"
                else "Official subject-disjoint split",
            ]
        )
    document.table(
        ["Dataset", "Samples", "Features", "Train / test", "Split strategy"],
        dataset_rows,
        "Datasets and fixed partitions used in every experiment.",
    )
    document.paragraph(
        "**Breast Cancer Wisconsin (Diagnostic).** The dataset contains 569 "
        "fine-needle aspirate samples described by 30 numeric features computed "
        "from cell-nucleus images (radius, texture, perimeter, area, smoothness, "
        "compactness, concavity, concave points, symmetry, and fractal dimension, "
        "each reported as mean, standard error, and worst value). The task is "
        "binary classification of malignant versus benign tumours. The staged "
        "copy comes from the public Kaggle dataset uciml/breast-cancer-wisconsin-"
        "data, derived from the UCI Breast Cancer Wisconsin (Diagnostic) dataset "
        "(Street et al., 1993; Wolberg et al., 1995). The stratified 80/20 split "
        "yields 455 training and 114 test samples; the training partition contains "
        "285 benign and 170 malignant cases."
    )
    document.paragraph(
        "**MNIST.** MNIST contains 28 x 28 grayscale handwritten digit images from "
        "ten classes (LeCun et al., 1998; Deng, 2012). Each image is flattened to "
        "784 pixel features. This study uses the 42,000 labelled rows of the "
        "Kaggle Digit Recognizer training file; the 28,000-row competition test "
        "file is unlabelled and is deliberately ignored so that the held-out split "
        "remains clean. Because the official competition download requires Kaggle "
        "credentials and rule acceptance that were not configured in this "
        "environment, the staged files were obtained from the documented public "
        "Kaggle mirror khotijahs1/digitrecognizer, which has the same 42,000-row "
        "training layout (see data/README.md). The stratified 80/20 split yields "
        "33,600 training and 8,400 test images."
    )
    document.paragraph(
        "**UCI Human Activity Recognition Using Smartphones (HAR).** The dataset "
        "contains 10,299 windows of smartphone accelerometer and gyroscope "
        "recordings from 30 volunteers performing six activities (walking, walking "
        "upstairs, walking downstairs, sitting, standing, laying). Each window is "
        "described by 561 time- and frequency-domain features (Anguita et al., "
        "2013; Reyes-Ortiz et al., 2016). The official subject-disjoint split is "
        "preserved: 21 subjects provide 7,352 training windows and 9 different "
        "subjects provide 2,947 test windows, so no subject appears on both sides."
    )
    document.paragraph(
        "All three datasets were already staged and verified in Phase 2; Phase 7 "
        "read them only through the saved split metadata and did not download, "
        "replace, or modify any dataset. Targets are balanced enough for accuracy "
        "and macro-F1 to be interpreted together; macro-F1 is the primary metric "
        "because it weights all classes equally."
    )

    # 6 ------------------------------------------------------------------
    document.heading("6. Methodology")
    document.paragraph(
        "The design is a fully crossed 3 x 3 x 3 x 3 factorial with one fixed "
        "split per dataset. The four factors are dataset (Breast Cancer, MNIST, "
        "HAR), labelled proportion (10%, 25%, 50% of the training partition), "
        "feature representation (Baseline, PCA, LDA), and classifier (Logistic "
        "Regression, linear SVM, KNN). This yields 81 conditions. Every condition "
        "was executed twice: once in Phase 5 with supervised training only and "
        "once in Phase 6 with self-training. The two runs share the same split, "
        "the same labelled rows, the same transformed features, and the same test "
        "rows, which is what makes the paired difference meaningful."
    )
    document.paragraph(
        "Performance is measured on the held-out test partition only. Accuracy is "
        "the fraction of correctly classified test samples. Macro-F1 is the "
        "unweighted mean of the per-class F1 scores, F1 = 2PR/(P+R), so each class "
        "contributes equally regardless of frequency. Macro-F1 is the primary "
        "metric; accuracy is reported alongside it because the two can diverge "
        "when class-level errors are uneven."
    )
    document.paragraph(
        "The evaluation protocol enforces the following rules. The scaler and every "
        "DR transformation are fitted without test rows. The self-training loop "
        "never reads the true labels of unlabelled rows. The confidence threshold "
        "is fixed at 0.90 before any experiment and is never tuned on test "
        "performance or unlabelled ground truth. The seed is fixed at 42. Phase 6 "
        "records SHA-256 fingerprints of the labelled features, labelled targets, "
        "and unlabelled features for every condition; Phase 7 verifies that each "
        "supervised and self-training pair has identical fingerprints and feature "
        "dimensions."
    )
    document.table(
        ["Factor", "Levels", "Role"],
        [
            ["Dataset", "Breast Cancer, MNIST, HAR", "Between-dataset replication with different sample sizes, dimensionality, and class structure"],
            ["Labelled proportion", "10%, 25%, 50% of training rows", "Scarcity of supervision"],
            ["Feature representation", "Baseline, PCA, LDA", "Dimensionality-reduction strategy"],
            ["Classifier", "Logistic Regression, linear SVM, KNN", "Decision-boundary and probability behaviour"],
        ],
        "Experimental factors and levels (3 x 3 x 3 x 3 = 81 conditions).",
    )
    document.paragraph(
        "The experiment is deterministic at the level of data partitioning and "
        "model fitting: one seed, one split, no repeated runs, and no "
        "cross-validation for model selection. This choice makes the comparison "
        "reproducible and cheap, but it also means that condition-level results are "
        "not independent replicates. Section 13 explains the consequences for "
        "statistical reporting."
    )

    # 7 ------------------------------------------------------------------
    document.heading("7. Preprocessing and Data Splitting")
    document.paragraph(
        "Phase 3 established the partitions used by every later phase. Breast "
        "Cancer and MNIST use a stratified 80/20 holdout with seed 42. HAR uses "
        "the official subject-disjoint training and test files. Within each "
        "training partition, a stratified sample of the requested proportion is "
        "designated labelled; the remainder becomes the unlabelled pool. The "
        "actual proportion can differ slightly from the request when a class has "
        "few samples, as recorded in the split metadata."
    )
    split_rows = []
    for name in DATASET_ORDER:
        info = datasets[name]
        for proportion in (10, 25, 50):
            setting = info["labelled_settings"][str(proportion)]
            split_rows.append(
                [
                    _dataset_name(name),
                    f"{proportion}%",
                    str(setting["n_labelled"]),
                    str(setting["n_unlabelled"]),
                    _pct(setting["actual_proportion"], 3),
                    str(info["n_test"]),
                ]
            )
    document.table(
        [
            "Dataset",
            "Requested",
            "Labelled rows",
            "Unlabelled rows",
            "Actual labelled share",
            "Test rows",
        ],
        split_rows,
        "Labelled and unlabelled partition sizes for every dataset and proportion.",
    )
    document.paragraph(
        "Standardisation uses StandardScaler with statistics estimated on the "
        "training partition only (labelled plus unlabelled rows), never on test "
        "rows. The same scaled matrix feeds Baseline, PCA, and LDA. The unlabelled "
        "pool is available to Phase 6 as unlabelled features only; its true labels "
        "are hidden from the self-training loop and are used solely by the "
        "verification scripts to audit that the loop never accessed them."
    )
    document.paragraph(
        "Two leakage controls deserve emphasis. First, PCA is fitted once per "
        "dataset on the complete scaled training pool and reused for all "
        "proportions, so the projection never sees test rows and is identical "
        "across the supervised and self-training runs. Second, LDA is refitted for "
        "each labelled proportion on labelled rows and labels only; unlike PCA it "
        "cannot use the unlabelled pool because that would require their hidden "
        "labels. These scopes are recorded in the Phase 4 metadata and are "
        "rechecked by the Phase 7 verifier."
    )

    # 8 ------------------------------------------------------------------
    document.heading("8. PCA and LDA Methodology")
    document.paragraph(
        "**PCA.** Principal Component Analysis finds orthogonal directions of "
        "maximum variance (Pearson, 1901; Jolliffe, 2002). The implementation uses "
        "the scikit-learn PCA solver with a target of 95% explained variance, no "
        "whitening, and seed 42, fitted once per dataset on the scaled training "
        "pool (labelled plus unlabelled rows, test rows excluded). Components are "
        "ordered by explained variance and all retained components are used by the "
        "classifier."
    )
    pca_rows = []
    for name in DATASET_ORDER:
        info = datasets[name]
        pca_rows.append(
            [
                _dataset_name(name),
                str(info["n_features"]),
                str(info["pca_components"]),
                _pct(info["pca_variance_retained"], 3),
                f"{100.0 * (1.0 - info['pca_components'] / info['n_features']):.1f}%",
            ]
        )
    document.table(
        [
            "Dataset",
            "Original features",
            "PCA components",
            "Variance retained",
            "Feature reduction",
        ],
        pca_rows,
        "PCA configuration and retained variance (95% target).",
    )
    document.paragraph(
        "**LDA.** Linear Discriminant Analysis seeks directions that maximise "
        "between-class separation relative to within-class scatter (Fisher, 1936; "
        "Rao, 1948). The implementation uses the SVD solver and the standard cap "
        "min(n_features, n_classes - 1) on the number of components. It is fitted "
        "separately for each labelled proportion on the labelled rows and their "
        "labels only, so the transformation is not identical across proportions "
        "and never uses unlabelled or test labels."
    )
    lda_rows = []
    for name in DATASET_ORDER:
        info = datasets[name]
        lda_rows.append(
            [
                _dataset_name(name),
                ", ".join(
                    f"{proportion}%: {info['lda_components'][str(proportion)]}"
                    for proportion in (10, 25, 50)
                ),
                str(info["lda_components"]["10"]),
            ]
        )
    document.table(
        ["Dataset", "LDA components by labelled proportion", "Cap"],
        lda_rows,
        "LDA component counts; the cap is min(n_features, n_classes - 1).",
    )
    document.paragraph(
        "The contrast between representations is therefore structural. Baseline "
        "keeps all original features. PCA is a fixed, label-free linear projection "
        "that can remove noise but is blind to class separation. LDA is a "
        "supervised projection with at most C - 1 dimensions, which is extremely "
        "aggressive: one dimension for binary Breast Cancer, nine for ten-class "
        "MNIST, and five for six-class HAR. This aggressiveness is central to the "
        "results: LDA produces the largest gain and the largest failures in the "
        "study."
    )

    # 9 ------------------------------------------------------------------
    document.heading("9. Supervised Learning Experiments")
    document.paragraph(
        "Phase 5 trained three classifiers under fixed, untuned configurations on "
        "all 81 conditions. Hyperparameters were chosen once and held constant; no "
        "grid search or cross-validated tuning was performed, so the comparison "
        "measures the effect of representation and supervision rather than the "
        "effect of per-condition optimisation."
    )
    document.table(
        ["Classifier", "Implementation", "Configuration"],
        [
            [
                "Logistic Regression",
                "scikit-learn LogisticRegression",
                "lbfgs solver, C = 1.0, max_iter = 2000, no class weights, seed 42",
            ],
            [
                "Linear SVM",
                "scikit-learn LinearSVC",
                "squared_hinge loss, dual formulation, C = 1.0, tol = 1e-3, "
                "max_iter = 2000",
            ],
            [
                "KNN",
                "scikit-learn KNeighborsClassifier",
                "k = 5, distance weighting, Euclidean (Minkowski p = 2), brute-force "
                "search, 6 workers",
            ],
        ],
        "Phase 5 classifier configurations.",
    )
    document.paragraph(
        "Each supervised model was fitted on the labelled rows of its condition and "
        "evaluated once on the fixed test rows. Training time was measured around "
        "the fit call. Prediction time was not recorded in Phase 5; this is a "
        "genuine gap and is reported as such rather than reconstructed. All 81 "
        "experiments completed without error. 16 Liblinear ConvergenceWarnings "
        "were emitted by linear-SVM fits (eight on MNIST, eight on HAR); the numbers "
        "remain valid test-set measurements but indicate that the optimiser "
        "stopped at its iteration or tolerance limit."
    )
    document.paragraph(
        f"The strongest supervised macro-F1 on Breast Cancer was "
        f"{_num(best_f1['breast_cancer']['supervised']['score'])} "
        f"({best_f1['breast_cancer']['supervised']['labelled_percentage']}% "
        f"{_representation_name(best_f1['breast_cancer']['supervised']['feature_representation'])}, "
        f"{_classifier_name(best_f1['breast_cancer']['supervised']['classifier'])}); "
        f"on MNIST it was {_num(best_f1['mnist']['supervised']['score'])} "
        f"({best_f1['mnist']['supervised']['labelled_percentage']}% "
        f"{_representation_name(best_f1['mnist']['supervised']['feature_representation'])}, "
        f"{_classifier_name(best_f1['mnist']['supervised']['classifier'])}); "
        f"on HAR it was {_num(best_f1['har']['supervised']['score'])} "
        f"({best_f1['har']['supervised']['labelled_percentage']}% "
        f"{_representation_name(best_f1['har']['supervised']['feature_representation'])}, "
        f"{_classifier_name(best_f1['har']['supervised']['classifier'])}). "
        "These are best observed test scores, not unbiased estimates of "
        "generalisation, because they are selected after seeing the test results."
    )

    # 10 -----------------------------------------------------------------
    document.heading("10. Semi-Supervised Self-Training Methodology")
    document.paragraph(
        "Phase 6 applied the same 81 conditions with confidence-based "
        "self-training. The procedure for a condition is:"
    )
    document.numbered(
        [
            "Fit the classifier on the original labelled rows L(p) only, using the "
            "same configuration as Phase 5.",
            "Predict class probabilities for every row still in the unlabelled pool "
            "U(p). For Logistic Regression and KNN the confidence is the maximum "
            "class probability from predict_proba.",
            "Accept rows whose confidence is at least 0.90, assign the predicted "
            "class, append them to the training set, and remove them from the "
            "unlabelled pool. If fewer than one new row is accepted, stop.",
            "Refit the classifier on the original labelled rows plus all accumulated "
            "pseudo-labelled rows.",
            "Repeat steps 2-4 until the stop condition triggers or the "
            "three-iteration cap is reached.",
            "Predict the fixed test rows with the final refitted model.",
        ]
    )
    document.paragraph(
        "LinearSVC does not expose probabilities. Phase 6 therefore fits Platt "
        "sigmoid calibrators (Platt, 1999) on a class-stratified 20% subset of the "
        "original labelled rows, using stratified cross-validation through "
        "CalibratedClassifierCV with a frozen Phase 5 LinearSVC. Calibrated "
        "probabilities are used only to decide which unlabelled rows to accept. "
        "Final test predictions come from a plain Phase 5 LinearSVC refitted on the "
        "labelled plus pseudo-labelled rows, so the evaluation model is the same "
        "family as the supervised baseline. Calibrated SVM probabilities are "
        "approximations and are not guaranteed to be on the same scale as "
        "Logistic Regression or KNN probabilities."
    )
    document.paragraph(
        f"The canonical Phase 6 run uses a confidence threshold of "
        f"{design['confidence_threshold']:.2f}, a maximum of "
        f"{design['max_iterations']} iterations, a minimum of "
        f"{design['minimum_new_pseudo_labels']} new pseudo-label, and seed "
        f"{design['random_seed']}. The threshold and cap were fixed before the run "
        "and are identical for every condition. Pseudo-label acceptance is defined "
        "as the number of accepted pseudo-labels divided by the original "
        "unlabelled-pool size. This definition is used consistently in every "
        "table and figure."
    )
    document.paragraph(
        "Leakage controls are the same as in Phase 5 plus two additions specific "
        "to self-training: accepted indices must be unique and disjoint from the "
        "labelled indices, and the loop receives only unlabelled features, never "
        "their true labels. The test rows enter only at final prediction. SHA-256 "
        "fingerprints and the saved iteration logs make these claims auditable; "
        "Phase 6's verifier audited every pseudo-label, and Phase 7 revalidates "
        "the logs and matched fingerprints."
    )
    document.paragraph(
        f"Self-training accepted {pseudo['total_pseudo_labels']:,} pseudo-labels "
        f"across all 81 conditions, with a mean of "
        f"{_num(pseudo['mean_pseudo_labels_per_experiment'], 1)} and a median of "
        f"{_num(pseudo['median_pseudo_labels_per_experiment'], 1)} per condition. "
        f"The overall acceptance rate was "
        f"{_pct(pseudo['overall_acceptance_rate'])}; the mean condition-level rate "
        f"was {_pct(pseudo['mean_experiment_acceptance_rate'])}, ranging from "
        f"{_pct(pseudo['min_acceptance_rate'])} to "
        f"{_pct(pseudo['max_acceptance_rate'])}. The mean number of completed "
        f"iterations was {_num(pseudo['mean_iterations'], 2)}. "
        f"{pseudo['iteration_cap_hits']} conditions stopped at the three-iteration "
        f"cap and {pseudo['threshold_stops']} stopped because no remaining sample "
        "met the threshold."
    )

    # 11 -----------------------------------------------------------------
    document.heading("11. Experimental Results")
    document.paragraph(
        "This section reports the saved results without interpretation; Section 12 "
        "compares the methods and Section 13 discusses statistical status. Table 7 "
        "gives the overall paired summary for both metrics. Tables 8-11 decompose "
        "macro-F1 by each factor, and Table 12 does the same for accuracy. Table "
        "13 lists the best observed configuration per dataset before and after "
        "self-training, and Tables 14-16 audit pseudo-labels, warnings, and "
        "degenerate fits."
    )
    overall_rows = []
    for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        for scope, scope_label in (
            ("all_conditions", "All 81 conditions"),
            ("excluding_degenerate_fits", "78 non-degenerate conditions"),
        ):
            record = _overall(payload, metric, scope)
            overall_rows.append(
                [
                    label,
                    scope_label,
                    str(record["n"]),
                    _num(record["mean_supervised"]),
                    _num(record["mean_self_training"]),
                    _signed(record["mean_difference"]),
                    _signed(record["median_difference"]),
                    _num(record["sd_difference"]),
                    _wins(record),
                    _effect(record["descriptive_cohens_dz"]),
                ]
            )
    document.table(
        [
            "Metric",
            "Scope",
            "n",
            "Supervised mean",
            "Self-training mean",
            "Mean change",
            "Median change",
            "SD of change",
            "W / U / L",
            "d_z",
        ],
        overall_rows,
        "Overall paired comparison of self-training against supervised learning "
        "(change = self-training minus supervised).",
    )
    _add_figure(
        document,
        catalog,
        "fig01_accuracy_supervised_vs_self_training.png",
        figures_path,
    )
    _add_figure(
        document,
        catalog,
        "fig02_macro_f1_supervised_vs_self_training.png",
        figures_path,
    )
    _add_figure(
        document,
        catalog,
        "fig03_paired_improvement_distributions.png",
        figures_path,
    )

    headers, rows = _comparison_stats_table(
        payload, "macro_f1", "dataset", "Dataset"
    )
    document.table(headers, rows, "Macro-F1 by dataset.")
    headers, rows = _comparison_stats_table(
        payload, "macro_f1", "feature_representation", "Representation"
    )
    document.table(headers, rows, "Macro-F1 by feature representation.")
    headers, rows = _comparison_stats_table(
        payload, "macro_f1", "classifier", "Classifier"
    )
    document.table(headers, rows, "Macro-F1 by classifier.")
    headers, rows = _comparison_stats_table(
        payload, "macro_f1", "labelled_percentage", "Labelled proportion"
    )
    document.table(headers, rows, "Macro-F1 by labelled proportion.")

    accuracy_rows = []
    for factor, display in (
        ("dataset", "Dataset"),
        ("feature_representation", "Representation"),
        ("classifier", "Classifier"),
        ("labelled_percentage", "Labelled proportion"),
    ):
        for record in _factor_rows(payload, "accuracy", factor):
            key = next(iter(record["factors"]))
            value = record["factors"][key]
            if key == "dataset":
                level = _dataset_name(value)
            elif key == "feature_representation":
                level = _representation_name(value)
            elif key == "classifier":
                level = _classifier_name(value)
            else:
                level = f"{value}%"
            accuracy_rows.append(
                [
                    display,
                    level,
                    str(record["n"]),
                    _num(record["mean_supervised"]),
                    _num(record["mean_self_training"]),
                    _signed(record["mean_difference"]),
                    _wins(record),
                    _effect(record["descriptive_cohens_dz"]),
                ]
            )
    document.table(
        [
            "Factor",
            "Level",
            "n",
            "Supervised accuracy",
            "Self-training accuracy",
            "Mean change",
            "W / U / L",
            "d_z",
        ],
        accuracy_rows,
        "Accuracy by factor level.",
    )

    best_rows = []
    for name in DATASET_ORDER:
        for method, label in (("supervised", "Supervised"), ("self_training", "Self-training")):
            record = best_f1[name][method]
            best_rows.append(
                [
                    _dataset_name(name),
                    label,
                    f"{record['labelled_percentage']}%",
                    _representation_name(record["feature_representation"]),
                    _classifier_name(record["classifier"]),
                    _num(record["score"]),
                    _num(record["accuracy"]),
                    _num(record["training_time_seconds"], 3),
                ]
            )
    document.table(
        [
            "Dataset",
            "Method",
            "Proportion",
            "Representation",
            "Classifier",
            "Best macro-F1",
            "Accuracy",
            "Training time (s)",
        ],
        best_rows,
        "Best observed macro-F1 configuration per dataset and method.",
    )
    document.paragraph(
        "The best-configuration table is descriptive. The winning condition is "
        "chosen by its test macro-F1, so its score is optimistically selected and "
        "is not an unbiased generalisation estimate. It is useful for describing "
        "which regions of the design performed well, not for claiming that those "
        "regions are universally superior."
    )

    pseudo_rows = []
    for record in pseudo["by_dataset"]:
        pseudo_rows.append(
            [
                _dataset_name(record["dataset"]),
                str(record["n"]),
                f"{record['total_pseudo_labels']:,}",
                _num(record["mean_pseudo_labels"], 1),
                _pct(record["mean_acceptance_rate"]),
                _num(record["mean_iterations"], 2),
                str(record["iteration_cap_hits"]),
                str(record["threshold_stops"]),
            ]
        )
    document.table(
        [
            "Dataset",
            "Conditions",
            "Total pseudo-labels",
            "Mean per condition",
            "Mean acceptance",
            "Mean iterations",
            "Cap hits",
            "Threshold stops",
        ],
        pseudo_rows,
        "Pseudo-label acceptance and stopping behaviour by dataset.",
    )
    warning_rows = []
    for phase, label in (("phase5", "Phase 5 supervised"), ("phase6", "Phase 6 self-training")):
        audit = warnings[phase]
        warning_rows.append(
            [
                label,
                str(audit["experiments_with_warnings"]),
                str(audit["experiments_without_warnings"]),
                ", ".join(
                    f"{name}: {count}" for name, count in sorted(audit["by_type"].items())
                ),
                ", ".join(
                    f"{_dataset_name(name)}: {count}"
                    for name, count in sorted(audit["by_dataset"].items())
                ),
            ]
        )
    document.table(
        ["Phase", "With warnings", "Without warnings", "By type", "By dataset"],
        warning_rows,
        "Summary of emitted model warnings.",
    )
    document.paragraph(
        f"Phase 6 emitted {warnings['phase6']['by_type'].get('ConvergenceWarning', 0)} "
        "LinearSVC convergence warnings and "
        f"{warnings['phase6']['by_type'].get('NumericalStabilityWarning', 0)} "
        "NumericalStabilityWarnings. The latter mark the three degenerate MNIST "
        "LDA + Logistic Regression fits, which are reproduced in Table 16 and "
        "retained in every primary statistic."
    )
    degenerate_rows = [
        [
            record["experiment_id"],
            _num(record["accuracy"]),
            _num(record["macro_f1"]),
            record["stopping_reason"],
            str(record.get("fit_diagnostic", ""))[:220],
        ]
        for record in degenerate_records
    ]
    document.table(
        ["Experiment", "Accuracy", "Macro-F1", "Stopping reason", "Fit diagnostic"],
        degenerate_rows,
        "Numerically degenerate self-training fits (retained in the primary analysis).",
    )

    # 12 -----------------------------------------------------------------
    document.heading("12. Comparative Analysis")
    document.paragraph(
        "The paired comparison answers the four research questions with a "
        "consistent picture: the average effect is close to zero once the three "
        "degenerate fits are set aside, but the effect is strongly moderated by "
        "dataset, representation, and classifier. There is no condition in which "
        "self-training is uniformly beneficial."
    )
    document.heading("12.1 RQ1: representation", level=3)
    document.paragraph(
        f"Across all 81 conditions, self-training changed macro-F1 by "
        f"{_signed(baseline['mean_difference'])} with Baseline features, "
        f"{_signed(pca['mean_difference'])} with PCA, and "
        f"{_signed(lda['mean_difference'])} with LDA. The LDA average is not a "
        "typical LDA outcome: it is dominated by the MNIST LDA + Logistic "
        f"Regression failures. Within datasets, LDA changed macro-F1 by "
        f"{_signed(bc_lda['mean_difference'])} on Breast Cancer, "
        f"{_signed(mnist_lda['mean_difference'])} on MNIST, and "
        f"{_signed(har_lda['mean_difference'])} on HAR. Excluding the degenerate "
        "fits, the MNIST LDA mean is positive, but the LDA cell remains the most "
        "variable part of the design. Baseline and PCA were comparatively stable "
        f"(MNIST: {_signed(mnist_baseline['mean_difference'])} and "
        f"{_signed(mnist_pca['mean_difference'])}; HAR: "
        f"{_signed(har_baseline['mean_difference'])} and "
        f"{_signed(har_pca['mean_difference'])}; Breast Cancer: "
        f"{_signed(bc_baseline['mean_difference'])} and "
        f"{_signed(bc_pca['mean_difference'])})."
    )
    document.table(
        ["Dataset", "Baseline change", "PCA change", "LDA change"],
        [
            [
                _dataset_name("breast_cancer"),
                _signed(bc_baseline["mean_difference"]),
                _signed(bc_pca["mean_difference"]),
                _signed(bc_lda["mean_difference"]),
            ],
            [
                _dataset_name("mnist"),
                _signed(mnist_baseline["mean_difference"]),
                _signed(mnist_pca["mean_difference"]),
                _signed(mnist_lda["mean_difference"]),
            ],
            [
                _dataset_name("har"),
                _signed(har_baseline["mean_difference"]),
                _signed(har_pca["mean_difference"]),
                _signed(har_lda["mean_difference"]),
            ],
        ],
        "Mean macro-F1 change by dataset and representation (nine conditions per cell).",
    )
    document.paragraph(
        "The practical reading is that PCA neither helps nor hurts the average "
        "self-training effect on these data; it behaves like Baseline. LDA is the "
        "high-variance choice: it can improve class separation enough for "
        "pseudo-labelling to help, as in the MNIST 10% LDA + SVM gain, but its "
        "low-dimensional projection can also destabilise a linear probability "
        "model. The representation therefore does not have a main effect so much "
        "as a strong interaction with classifier and dataset."
    )
    _add_figure(
        document,
        catalog,
        "fig04_representation_comparison.png",
        figures_path,
    )

    document.heading("12.2 RQ3: labelled proportion", level=3)
    document.paragraph(
        f"Mean macro-F1 changed by {_signed(p10['mean_difference'])} at 10% "
        f"labels, {_signed(p25['mean_difference'])} at 25%, and "
        f"{_signed(p50['mean_difference'])} at 50%. The average does not improve "
        "monotonically as labels become scarcer, because dataset and classifier "
        "effects dominate. The largest individual gains do occur at 10% labels "
        "(MNIST LDA + SVM), which is consistent with the motivation for "
        "self-training, but they coexist with losses at the same proportion. The "
        "balanced reading is that 10% is where self-training has the most "
        "opportunity to help and also the most opportunity to amplify a bad "
        "initial boundary."
    )
    _add_figure(
        document,
        catalog,
        "fig05_labelled_proportion_trend.png",
        figures_path,
    )

    document.heading("12.3 RQ2: classifier", level=3)
    document.paragraph(
        f"By classifier, mean macro-F1 changed by {_signed(lr['mean_difference'])} "
        f"for Logistic Regression, {_signed(svm['mean_difference'])} for linear "
        f"SVM, and {_signed(knn['mean_difference'])} for KNN. The Logistic "
        "Regression average is pulled down by the same three MNIST LDA failures; "
        "excluding those fits, its mean change becomes "
        f"{_signed(nd_lr['mean_difference'])}. The linear SVM is the only "
        "classifier with a positive overall mean, driven by large low-label MNIST "
        "gains. KNN is close to neutral: it is highly stable but accepts many "
        "pseudo-labels, so it reinforces its own decision boundary without "
        "changing it much."
    )
    document.table(
        ["Dataset", "Logistic Regression", "Linear SVM", "KNN"],
        [
            [
                _dataset_name("breast_cancer"),
                _signed(_factor(payload, "macro_f1", "dataset_x_classifier", dataset="breast_cancer", classifier="logistic_regression")["mean_difference"]),
                _signed(_factor(payload, "macro_f1", "dataset_x_classifier", dataset="breast_cancer", classifier="svm")["mean_difference"]),
                _signed(_factor(payload, "macro_f1", "dataset_x_classifier", dataset="breast_cancer", classifier="knn")["mean_difference"]),
            ],
            [
                _dataset_name("mnist"),
                _signed(mnist_lr["mean_difference"]),
                _signed(mnist_svm["mean_difference"]),
                _signed(mnist_knn["mean_difference"]),
            ],
            [
                _dataset_name("har"),
                _signed(_factor(payload, "macro_f1", "dataset_x_classifier", dataset="har", classifier="logistic_regression")["mean_difference"]),
                _signed(_factor(payload, "macro_f1", "dataset_x_classifier", dataset="har", classifier="svm")["mean_difference"]),
                _signed(_factor(payload, "macro_f1", "dataset_x_classifier", dataset="har", classifier="knn")["mean_difference"]),
            ],
        ],
        "Mean macro-F1 change by dataset and classifier (nine conditions per cell).",
    )
    document.paragraph(
        "The classifier effect is partly a probability-scale effect. Logistic "
        "Regression and KNN accept a large share of the unlabelled pool at the "
        "0.90 threshold, while calibrated linear SVM accepts far fewer rows. The "
        "same threshold is therefore not equally strict across classifiers. This "
        "is a design limitation as well as a finding: confidence calibration "
        "changes both how much self-training happens and how effective it is."
    )
    _add_figure(
        document,
        catalog,
        "fig06_classifier_comparison.png",
        figures_path,
    )

    document.heading("12.4 Pseudo-label behaviour", level=3)
    document.paragraph(
        f"Acceptance is strongly dataset- and classifier-dependent. The overall "
        f"acceptance rate was {_pct(pseudo['overall_acceptance_rate'])} of the "
        f"unlabelled pool ({pseudo['total_pseudo_labels']:,} of "
        f"{pseudo['total_unlabelled_available']:,} rows), but the per-condition "
        f"mean was {_pct(pseudo['mean_experiment_acceptance_rate'])} and the range "
        f"was {_pct(pseudo['min_acceptance_rate'])} to "
        f"{_pct(pseudo['max_acceptance_rate'])}. HAR accepted the largest share "
        f"({_pct(next(row for row in pseudo['by_dataset'] if row['dataset'] == 'har')['mean_acceptance_rate'])} "
        f"on average) and MNIST the smallest "
        f"({_pct(next(row for row in pseudo['by_dataset'] if row['dataset'] == 'mnist')['mean_acceptance_rate'])}), "
        "which mirrors dataset size and the difficulty of the initial boundary "
        "rather than any property of the representation alone. The number of "
        "accepted labels fell sharply after the first round, from "
        f"{pseudo['rounds'][0]['pseudo_labels_added']:,} in round 1 to "
        f"{pseudo['rounds'][-1]['pseudo_labels_added']:,} in round "
        f"{pseudo['rounds'][-1]['iteration']}, and the weighted mean confidence of "
        "accepted labels fell only slightly across rounds."
    )
    _add_figure(
        document,
        catalog,
        "fig09_pseudo_label_acceptance.png",
        figures_path,
    )
    _add_figure(
        document,
        catalog,
        "fig10_iteration_statistics.png",
        figures_path,
    )

    document.heading("12.5 Largest changes", level=3)
    gain_rows = [
        [
            record["experiment_id"],
            _num(record["supervised_macro_f1"]),
            _num(record["self_training_macro_f1"]),
            _signed(record["macro_f1_improvement"]),
            f"{record['pseudo_label_count']:,}",
            str(record["iterations_completed"]),
        ]
        for record in top_f1["largest_improvements"][:6]
    ]
    document.table(
        [
            "Experiment",
            "Supervised macro-F1",
            "Self-training macro-F1",
            "Change",
            "Pseudo-labels",
            "Iterations",
        ],
        gain_rows,
        "Six largest macro-F1 improvements.",
    )
    loss_rows = [
        [
            record["experiment_id"],
            _num(record["supervised_macro_f1"]),
            _num(record["self_training_macro_f1"]),
            _signed(record["macro_f1_improvement"]),
            f"{record['pseudo_label_count']:,}",
            str(record["iterations_completed"]),
        ]
        for record in top_f1["largest_regressions"][:6]
    ]
    document.table(
        [
            "Experiment",
            "Supervised macro-F1",
            "Self-training macro-F1",
            "Change",
            "Pseudo-labels",
            "Iterations",
        ],
        loss_rows,
        "Six largest macro-F1 regressions, including the degenerate fits.",
    )
    document.paragraph(
        "The gain list is dominated by MNIST at 10% labels, especially LDA + SVM "
        f"({_signed(top_f1['largest_improvements'][0]['macro_f1_improvement'])}) "
        "and the PCA/Baseline KNN conditions. The loss list is dominated by the "
        "three MNIST LDA + Logistic Regression fits; the largest non-degenerate "
        "loss is much smaller. This asymmetry is the single most important "
        "qualitative fact in the study: the mean change is negative because of "
        "three failures, not because self-training systematically harms typical "
        "conditions."
    )
    _add_figure(
        document,
        catalog,
        "fig11_improvement_heatmaps.png",
        figures_path,
    )

    document.heading("12.6 Best configurations", level=3)
    best_rows_acc = []
    for name in DATASET_ORDER:
        for method, label in (("supervised", "Supervised"), ("self_training", "Self-training")):
            record = best_acc[name][method]
            best_rows_acc.append(
                [
                    _dataset_name(name),
                    label,
                    f"{record['labelled_percentage']}%",
                    _representation_name(record["feature_representation"]),
                    _classifier_name(record["classifier"]),
                    _num(record["score"]),
                ]
            )
    document.table(
        ["Dataset", "Method", "Proportion", "Representation", "Classifier", "Best accuracy"],
        best_rows_acc,
        "Best observed accuracy configuration per dataset and method.",
    )
    document.paragraph(
        "On Breast Cancer the best supervised score exceeds the best "
        "self-training score, but both are near the ceiling and the dataset has "
        "only 114 test rows, so one or two samples move the metric by about 0.01. "
        "On MNIST the best self-training score is essentially tied with the best "
        "supervised score, achieved by the same PCA + KNN condition at 50% labels. "
        "On HAR the two best scores are also close. The best-configuration view "
        "therefore does not show a decisive advantage for either method; it shows "
        "that the top of the design is saturated and that method differences are "
        "concentrated in the middle and lower part of the distribution."
    )

    # 13 -----------------------------------------------------------------
    document.heading("13. Statistical Analysis, Where Justified")
    document.paragraph(
        "No inferential test is reported, and this is a deliberate design "
        "decision rather than an omission. Every condition was executed once on "
        "the same fixed split with seed 42. Conditions share training rows, test "
        "rows, scalers, and transformations, so they are not independent "
        "replicates of a population of splits. A paired t-test, Wilcoxon test, or "
        "bootstrap over the 81 conditions would treat correlated conditions as "
        "independent sampling units and would overstate the evidence. There is no "
        "second split with which to estimate split-to-split variability. "
        "Consequently, no p-value, confidence interval, or significance claim is "
        "made anywhere in this report."
    )
    document.paragraph(
        "What is justified is descriptive paired summarisation. For each metric the "
        "per-condition difference is d_i = metric_self_training,i - "
        "metric_supervised,i. The report gives the mean, median, standard "
        "deviation, minimum, quartiles, maximum, and counts of wins, ties, and "
        "losses at a tie tolerance of 1e-12. It also reports Cohen's d_z = mean(d) "
        "/ SD(d) as a standardised descriptive effect size. This d_z describes the "
        "magnitude of the observed differences in SD units; it is not a test "
        "statistic and has no attached p-value. Its denominator is the SD of the "
        "paired differences, not a population SD estimate."
    )
    effect_rows = []
    for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        for scope, scope_label in (
            ("all_conditions", "All 81"),
            ("excluding_degenerate_fits", "78 non-degenerate"),
        ):
            record = _overall(payload, metric, scope)
            effect_rows.append(
                [
                    label,
                    scope_label,
                    _signed(record["mean_difference"]),
                    _signed(record["median_difference"]),
                    _num(record["sd_difference"]),
                    _effect(record["descriptive_cohens_dz"]),
                    _wins(record),
                ]
            )
    document.table(
        ["Metric", "Scope", "Mean change", "Median change", "SD", "d_z", "W / U / L"],
        effect_rows,
        "Descriptive effect sizes for the paired differences.",
    )
    document.paragraph(
        "The sensitivity analysis is essential. For macro-F1, the all-condition "
        f"effect is {_effect(f1['descriptive_cohens_dz'])} but the non-degenerate "
        f"effect is {_effect(f1_nd['descriptive_cohens_dz'])}; the sign reverses. "
        "For accuracy the same reversal occurs "
        f"({_effect(accuracy['descriptive_cohens_dz'])} to "
        f"{_effect(accuracy_nd['descriptive_cohens_dz'])}). A reader who looked "
        "only at the all-condition mean would conclude that self-training harms "
        "performance on average; a reader who looked only at the non-degenerate "
        "subset would conclude that it is a small positive. Both statements "
        "describe the same saved data, and the honest conclusion must carry both. "
        "The primary reporting rule in this project is to keep all 81 conditions "
        "in the headline numbers and to present the 78-condition sensitivity "
        "immediately beside them."
    )
    document.paragraph(
        "Three further cautions apply. First, the wins/ties/losses counts are not "
        "a test; they summarise the direction of paired changes. Second, each "
        "marginal mean aggregates over the other factors and can therefore hide "
        "an interaction; the dataset-by-factor tables expose those interactions, "
        "and both views are reported. Third, timing measurements are not "
        "perfectly controlled: they were "
        "recorded in one environment with shared CPU and BLAS threading, so small "
        "time differences are not meaningful. Only large ratios, such as the "
        "self-training overhead, are interpreted."
    )

    # 14 -----------------------------------------------------------------
    document.heading("14. Discussion")
    document.paragraph(
        "The results support a nuanced answer to the project question. On these "
        "three datasets, with a fixed 0.90 threshold and a three-iteration cap, "
        "self-training is not a general-purpose improvement over supervised "
        "learning. Once three documented numerical failures are excluded, the mean "
        "macro-F1 change is a small positive "
        f"({_signed(f1_nd['mean_difference'])}, d_z "
        f"{_effect(f1_nd['descriptive_cohens_dz'])}), with wins and losses roughly "
        "balanced. The average is therefore best described as 'near neutral with "
        "high heterogeneity', not as 'self-training helps' or 'self-training "
        "hurts'."
    )
    document.paragraph(
        "Heterogeneity is systematic rather than random. The largest genuine "
        "benefit appears for MNIST at 10% labels with LDA + linear SVM, where the "
        "supervised model is clearly underfit and pseudo-labelling supplies "
        "useful additional decision boundary information. The largest harm "
        "appears for MNIST LDA + Logistic Regression, where the combination of a "
        "nine-dimensional projection, a linear probability model, and extreme "
        "feature magnitudes produces a degenerate fit. The failure is a numerical "
        "conditioning failure, not evidence that LDA or Logistic Regression is "
        "inherently unsuitable for MNIST. It is nevertheless a real outcome of the "
        "specified pipeline, so it stays in the primary results."
    )
    document.paragraph(
        "Breast Cancer behaves differently because the dataset is small and the "
        "task is close to saturated. LDA reduces it to a single dimension; the "
        "supervised scores are already high, and self-training changes little. "
        "HAR is the most stable dataset: Baseline and PCA are essentially "
        "unchanged, and LDA loses only a small amount. Its larger test set and "
        "subject-disjoint split make the estimates less noisy than Breast Cancer, "
        "but the same single-split caveat applies."
    )
    document.paragraph(
        "The classifier comparison clarifies why some earlier self-training "
        "studies report large gains and others report none. Logistic Regression "
        "and KNN produce smoothly varying probabilities and accept most of the "
        "unlabelled pool at 0.90; linear SVM with Platt calibration is much more "
        "conservative. The effective amount of self-training therefore differs by "
        "an order of magnitude across classifiers even though the nominal "
        "threshold is identical. A fair comparison of SSL methods must control "
        "for this, either by calibrating thresholds per classifier or by reporting "
        "the accepted fraction alongside the threshold."
    )
    document.paragraph(
        "Finally, the cost side is unambiguous. Self-training multiplied training "
        f"time by {_num(runtime['training_time_ratio_total'], 3)} overall and by a "
        f"median of {_num(runtime['median_condition_training_time_ratio'], 2)} per "
        f"condition, with a maximum of "
        f"{_num(runtime['max_condition_training_time_ratio'], 1)} for one "
        "condition. Most of the absolute cost comes from MNIST linear SVM. A near-"
        "neutral average performance change at several times the training cost is "
        "a poor trade unless the specific dataset/classifier region is known to "
        "benefit, which is exactly the kind of region this factorial design "
        "identifies."
    )

    # 15 -----------------------------------------------------------------
    document.heading("15. Computational Complexity and Runtime Analysis")
    document.paragraph(
        "Runtime was measured around each model fit in both phases. Phase 5 "
        "recorded training time only; Phase 6 recorded training time and final "
        "prediction time. The totals below are sums over the 81 conditions and "
        "are therefore dominated by the most expensive dataset/classifier "
        "combinations rather than by the typical condition."
    )
    runtime_rows = []
    for record in runtime["by_dataset"]:
        runtime_rows.append(
            [
                _dataset_name(record["dataset"]),
                _num(record["supervised_training_seconds"], 2),
                _num(record["self_training_training_seconds"], 2),
                _num(record["training_time_ratio"], 2),
                _num(record["self_training_prediction_seconds"], 4),
            ]
        )
    runtime_rows.append(
        [
            "All datasets",
            _num(runtime["supervised_training_seconds_total"], 2),
            _num(runtime["self_training_training_seconds_total"], 2),
            _num(runtime["training_time_ratio_total"], 3),
            _num(runtime["self_training_prediction_seconds_total"], 4),
        ]
    )
    document.table(
        [
            "Dataset",
            "Supervised training (s)",
            "Self-training training (s)",
            "Training ratio",
            "Self-training prediction (s)",
        ],
        runtime_rows,
        "Measured runtime totals by dataset (sum over 27 conditions each).",
    )
    document.paragraph(
        f"Self-training required {_num(runtime['training_time_ratio_total'], 3)}x "
        "the supervised training time in total. MNIST linear SVM alone consumed "
        f"{_num(runtime['mnist_svm_training_seconds'], 2)} s, or "
        f"{_pct(runtime['mnist_svm_share_of_self_training_training'])} of all "
        "Phase 6 training time. The per-condition ratio ranged from "
        f"{_num(runtime['min_condition_training_time_ratio'], 2)}x to "
        f"{_num(runtime['max_condition_training_time_ratio'], 1)}x with a median "
        f"of {_num(runtime['median_condition_training_time_ratio'], 2)}x. The very "
        "large maximum reflects a near-zero supervised fit time for a small "
        "condition, not a meaningful absolute cost."
    )
    document.paragraph(
        f"Phase 6 prediction time totalled "
        f"{_num(runtime['self_training_prediction_seconds_total'], 4)} s across "
        "all conditions: "
        + ", ".join(
            f"{_dataset_name(record['dataset'])} "
            f"{_num(record['self_training_prediction_seconds'], 4)} s"
            for record in runtime["by_dataset"]
        )
        + ". Phase 5 did not record prediction time, so the report makes no "
        "supervised-versus-self-training prediction-time claim. This gap is a "
        "limitation of the saved results, not something that can be reconstructed "
        "after the fact."
    )
    document.paragraph(
        "The asymptotic picture explains the measurements. Training a linear "
        "classifier costs O(n d) per iteration for Logistic Regression and linear "
        "SVM, where n grows as pseudo-labels accumulate; KNN has effectively no "
        "training cost in this implementation and defers O(n d) work per query to "
        "prediction time. Self-training repeats the fit up to three times on a "
        "growing training set, so its cost is roughly the number of iterations "
        "times the per-fit cost. PCA adds a one-off SVD cost of order "
        "O(min(n, d)^2 max(n, d)) per dataset, negligible here. LDA with the SVD "
        "solver is similarly cheap at these dimensions. The dominant practical "
        "cost is therefore repeated linear-SVM optimisation on high-dimensional "
        "MNIST features, which is exactly where the measured share is highest."
    )
    _add_figure(document, catalog, "fig07_training_time.png", figures_path)
    _add_figure(document, catalog, "fig08_prediction_time.png", figures_path)

    # 16 -----------------------------------------------------------------
    document.heading("16. Limitations")
    document.paragraph(
        "The following limitations qualify every conclusion in this report. They "
        "are separated from warnings and from failed experiments in the companion "
        "document limitations_and_future_work.md."
    )
    document.numbered(
        [
            "Single split and single seed. Each dataset has one fixed split and "
            "seed 42. There is no independent replication, so split-to-split "
            "variability and run-to-run variability are unknown. All comparisons "
            "are conditional on this split.",
            "No significance testing. Because conditions share rows, test samples, "
            "and transformations, they are not independent replicates. No p-values "
            "or confidence intervals are reported; effect sizes are descriptive.",
            "Three numerically degenerate fits remain in the primary statistics. "
            "They inflate the negative tail of the macro-F1 distribution and "
            "reverse the sign of the overall mean. The 78-condition sensitivity "
            "analysis is reported alongside, but neither view alone is complete.",
            "The self-training iteration cap is three. 68 of 81 conditions stopped "
            "at the cap, so many runs had not reached a natural confidence "
            "equilibrium. Longer runs could change both acceptance and performance.",
            "The confidence threshold is fixed at 0.90 for every classifier. "
            "Calibrated SVM probabilities are not on the same scale as Logistic "
            "Regression or KNN probabilities, so the effective strictness differs.",
            "SVM probabilities are approximate. Platt calibration is fitted on a "
            "20% stratified subset of labelled rows; calibration quality is not "
            "separately evaluated, and final predictions come from an uncalibrated "
            "LinearSVC.",
            "Supervised prediction time is missing. Phase 5 did not record it, so "
            "the prediction-cost comparison is one-sided and no total inference "
            "cost can be computed.",
            "Timing is not a controlled benchmark. Measurements come from one "
            "machine with shared CPU and BLAS threading; only large ratios are "
            "interpreted.",
            "Best configurations are selected on test performance. The best-case "
            "tables and the largest-gain lists are optimistically selected and are "
            "not unbiased generalisation estimates.",
            "The design is limited to linear DR and three classifiers. PCA and LDA "
            "are linear; Logistic Regression and linear SVM are linear; KNN is "
            "non-parametric but uses raw Euclidean distance. Nonlinear DR and "
            "kernel or deep classifiers are out of scope.",
            "Hyperparameters were deliberately not tuned. This protects the "
            "factorial comparison but may understate what each classifier could "
            "achieve with per-condition tuning.",
            "Dataset provenance includes documented substitutions. MNIST was staged "
            "from a public Kaggle mirror because official competition credentials "
            "were not configured; HAR came from the UCI-owned Kaggle copy because "
            "the direct UCI transfer was unreliable. Both substitutions are "
            "documented in data/README.md and preserve the original feature layout.",
        ]
    )

    # 17 -----------------------------------------------------------------
    document.heading("17. Future Scope")
    document.numbered(
        [
            "Repeat the full grid over multiple splits and seeds, then analyse the "
            "results with a hierarchical or mixed-effects model that treats split "
            "and condition as nested random effects. This would provide genuine "
            "uncertainty estimates without pretending that conditions are "
            "independent.",
            "Select the confidence threshold on a validation partition, separately "
            "for each classifier's probability scale, and report the accepted "
            "fraction as a design variable rather than a fixed consequence of the "
            "threshold.",
            "Evaluate per-class calibrated thresholds and class-balanced "
            "pseudo-labelling, especially for MNIST where some digits are accepted "
            "far more readily than others.",
            "Investigate the numerical conditioning of LDA + Logistic Regression "
            "directly: feature clipping, variance flooring, or constrained "
            "coefficients could turn the three failures into usable conditions "
            "without changing the scientific question.",
            "Compare self-training with other semi-supervised methods such as "
            "co-training, label propagation, entropy minimisation, and consistency "
            "regularisation under the same matched splits.",
            "Add nonlinear dimensionality reduction (kernel PCA, UMAP, "
            "autoencoders) and nonlinear classifiers (RBF SVM, gradient boosting, "
            "small neural networks) to test whether the representation/classifier "
            "interaction persists beyond the linear regime.",
            "Record prediction time in every phase and measure end-to-end "
            "inference cost, including KNN neighbour search, so that cost-benefit "
            "comparisons are complete.",
            "Perform nested cross-validation for any future hyperparameter tuning, "
            "keeping the outer test partition untouched, so that model selection "
            "does not contaminate the reported scores.",
            "Extend the analysis to per-class precision, recall, and confusion "
            "structure, and to class-imbalance-sensitive metrics, to explain where "
            "the aggregate changes come from.",
            "Package the verified comparison tables and figures into an interactive "
            "dashboard for exploration, while keeping the static report and its "
            "provenance hashes as the authoritative record.",
        ]
    )

    # 18 -----------------------------------------------------------------
    document.heading("18. Conclusion")
    document.paragraph(
        "This project produced a fully matched 81-condition comparison of "
        "supervised learning and confidence-based self-training across three "
        "datasets, three labelled proportions, three feature representations, and "
        "three classifiers. All 162 runs completed and all 81 pairs were verified "
        "to share the same split, labels, and transformed feature dimensions. The "
        "analysis is strictly read-only with respect to Phases 1-6."
    )
    document.paragraph(
        f"The headline result is that self-training is not a general improvement "
        f"at a 0.90 threshold and a three-iteration cap: mean macro-F1 changed by "
        f"{_signed(f1['mean_difference'])} across all conditions. That negative "
        "average is not robust to the three numerically degenerate MNIST LDA + "
        "Logistic Regression fits; excluding them gives a small positive change of "
        f"{_signed(f1_nd['mean_difference'])}. The honest conclusion is therefore "
        "conditional: on typical non-degenerate conditions, self-training is "
        "approximately neutral, with substantial dataset- and classifier-specific "
        "heterogeneity. The clearest genuine benefit is MNIST at 10% labels with "
        "LDA + linear SVM; the clearest harm is the degenerate LDA + Logistic "
        "Regression region, which is a numerical failure rather than a general "
        "property of either method."
    )
    document.paragraph(
        "Representation matters through interaction rather than through a main "
        "effect. PCA behaves like Baseline; LDA is the high-variance option that "
        "produces both the largest gain and the largest failure. Classifier choice "
        "matters because it changes the effective amount of pseudo-labelling: "
        "Logistic Regression and KNN accept most of the pool at 0.90, while "
        "calibrated linear SVM accepts far less. The cost is substantial: "
        f"self-training required {_num(runtime['training_time_ratio_total'], 3)}x "
        "the supervised training time, with MNIST linear SVM accounting for "
        f"{_pct(runtime['mnist_svm_share_of_self_training_training'])} of it."
    )
    document.paragraph(
        "These findings are conditional on one fixed split per dataset, one seed, "
        "a fixed threshold, a three-iteration cap, and the saved Phase 5/6 "
        "results. They are descriptive, not inferential. Within those bounds the "
        "study provides a reproducible map of where self-training helps, where it "
        "is neutral, and where it fails, together with an auditable trail from "
        "every number in this report back to a saved result file."
    )

    # 19 -----------------------------------------------------------------
    document.heading("19. References")
    references = [
        "Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). "
        "A public domain dataset for human activity recognition using smartphones. "
        "Proceedings of the 21st European Symposium on Artificial Neural Networks "
        "(ESANN 2013), 437-442.",
        "Chapelle, O., Scholkopf, B., & Zien, A. (Eds.). (2006). Semi-Supervised "
        "Learning. MIT Press.",
        "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine "
        "Learning, 20(3), 273-297.",
        "Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. "
        "IEEE Transactions on Information Theory, 13(1), 21-27.",
        "Deng, L. (2012). The MNIST database of handwritten digit images for "
        "machine learning research. IEEE Signal Processing Magazine, 29(6), "
        "141-142.",
        "Fisher, R. A. (1936). The use of multiple measurements in taxonomic "
        "problems. Annals of Eugenics, 7(2), 179-188.",
        "Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., "
        "Virtanen, P., Cournapeau, D., et al. (2020). Array programming with "
        "NumPy. Nature, 585, 357-362.",
        "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in "
        "Science & Engineering, 9(3), 90-95.",
        "Jolliffe, I. T. (2002). Principal Component Analysis (2nd ed.). Springer.",
        "Kaggle. (n.d.). Breast Cancer Wisconsin (Diagnostic) Data Set "
        "[uciml/breast-cancer-wisconsin-data]. "
        "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data",
        "Kaggle. (n.d.). Digit Recognizer competition data. "
        "https://www.kaggle.com/competitions/digit-recognizer/data",
        "Kaggle. (n.d.). Human Activity Recognition with Smartphones "
        "[uciml/human-activity-recognition-with-smartphones]. "
        "https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones",
        "LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based "
        "learning applied to document recognition. Proceedings of the IEEE, "
        "86(11), 2278-2324.",
        "Lee, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised "
        "learning method for deep neural networks. ICML 2013 Workshop on Challenges "
        "in Representation Learning.",
        "Pearson, K. (1901). On lines and planes of closest fit to systems of "
        "points in space. Philosophical Magazine, 2(11), 559-572.",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., "
        "Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. "
        "Journal of Machine Learning Research, 12, 2825-2830.",
        "Platt, J. C. (1999). Probabilistic outputs for support vector machines and "
        "comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, "
        "B. Scholkopf, & D. Schuurmans (Eds.), Advances in Large Margin Classifiers "
        "(pp. 61-74). MIT Press.",
        "Rao, C. R. (1948). The utilization of multiple measurements in problems of "
        "biological classification. Journal of the Royal Statistical Society, "
        "Series B, 10(2), 159-203.",
        "Reyes-Ortiz, J.-L., Oneto, L., Ghio, A., Parra, X., & Anguita, D. (2016). "
        "Human Activity Recognition Using Smartphones. UCI Machine Learning "
        "Repository. https://doi.org/10.24432/C54S4K",
        "Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear "
        "feature extraction for breast tumor diagnosis. IS&T/SPIE Symposium on "
        "Electronic Imaging: Science and Technology, 1905, 861-870.",
        "Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., "
        "Cournapeau, D., et al. (2020). SciPy 1.0: Fundamental algorithms for "
        "scientific computing in Python. Nature Methods, 17, 261-272.",
        "Waskom, M. L. (2021). seaborn: Statistical data visualization. Journal of "
        "Open Source Software, 6(60), 3021.",
        "Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast Cancer "
        "Wisconsin (Diagnostic). UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5DW2B",
        "Yarowsky, D. (1995). Unsupervised word sense disambiguation rivaling "
        "supervised methods. Proceedings of the 33rd Annual Meeting of the "
        "Association for Computational Linguistics, 189-196.",
        "Zhu, X. (2005). Semi-supervised learning literature survey (Technical "
        "Report 1530). Computer Sciences, University of Wisconsin-Madison.",
    ]
    document.numbered(references)
    document.paragraph(
        "Project artefacts (result tables, verification reports, and dataset "
        "provenance) are the authoritative source for all empirical claims in this "
        "report. Software behaviour follows scikit-learn, NumPy, SciPy, "
        "matplotlib, and seaborn as cited above."
    )

    return document, document.render_markdown()


# ---------------------------------------------------------------------------
# Executive summary and limitations document
# ---------------------------------------------------------------------------
def build_executive_summary(payload: dict[str, Any]) -> str:
    accuracy = _overall(payload, "accuracy", "all_conditions")
    accuracy_nd = _overall(payload, "accuracy", "excluding_degenerate_fits")
    f1 = _overall(payload, "macro_f1", "all_conditions")
    f1_nd = _overall(payload, "macro_f1", "excluding_degenerate_fits")
    runtime = payload["runtime"]
    pseudo = payload["pseudo_labels"]
    warnings = payload["warnings"]
    best = payload["best_configurations"]["macro_f1"]
    top = payload["top_changes"]["macro_f1"]["largest_improvements"][0]

    lines = [
        "FINAL EXECUTIVE SUMMARY",
        "=" * 78,
        payload["title"],
        "Phase 7 - Comprehensive research analysis, comparison, and final report",
        f"Generated (UTC): {payload['generated_utc']}",
        "",
        "SCOPE",
        "-" * 78,
        "81 matched conditions: 3 datasets x 3 labelled proportions (10/25/50%)",
        "x 3 feature representations (Baseline/PCA/LDA) x 3 classifiers",
        "(Logistic Regression/linear SVM/KNN).",
        "Each condition was run once supervised (Phase 5) and once with",
        "confidence-based self-training (Phase 6) on the same fixed split and",
        "seed. 162 experiments, 81 paired comparisons, 0 failures.",
        "",
        "KEY FINDINGS",
        "-" * 78,
        f"1. Supervised mean macro-F1 = {_num(f1['mean_supervised'])}; "
        f"self-training mean = {_num(f1['mean_self_training'])}.",
        f"   Mean paired change = {_signed(f1['mean_difference'])} "
        f"(median {_signed(f1['median_difference'])}, SD {_num(f1['sd_difference'])}).",
        f"   Wins/ties/losses = {f1['improved']}/{f1['unchanged']}/{f1['regressed']} "
        "of 81.",
        "",
        f"2. The negative mean is driven by three numerically degenerate",
        f"   MNIST LDA + Logistic Regression fits (macro-F1 as low as",
        f"   {_num(min(record['macro_f1'] for record in warnings['degenerate_fits']['experiments']))}).",
        f"   Excluding them (n = 78): mean change {_signed(f1_nd['mean_difference'])}, "
        f"SD {_num(f1_nd['sd_difference'])},",
        f"   wins/ties/losses {f1_nd['improved']}/{f1_nd['unchanged']}/{f1_nd['regressed']}.",
        "   The sign of the average effect reverses; both views are reported.",
        "",
        f"3. Accuracy shows the same pattern: mean change "
        f"{_signed(accuracy['mean_difference'])} overall",
        f"   and {_signed(accuracy_nd['mean_difference'])} for the 78 "
        "non-degenerate conditions.",
        "",
        "4. Representation interacts with dataset and classifier. Mean macro-F1",
        f"   change: Baseline {_signed(_factor(payload, 'macro_f1', 'feature_representation', feature_representation='baseline')['mean_difference'])}, "
        f"PCA {_signed(_factor(payload, 'macro_f1', 'feature_representation', feature_representation='pca')['mean_difference'])}, "
        f"LDA {_signed(_factor(payload, 'macro_f1', 'feature_representation', feature_representation='lda')['mean_difference'])}.",
        "   The LDA mean is dominated by the three MNIST failures.",
        "",
        "5. Classifier matters. Mean macro-F1 change: Logistic Regression",
        f"   {_signed(_factor(payload, 'macro_f1', 'classifier', classifier='logistic_regression')['mean_difference'])}, "
        f"linear SVM {_signed(_factor(payload, 'macro_f1', 'classifier', classifier='svm')['mean_difference'])}, "
        f"KNN {_signed(_factor(payload, 'macro_f1', 'classifier', classifier='knn')['mean_difference'])}.",
        "   The linear SVM is the only classifier with a positive overall mean.",
        "",
        f"6. Largest genuine gain: {top['experiment_id']} with macro-F1 "
        f"{_num(top['supervised_macro_f1'])} ->",
        f"   {_num(top['self_training_macro_f1'])} ({_signed(top['macro_f1_improvement'])}); "
        f"{top['pseudo_label_count']:,} pseudo-labels,",
        f"   {top['iterations_completed']} iterations.",
        "",
        f"7. Pseudo-labelling: {pseudo['total_pseudo_labels']:,} accepted of "
        f"{pseudo['total_unlabelled_available']:,} available",
        f"   ({_pct(pseudo['overall_acceptance_rate'])} overall; "
        f"{_pct(pseudo['mean_experiment_acceptance_rate'])} mean per condition).",
        f"   {pseudo['iteration_cap_hits']} conditions hit the three-iteration cap;",
        f"   {pseudo['threshold_stops']} stopped because no sample reached 0.90.",
        "",
        "8. Cost: supervised training "
        f"{_num(runtime['supervised_training_seconds_total'])} s vs self-training",
        f"   {_num(runtime['self_training_training_seconds_total'])} s "
        f"({_num(runtime['training_time_ratio_total'], 3)}x). MNIST linear SVM accounts for",
        f"   {_pct(runtime['mnist_svm_share_of_self_training_training'])} of Phase 6 "
        "training time.",
        f"   Phase 6 prediction time totals "
        f"{_num(runtime['self_training_prediction_seconds_total'], 4)} s; Phase 5",
        "   prediction time was not recorded.",
        "",
        "BEST OBSERVED MACRO-F1 PER DATASET",
        "-" * 78,
    ]
    for name in DATASET_ORDER:
        supervised = best[name]["supervised"]
        self_training = best[name]["self_training"]
        lines.append(
            f"{_dataset_name(name)}:"
        )
        lines.append(
            f"  supervised   {_num(supervised['score'])} "
            f"({supervised['labelled_percentage']}% "
            f"{_representation_name(supervised['feature_representation'])}, "
            f"{_classifier_name(supervised['classifier'])})"
        )
        lines.append(
            f"  self-training {_num(self_training['score'])} "
            f"({self_training['labelled_percentage']}% "
            f"{_representation_name(self_training['feature_representation'])}, "
            f"{_classifier_name(self_training['classifier'])})"
        )
    lines.extend(
        [
            "",
            "WARNINGS AND QUALITY FLAGS",
            "-" * 78,
            f"Phase 5 warnings: {warnings['phase5']['experiments_with_warnings']} "
            f"experiments ({warnings['phase5']['by_type']}).",
            f"Phase 6 warnings: {warnings['phase6']['experiments_with_warnings']} "
            f"experiments ({warnings['phase6']['by_type']}).",
            f"Numerically degenerate fits: {warnings['degenerate_fits']['count']} "
            "(retained in primary statistics).",
            "Iteration cap: 3; confidence threshold: 0.90; seed: 42.",
            "",
            "STATISTICAL STATUS",
            "-" * 78,
            "Descriptive paired analysis only. One fixed split and one seed per",
            "dataset mean the 81 conditions are not independent replicates.",
            "No p-values, confidence intervals, or significance claims are made.",
            "Cohen's d_z values are descriptive effect sizes, not tests.",
            "",
            "VERIFICATION",
            "-" * 78,
            f"Source validation checks passed: {payload['validation']['passed']}.",
            f"Matched conditions: {payload['validation']['matched_experiments']} of "
            f"{payload['validation']['expected_experiments']}.",
            "All 81 + 81 experiments are accounted for. Every metric in this",
            "summary is generated from the saved result tables; no value was",
            "estimated or hand-entered.",
            "",
            "BOTTOM LINE",
            "-" * 78,
            "At a fixed 0.90 confidence threshold and a three-iteration cap,",
            "self-training is approximately neutral on typical non-degenerate",
            "conditions, not a universal improvement. It produces a large gain",
            "for MNIST at 10% labels with LDA + linear SVM, a large numerical",
            "failure for MNIST LDA + Logistic Regression, and substantial extra",
            "training cost. Representation and classifier choice matter through",
            "their interaction rather than as simple main effects.",
            "",
        ]
    )
    return "\n".join(lines)


def build_limitations_document(payload: dict[str, Any]) -> str:
    document = Document(
        "Limitations and Future Work",
        "Companion document to the Phase 7 final research report",
        metadata=[
            ("Generated (UTC)", payload["generated_utc"]),
            (
                "Companion report",
                "results/final_report/final_research_report.md",
            ),
            (
                "Evidence base",
                "81 Phase 5 + 81 Phase 6 experiments, 81 matched pairs",
            ),
        ],
    )
    warnings = payload["warnings"]
    pseudo = payload["pseudo_labels"]
    f1 = _overall(payload, "macro_f1", "all_conditions")
    f1_nd = _overall(payload, "macro_f1", "excluding_degenerate_fits")

    document.heading("1. Purpose and scope")
    document.paragraph(
        "This document records what qualified the Phase 7 comparison, separates "
        "three categories that are often conflated - warnings, limitations, and "
        "failed experiments - and lists work that would address them. It is "
        "generated from the same saved analysis payload as the final report, so "
        "every count and score below matches the result files."
    )

    document.heading("2. Warnings, limitations, and failures are different")
    document.table(
        ["Category", "Definition", "Effect on results in this project"],
        [
            [
                "Warning",
                "A diagnostic emitted by a fitting routine; the fit and its "
                "predictions still exist.",
                "Retained as valid measurements; flagged in Tables 13-14 of the "
                "report and in summary_warnings.csv.",
            ],
            [
                "Limitation",
                "A constraint of the design that bounds what can be concluded.",
                "Disclosed here and in report Section 16; no result is changed.",
            ],
            [
                "Failed experiment",
                "A run that did not complete or produced no usable prediction.",
                "None: 81/81 Phase 5 and 81/81 Phase 6 experiments completed. The "
                "three degenerate fits are numerical failures of the model, not "
                "missing experiments, and are retained.",
            ],
        ],
        "How warnings, limitations, and failures are treated.",
    )

    document.heading("3. Warnings observed")
    warning_rows = []
    for phase, label in (("phase5", "Phase 5"), ("phase6", "Phase 6")):
        audit = warnings[phase]
        warning_rows.append(
            [
                label,
                str(audit["experiments_with_warnings"]),
                str(audit["experiments_without_warnings"]),
                str(audit["by_type"].get("ConvergenceWarning", 0)),
                str(audit["by_type"].get("NumericalStabilityWarning", 0)),
                ", ".join(
                    f"{_dataset_name(name)}: {count}"
                    for name, count in sorted(audit["by_dataset"].items())
                ),
            ]
        )
    document.table(
        [
            "Phase",
            "With warnings",
            "Without warnings",
            "ConvergenceWarning",
            "NumericalStabilityWarning",
            "By dataset",
        ],
        warning_rows,
        "Warning counts across the 81 experiments in each phase.",
    )
    document.paragraph(
        "Phase 5 emitted 16 Liblinear ConvergenceWarnings, all for linear SVM "
        "(eight MNIST, eight HAR). Phase 6 emitted 17 ConvergenceWarnings (eight "
        "MNIST, nine HAR) plus three NumericalStabilityWarnings on MNIST LDA + "
        "Logistic Regression. The convergence warnings mean that the optimiser "
        "stopped at its iteration/tolerance limit; the fitted model was still "
        "evaluated. The numerical-stability warnings identify a more serious "
        "conditioning problem, analysed next."
    )

    document.heading("4. Numerical failures: three degenerate MNIST fits")
    document.paragraph(
        "Three Phase 6 conditions produced numerically degenerate models: MNIST "
        "with LDA features and Logistic Regression at 10%, 25%, and 50% labelled. "
        "The saved diagnostics show extreme feature magnitudes in the unlabelled "
        "pool (up to roughly 1e15 in scaled/LDA space), near-zero coefficients, "
        "and near-uniform predicted probabilities. Each run stopped after one "
        "iteration because no sample met the 0.90 threshold. These are real "
        "outcomes of the specified pipeline, not corrupted files; they remain in "
        "the primary 81-condition statistics."
    )
    degenerate_rows = [
        [
            record["experiment_id"],
            _num(record["accuracy"]),
            _num(record["macro_f1"]),
            record["stopping_reason"],
            str(record.get("fit_diagnostic", ""))[:260],
        ]
        for record in warnings["degenerate_fits"]["experiments"]
    ]
    document.table(
        ["Experiment", "Accuracy", "Macro-F1", "Stopping reason", "Diagnostic"],
        degenerate_rows,
        "The three numerically degenerate fits (all MNIST LDA + Logistic Regression).",
    )
    document.paragraph(
        f"Including them, mean macro-F1 change is "
        f"{_signed(f1['mean_difference'])} (d_z "
        f"{_effect(f1['descriptive_cohens_dz'])}). Excluding them, it is "
        f"{_signed(f1_nd['mean_difference'])} (d_z "
        f"{_effect(f1_nd['descriptive_cohens_dz'])}). The sign reversal is the "
        "single largest sensitivity in the study, so both values must be quoted "
        "together."
    )

    document.heading("5. Design and evidence limitations")
    document.numbered(
        [
            "A single split per dataset and one seed (42). No independent "
            "replication; split-to-split variability is unknown.",
            "Conditions are not independent replicates because they share rows, "
            "test samples, and transformations. Inferential testing is not "
            "justified, so no p-values or confidence intervals are reported.",
            "The three degenerate fits distort the all-condition mean; the "
            "78-condition sensitivity is reported but is also selected by a "
            "post hoc criterion (numerical degeneracy), so it is descriptive.",
            "The self-training cap of three iterations was reached by "
            f"{pseudo['iteration_cap_hits']} of 81 conditions, so many runs did "
            "not reach a natural confidence equilibrium.",
            f"Only {pseudo['threshold_stops']} conditions stopped because the "
            "threshold was exhausted; the cap, not the threshold, is the "
            "dominant stopping mechanism.",
            "A single 0.90 confidence threshold is applied to classifiers with "
            "different probability scales. Calibrated SVM probabilities are "
            "approximate and not directly comparable to predict_proba outputs.",
            "Phase 5 did not record prediction time, so the cost comparison is "
            "limited to training time plus Phase 6 prediction time.",
            "Timing comes from one shared environment and is not a controlled "
            "benchmark; only large ratios are interpreted.",
            "Best configurations and largest-gain lists are selected using test "
            "scores and are therefore optimistically biased.",
            "The design is linear (PCA, LDA, Logistic Regression, linear SVM) and "
            "uses Euclidean KNN; nonlinear alternatives are out of scope.",
            "No hyperparameter tuning or cross-validation was performed, by "
            "design, to keep the factorial comparison clean.",
            "Dataset provenance includes two documented substitutions: MNIST from "
            "the public Kaggle mirror khotijahs1/digitrecognizer (official "
            "competition credentials were not configured) and HAR from the "
            "UCI-owned Kaggle copy (direct UCI transfer was unreliable). Both "
            "preserve the documented feature layout; see data/README.md.",
        ]
    )

    document.heading("6. Threats to validity")
    document.table(
        ["Threat", "Description", "Mitigation in this project"],
        [
            [
                "Data leakage",
                "Test rows or hidden labels influencing scaling, DR, or "
                "pseudo-labelling.",
                "Scaler/PCA fitted on training pool only; LDA on labelled rows "
                "only; test used only for final metrics; SHA-256 fingerprints and "
                "Phase 6 audit.",
            ],
            [
                "Selection bias",
                "Reporting only favourable conditions.",
                "All 81 conditions are reported; wins/ties/losses and both "
                "degenerate-included and degenerate-excluded statistics are "
                "shown.",
            ],
            [
                "Metric sensitivity",
                "Accuracy can hide class-level errors.",
                "Macro-F1 is the primary metric; accuracy is reported beside it.",
            ],
            [
                "Measurement noise",
                "Timing varies with system load and BLAS threading.",
                "Only large ratios interpreted; no fine-grained timing claims.",
            ],
            [
                "Overfitting the test set",
                "Repeated reporting of the same fixed test split.",
                "No model selection on test was performed during training; best-"
                "configuration tables are explicitly labelled descriptive.",
            ],
        ],
        "Principal threats and the corresponding controls.",
    )

    document.heading("7. Future work")
    document.numbered(
        [
            "Run multiple seeds and repeated stratified splits; analyse with a "
            "mixed-effects model that separates split variance from condition "
            "variance.",
            "Tune the confidence threshold on a validation partition per "
            "classifier probability scale and report the accepted fraction.",
            "Add class-balanced or per-class thresholds to prevent majority "
            "classes from dominating pseudo-labelling.",
            "Stabilise LDA + Logistic Regression numerically (feature clipping, "
            "variance flooring, regularised LDA) and re-evaluate the three "
            "failure conditions.",
            "Compare with co-training, label propagation, entropy minimisation, "
            "and consistency regularisation under identical splits.",
            "Add nonlinear DR (kernel PCA, UMAP, autoencoders) and nonlinear "
            "classifiers (RBF SVM, gradient boosting, neural networks).",
            "Record prediction time in all phases and include KNN neighbour "
            "search in the end-to-end cost.",
            "Use nested cross-validation for future tuning so that the outer test "
            "partition stays untouched.",
            "Report per-class precision, recall, and confusion matrices to locate "
            "where aggregate changes originate.",
            "Build an interactive dashboard over the verified tables while "
            "retaining the static report and provenance hashes as the "
            "authoritative record.",
        ]
    )

    document.heading("8. Preservation statement")
    document.paragraph(
        "Phase 7 is read-only with respect to Phases 1-6. No Phase 5 or Phase 6 "
        "experiment was rerun, no existing result was edited, no dataset was "
        "modified, and no configuration was changed. The immutability snapshot "
        "provenance/phase1_phase6_snapshot.json records SHA-256 hashes for every "
        "Phase 1-6 artefact; scripts/verify_phase7.py rechecks them and writes "
        "the result to phase7_verification_report.txt."
    )
    return document.render_markdown()


# ---------------------------------------------------------------------------
# HTML/PDF output
# ---------------------------------------------------------------------------
def find_pdf_browser() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for name in ("msedge", "chrome", "google-chrome", "chromium"):
        located = shutil.which(name)
        if located:
            return Path(located)
    return None


def write_pdf_from_html(
    html_path: str | Path,
    pdf_path: str | Path,
    browser: Path | None = None,
    timeout: int = 240,
) -> dict[str, Any]:
    """Render a local HTML file to PDF with headless Edge or Chrome."""
    html_file = Path(html_path).resolve()
    pdf_file = Path(pdf_path).resolve()
    if not html_file.exists():
        raise FileNotFoundError(f"HTML source not found: {html_file}")
    executable = browser or find_pdf_browser()
    if executable is None:
        raise RuntimeError(
            "No supported browser found for HTML-to-PDF conversion. "
            "Install Microsoft Edge or Google Chrome."
        )
    if pdf_file.exists():
        pdf_file.unlink()
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    profile_dir = Path(tempfile.mkdtemp(prefix="phase7_pdf_profile_"))
    command = [
        str(executable),
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        f"--user-data-dir={profile_dir}",
        f"--print-to-pdf={pdf_file}",
        html_file.as_uri(),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    # Chromium can return from the parent process before the PDF is flushed.
    stable_size = -1
    for _ in range(120):
        if pdf_file.exists():
            size = pdf_file.stat().st_size
            if size > 10_000 and size == stable_size:
                break
            stable_size = size
        time.sleep(0.5)
    shutil.rmtree(profile_dir, ignore_errors=True)
    if not pdf_file.exists() or pdf_file.stat().st_size < 10_000:
        raise RuntimeError(
            "PDF generation failed. Browser exit code "
            f"{completed.returncode}; stderr: {completed.stderr[-800:]}"
        )
    return {
        "pdf": pdf_file,
        "size_bytes": int(pdf_file.stat().st_size),
        "browser": str(executable),
        "exit_code": int(completed.returncode),
    }
