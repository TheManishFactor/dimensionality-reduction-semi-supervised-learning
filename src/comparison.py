"""Compare Phase 5 supervised results with Phase 6 self-training results.

Experiments are matched on the full experimental condition:
``dataset``, ``labelled_percentage``, ``classifier``, and
``feature_representation``. Improvements are always computed as

    improvement = self-training metric - supervised metric

so negative values honestly represent a regression.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .experiment_runner import (
    DATASET_ORDER,
    PROJECT_ROOT,
    REPRESENTATION_ORDER,
    _json_safe,
    load_config,
)
from .classifiers import CLASSIFIER_NAMES

KEY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "labelled_percentage",
    "classifier",
    "feature_representation",
)
METRIC_COLUMNS: tuple[str, ...] = ("accuracy", "macro_f1")
COMPARISON_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "dataset",
    "labelled_percentage",
    "classifier",
    "feature_representation",
    "supervised_accuracy",
    "self_training_accuracy",
    "accuracy_improvement",
    "supervised_macro_f1",
    "self_training_macro_f1",
    "macro_f1_improvement",
    "supervised_training_time_seconds",
    "self_training_training_time_seconds",
    "training_time_difference_seconds",
    "pseudo_label_count",
    "iterations_completed",
    "pseudo_label_acceptance_rate",
    "supervised_status",
    "self_training_status",
    "numerically_degenerate_fit",
    "stopping_reason",
)
IMPROVEMENT_COLUMNS: tuple[str, ...] = (
    "accuracy_improvement",
    "macro_f1_improvement",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_phase5_results(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, Path]:
    """Load the Phase 5 supervised results table."""
    settings = config or load_config()
    directory = _resolve(
        settings.get("results", {}).get(
            "classifier_evaluation_dir", "results/classifier_evaluation"
        )
    )
    path = directory / "tables" / "classifier_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Phase 5 results table not found: {path}")
    return pd.read_csv(path), path


def load_phase6_results(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, Path]:
    """Load the Phase 6 self-training results table."""
    settings = config or load_config()
    directory = _resolve(
        settings.get("results", {}).get(
            "self_training_dir", "results/self_training"
        )
    )
    path = directory / "tables" / "self_training_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Phase 6 results table not found: {path}")
    return pd.read_csv(path), path


def _validate_unique(frame: pd.DataFrame, label: str) -> None:
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} results are missing key column(s): {missing}.")
    duplicates = int(frame.duplicated(subset=list(KEY_COLUMNS)).sum())
    if duplicates:
        raise ValueError(f"{label} results contain {duplicates} duplicate key(s).")


def build_comparison(
    supervised: pd.DataFrame,
    self_training: pd.DataFrame,
) -> pd.DataFrame:
    """Match Phase 5 and Phase 6 rows and compute metric improvements."""
    _validate_unique(supervised, "Phase 5")
    _validate_unique(self_training, "Phase 6")

    left = supervised.copy()
    right = self_training.copy()
    left["experiment_id"] = left["experiment_id"].astype(str)
    right["experiment_id"] = right["experiment_id"].astype(str)

    merged = left.merge(
        right,
        on=list(KEY_COLUMNS),
        how="inner",
        suffixes=("_supervised", "_self_training"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No supervised/self-training experiments could be matched.")

    supervised_metrics = {
        "accuracy": "accuracy_supervised",
        "macro_f1": "macro_f1_supervised",
    }
    self_metrics = {
        "accuracy": "accuracy_self_training",
        "macro_f1": "macro_f1_self_training",
    }
    both_completed = (
        (merged["status_supervised"] == "completed")
        & (merged["status_self_training"] == "completed")
    )

    comparison = pd.DataFrame(
        {
            "experiment_id": merged["experiment_id_supervised"],
            "dataset": merged["dataset"],
            "labelled_percentage": merged["labelled_percentage"].astype(int),
            "classifier": merged["classifier"],
            "feature_representation": merged["feature_representation"],
            "supervised_accuracy": merged[supervised_metrics["accuracy"]],
            "self_training_accuracy": merged[self_metrics["accuracy"]],
            "supervised_macro_f1": merged[supervised_metrics["macro_f1"]],
            "self_training_macro_f1": merged[self_metrics["macro_f1"]],
            "supervised_training_time_seconds": merged["training_time_seconds_supervised"],
            "self_training_training_time_seconds": merged["training_time_seconds_self_training"],
            "pseudo_label_count": merged["pseudo_label_count"],
            "iterations_completed": merged["iterations_completed"],
            "pseudo_label_acceptance_rate": merged["pseudo_label_acceptance_rate"],
            "supervised_status": merged["status_supervised"],
            "self_training_status": merged["status_self_training"],
            "numerically_degenerate_fit": merged.get(
                "numerically_degenerate_fit",
                pd.Series(False, index=merged.index),
            ),
            "stopping_reason": merged["stopping_reason"],
        }
    )
    for metric in METRIC_COLUMNS:
        comparison[f"{metric}_improvement"] = (
            comparison[f"self_training_{metric}"]
            - comparison[f"supervised_{metric}"]
        )
    comparison["training_time_difference_seconds"] = (
        comparison["self_training_training_time_seconds"]
        - comparison["supervised_training_time_seconds"]
    )

    comparison.loc[~both_completed, list(IMPROVEMENT_COLUMNS)] = np.nan
    comparison.loc[~both_completed, "training_time_difference_seconds"] = np.nan
    comparison.loc[~both_completed, "supervised_accuracy"] = np.nan
    comparison.loc[~both_completed, "supervised_macro_f1"] = np.nan
    comparison.loc[~both_completed, "self_training_accuracy"] = np.nan
    comparison.loc[~both_completed, "self_training_macro_f1"] = np.nan

    dataset_rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    representation_rank = {
        name: index for index, name in enumerate(REPRESENTATION_ORDER)
    }
    classifier_rank = {name: index for index, name in enumerate(CLASSIFIER_NAMES)}
    comparison["_dataset_rank"] = comparison["dataset"].map(
        lambda value: dataset_rank.get(str(value), len(dataset_rank))
    )
    comparison["_representation_rank"] = comparison["feature_representation"].map(
        lambda value: representation_rank.get(str(value), len(representation_rank))
    )
    comparison["_classifier_rank"] = comparison["classifier"].map(
        lambda value: classifier_rank.get(str(value), len(classifier_rank))
    )
    comparison = comparison.sort_values(
        [
            "_dataset_rank",
            "labelled_percentage",
            "_representation_rank",
            "_classifier_rank",
        ]
    ).drop(
        columns=["_dataset_rank", "_representation_rank", "_classifier_rank"]
    )
    comparison = comparison.reset_index(drop=True)
    return comparison[list(COMPARISON_COLUMNS)]


def summarise_comparison(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate improvement statistics from matched rows."""
    valid = frame.dropna(subset=list(IMPROVEMENT_COLUMNS))
    summary: dict[str, Any] = {
        "matched_experiments": int(len(frame)),
        "comparable_experiments": int(len(valid)),
        "incomplete_matches": int(len(frame) - len(valid)),
    }
    if valid.empty:
        return summary

    for metric in METRIC_COLUMNS:
        improvement = valid[f"{metric}_improvement"]
        summary[metric] = {
            "mean_supervised": float(valid[f"supervised_{metric}"].mean()),
            "mean_self_training": float(valid[f"self_training_{metric}"].mean()),
            "mean_improvement": float(improvement.mean()),
            "median_improvement": float(improvement.median()),
            "min_improvement": float(improvement.min()),
            "max_improvement": float(improvement.max()),
            "improved_count": int((improvement > 1e-12).sum()),
            "unchanged_count": int((improvement.abs() <= 1e-12).sum()),
            "regressed_count": int((improvement < -1e-12).sum()),
        }
    return summary


def save_comparison(
    comparison: pd.DataFrame,
    output_directory: str | Path,
    supervised_path: Path | None = None,
    self_training_path: Path | None = None,
) -> dict[str, Path]:
    """Write the comparison CSV and a structured JSON summary."""
    directory = _resolve(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "supervised_vs_self_training.csv"
    comparison.to_csv(csv_path, index=False)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "supervised_source": supervised_path.as_posix() if supervised_path else None,
        "self_training_source": (
            self_training_path.as_posix() if self_training_path else None
        ),
        "summary": summarise_comparison(comparison),
        "comparison": comparison.to_dict(orient="records"),
    }
    json_path = directory / "supervised_vs_self_training.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return {"csv": csv_path, "json": json_path}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m src.comparison``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Phase 5 supervised and Phase 6 self-training results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    supervised, supervised_path = load_phase5_results(config)
    self_training, self_training_path = load_phase6_results(config)
    comparison = build_comparison(supervised, self_training)
    output_directory = (
        _resolve(
            config.get("results", {}).get(
                "self_training_dir", "results/self_training"
            )
        )
        / "tables"
    )
    paths = save_comparison(
        comparison, output_directory, supervised_path, self_training_path
    )
    summary = summarise_comparison(comparison)
    print(f"Matched experiments : {summary['matched_experiments']}")
    print(f"Comparable          : {summary['comparable_experiments']}")
    if "macro_f1" in summary:
        print(
            "Macro-F1 improvement: mean="
            f"{summary['macro_f1']['mean_improvement']:+.4f}, "
            f"improved={summary['macro_f1']['improved_count']}, "
            f"regressed={summary['macro_f1']['regressed_count']}"
        )
    print(f"CSV  : {paths['csv']}")
    print(f"JSON : {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
