"""Build the Phase 7 analysis tables from the completed Phase 5/6 results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.final_analysis import (  # noqa: E402
    AnalysisError,
    build_analysis,
    build_final_comparison,
    load_sources,
    validate_sources,
    write_comparison_outputs,
    write_immutability_snapshot,
)
from src.experiment_runner import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Phase 7 comparison tables from saved results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="Path to the project configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "final_report",
        help="Directory for final-report outputs.",
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Do not refresh the Phase 1-6 immutability snapshot.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = output_dir / "provenance" / "phase1_phase6_snapshot.json"
    if not args.skip_snapshot:
        snapshot = write_immutability_snapshot(snapshot_path)
        print(
            f"Immutability snapshot: {snapshot['file_count']} files, "
            f"{snapshot['total_size_bytes'] / (1024 ** 2):.1f} MiB"
        )

    try:
        sources = load_sources(config)
        validation = validate_sources(sources)
        payload = build_analysis(sources)
    except AnalysisError as error:
        print(f"ANALYSIS FAILED: {error}", file=sys.stderr)
        return 1

    frame = build_final_comparison(sources["phase5"], sources["phase6"])
    paths = write_comparison_outputs(frame, payload, output_dir)

    overall = payload["overall"]["macro_f1"]["all_conditions"]
    print(f"Validation checks passed : {validation['passed']}")
    print(f"Matched conditions       : {validation['matched_experiments']}")
    print(
        "Macro-F1 (all 81)        : "
        f"supervised={overall['mean_supervised']:.5f}, "
        f"self-training={overall['mean_self_training']:.5f}, "
        f"difference={overall['mean_difference']:+.5f}"
    )
    runtime = payload["runtime"]
    print(
        "Training time            : "
        f"supervised={runtime['supervised_training_seconds_total']:.2f}s, "
        f"self-training={runtime['self_training_training_seconds_total']:.2f}s"
    )
    print(f"Comparison CSV           : {paths['csv']}")
    print(f"Comparison JSON          : {paths['json']}")
    print(f"Snapshot                 : {snapshot_path}")
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        summary = {
            "phase": payload["phase"],
            "generated_utc": payload["generated_utc"],
            "validation": {
                key: validation[key]
                for key in ("expected_experiments", "matched_experiments", "passed", "failed")
            },
            "overall": payload["overall"],
            "runtime": payload["runtime"],
            "pseudo_labels": payload["pseudo_labels"],
            "warnings": payload["warnings"],
            "best_configurations": payload["best_configurations"],
        }
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Analysis summary         : {output_dir / 'analysis_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
