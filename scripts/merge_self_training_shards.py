"""Merge sharded Phase 6 self-training results into the canonical layout.

Shards exist only to shorten wall-clock time: each one runs the same
experiments with the same seed and writes to a private ``_shard_*`` directory.
This script moves the shard prediction, iteration-log, and pseudo-label files
into the standard ``results/self_training`` directories, rewrites the stored
paths, and rebuilds the canonical results CSV. Rows already present in the
canonical CSV are kept as-is, so re-running the merge is safe.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.self_training_runner import (  # noqa: E402
    _ordered_frame,
    _write_results,
    load_config,
    self_training_layout,
)

ARTIFACT_COLUMNS = (
    ("prediction_file", "predictions"),
    ("pseudo_label_file", "pseudo_labels"),
    ("iteration_log_file", "iterations"),
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def merge(config_path: Path, remove_shards: bool) -> int:
    config = load_config(config_path)
    layout = self_training_layout(config)
    base = layout["base"]
    main_csv = layout["tables"] / "self_training_results.csv"
    if not main_csv.is_file():
        raise FileNotFoundError(f"Canonical results table not found: {main_csv}")

    canonical = pd.read_csv(main_csv)
    rows = {
        str(row["experiment_id"]): dict(row)
        for row in canonical.to_dict(orient="records")
    }
    print(f"Canonical rows before merge: {len(rows)}")

    shard_directories = sorted(
        path for path in base.glob("_shard_*") if path.is_dir()
    )
    if not shard_directories:
        print("No shard directories found; nothing to merge.")
        return 0

    added = 0
    for shard in shard_directories:
        shard_csv = shard / "tables" / "self_training_results.csv"
        if not shard_csv.is_file():
            print(f"WARNING: no results table in {shard}; skipped.")
            continue
        frame = pd.read_csv(shard_csv)
        moved_here = 0
        for row in frame.to_dict(orient="records"):
            identifier = str(row["experiment_id"])
            if identifier in rows:
                continue
            for column, directory in ARTIFACT_COLUMNS:
                source = PROJECT_ROOT / str(row[column])
                if not source.is_file():
                    raise FileNotFoundError(
                        f"{identifier}: missing {column} artifact {source}"
                    )
                destination = layout[directory] / source.name
                if destination.exists():
                    raise FileExistsError(
                        f"{identifier}: canonical artifact already exists "
                        f"at {destination}"
                    )
                shutil.move(str(source), str(destination))
                row[column] = _relative(destination)
            rows[identifier] = row
            moved_here += 1
        added += moved_here
        print(f"Merged {moved_here} row(s) from {shard.name}")

    frame = _ordered_frame(list(rows.values()))
    if len(frame) != 81:
        print(
            f"WARNING: merged table has {len(frame)} rows; expected 81. "
            "Run the resume step to fill the remaining conditions."
        )
    _write_results(frame, main_csv)
    print(f"Canonical rows after merge : {len(frame)} ({added} added)")

    if remove_shards:
        for shard in shard_directories:
            shutil.rmtree(shard)
            print(f"Removed shard directory: {shard.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Do not delete the _shard_* directories after a successful merge.",
    )
    args = parser.parse_args(argv)
    return merge(args.config, remove_shards=not args.keep_shards)


if __name__ == "__main__":
    raise SystemExit(main())
