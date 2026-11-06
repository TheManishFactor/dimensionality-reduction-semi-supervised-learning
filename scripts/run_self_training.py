"""Run the Phase 6 semi-supervised self-training experiment grid.

Usage (from the project root):

    python scripts/run_self_training.py
    python scripts/run_self_training.py --resume

The runner writes results, predictions, iteration logs, pseudo-label audits,
figures, and the Phase 6 report under ``results/self_training/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.self_training_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
