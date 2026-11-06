"""Streamlit dashboard for exploring experiment results.

Planned views:

* Headline metric comparisons across reduction methods.
* Classifier-by-DR heatmaps for each labelled proportion.
* Self-training versus supervised-only deltas.
* Confusion matrices and per-dataset error analysis.

Phase 1: static placeholder. No results are loaded and the dashboard is not
started as part of project initialization.

Run later with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "experiment_metrics.csv"


def main() -> None:
    """Render the dashboard shell."""
    st.set_page_config(
        page_title="DR x Semi-Supervised Classification",
        layout="wide",
    )

    st.title("Dimensionality Reduction in a Semi-Supervised Setting")
    st.caption(
        "Impact of PCA/LDA on Logistic Regression, SVM, and KNN under "
        "self-training with limited labelled data."
    )

    st.info(
        "Phase 1 scaffold: experiment results are not available yet. "
        "Complete the data pipeline and run the experiment grid before using "
        "this dashboard."
    )

    st.subheader("Planned views")
    st.markdown(
        """
        - Metric comparisons across reduction methods
        - Classifier x dimensionality-reduction heatmaps
        - Supervised-only vs. self-training deltas
        - Confusion matrices and per-dataset error analysis
        """
    )

    # TODO(Phase 2): load METRICS_PATH into a dataframe when it exists, add
    # dataset/classifier/reduction filters, and render the planned charts.
    # TODO(Phase 2): show a clear empty-state message when no results exist.
    if not METRICS_PATH.exists():
        st.warning(f"Results file not found yet: {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
