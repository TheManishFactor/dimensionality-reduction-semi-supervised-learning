"""Source package for the dimensionality-reduction / semi-supervised study.

Data loading, preprocessing/splitting, and dimensionality reduction (baseline,
PCA, LDA) are implemented. Classification, self-training, evaluation metrics,
and the dashboard remain for later phases.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "classifiers",
    "data_loader",
    "dimensionality_reduction",
    "evaluation",
    "experiment_runner",
    "preprocessing",
    "self_training",
    "visualization",
]
