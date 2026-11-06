"""Structural and configuration smoke tests for the Phase 1 scaffold.

These tests verify that the expected project layout exists and that the YAML
configuration is valid. Model and pipeline tests are intentionally deferred.

Run from the project root:
    pytest tests/
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment_runner import load_config
from src.preprocessing import validate_labelled_proportion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

EXPECTED_DIRECTORIES = [
    "data",
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "config",
    "results",
    "results/metrics",
    "results/figures",
    "results/reports",
    "dashboard",
    "tests",
]

EXPECTED_FILES = [
    ".gitignore",
    "requirements.txt",
    "README.md",
    "data/README.md",
    "config/config.yaml",
    "src/__init__.py",
    "src/data_loader.py",
    "src/preprocessing.py",
    "src/dimensionality_reduction.py",
    "src/classifiers.py",
    "src/self_training.py",
    "src/evaluation.py",
    "src/experiment_runner.py",
    "src/visualization.py",
    "dashboard/app.py",
    "tests/__init__.py",
    "tests/test_project.py",
]


@pytest.mark.parametrize("relative_path", EXPECTED_DIRECTORIES)
def test_expected_directories_exist(relative_path: str) -> None:
    assert (PROJECT_ROOT / relative_path).is_dir(), f"Missing directory: {relative_path}"


@pytest.mark.parametrize("relative_path", EXPECTED_FILES)
def test_expected_files_exist(relative_path: str) -> None:
    assert (PROJECT_ROOT / relative_path).is_file(), f"Missing file: {relative_path}"


def test_config_loads_and_has_required_sections() -> None:
    config = load_config(CONFIG_PATH)

    assert config["project"]["random_seed"] == 42
    assert 0.0 < config["data"]["test_size"] < 1.0
    assert config["dimensionality_reduction"]["methods"] == ["none", "pca", "lda"]
    assert sorted(config["classifiers"]) == ["knn", "logistic_regression", "svm"]
    assert 0.0 < config["self_training"]["confidence_threshold"] <= 1.0


def test_configured_labelled_proportions_are_valid() -> None:
    config = load_config(CONFIG_PATH)
    proportions = config["data"]["labelled_proportions"]

    assert proportions == [0.10, 0.25, 0.50]
    for proportion in proportions:
        assert validate_labelled_proportion(proportion) == proportion


def test_invalid_labelled_proportions_are_rejected() -> None:
    for invalid in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            validate_labelled_proportion(invalid)


@pytest.mark.skip(reason="Pipeline smoke test arrives in Phase 2.")
def test_end_to_end_pipeline_smoke() -> None:
    """Placeholder for a future small end-to-end pipeline test."""
