# Impact of Dimensionality Reduction on the Performance of Subsequent Supervised Classifiers in a Semi-Supervised Setting

## Project status

**Phases 1–8 complete and verified.**

| Phase | Scope | Verification |
|---|---|---|
| 1–2 | Structure, environment, dataset staging | datasets checked |
| 3 | Preprocessing, scaling, fixed splits, labelled/unlabelled partitions | passed |
| 4 | Baseline, PCA, and LDA reduction (27 configurations) | 78/78 |
| 5 | 81 supervised classifier experiments | 66/66 |
| 6 | 81 matched self-training experiments | 86/86 |
| 7 | Final analysis, comparison tables, 11 figures, 19-section report, PDF | 1769/1769 |
| 8 | Submission package: presentation, viva preparation, final verification | `scripts/verify_final_project.py` |

All 162 experiment runs (81 supervised + 81 self-training) completed with no
failed runs. Every Phase 6 result is matched to its Phase 5 counterpart by
dataset, labelled percentage, classifier, and feature representation.

The final deliverables are in [`results/final_report/`](results/final_report/):
the Markdown/PDF research report, executive summary, comparison CSV/JSON,
limitations document, seven summary tables, eleven figures, and the Phase 7
verification report. The Phase 8 package adds
[`final_project_presentation.pptx`](final_project_presentation.pptx),
[`viva_questions_and_answers.md`](viva_questions_and_answers.md),
[`project_demonstration_guide.md`](project_demonstration_guide.md), and
[`final_submission_checklist.md`](final_submission_checklist.md).

See [`data/README.md`](data/README.md) for exact download provenance and manual
fallback instructions.

## Technologies used

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Data handling | NumPy, Pandas |
| Machine learning | scikit-learn 1.6+ (LogisticRegression, LinearSVC, KNN, PCA, LDA, CalibratedClassifierCV, FrozenEstimator) |
| Visualization | Matplotlib, Seaborn (300 DPI PNG) |
| Configuration | PyYAML, Joblib |
| Reporting | Custom Markdown/HTML builders, headless Edge/Chrome for PDF, python-pptx-compatible Office engine for the deck |
| Verification | Custom scripts, SHA-256 fingerprints, `pypdf` for PDF checks |
| Optional dashboard | Streamlit (placeholder) |

## Research objective

Investigate how dimensionality reduction (DR) affects the downstream
performance of supervised classifiers when only a small fraction of the
available data is labelled and the remaining data are used through
self-training.

The study isolates three factors:

1. The dimensionality reduction technique applied before classification.
2. The supervised classification algorithm.
3. The proportion of labelled training data available.

## Problem statement

In many real-world settings, labelled data are scarce or expensive while
unlabelled data are abundant. Semi-supervised learning can exploit unlabelled
samples, but its interaction with preprocessing choices such as dimensionality
reduction is not well understood. DR may remove noise and redundant features,
but it can also discard information that pseudo-labelling relies on. This
project systematically measures that interaction across datasets,
classifiers, DR methods, and labelled-data proportions.

## Datasets

| Dataset | Description | Source |
|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | Binary classification of malignant vs. benign tumours from numeric cell-nucleus features. | Kaggle `uciml/breast-cancer-wisconsin-data` |
| MNIST Digit Recognizer | Handwritten digit images (10 classes) represented as pixel features. | Kaggle Digit Recognizer competition files |
| Human Activity Recognition (HAR) | Smartphone sensor windows classified into six activity classes. | UCI ML Repository dataset 240 (UCI-owned Kaggle copy) |

Datasets are **not** committed to this repository. The staged files and their
provenance are documented in [`data/README.md`](data/README.md).

## Dimensionality reduction techniques

- **None (baseline)** — classifiers operate on the original scaled feature
  space (30 / 784 / 561 features).
- **Principal Component Analysis (PCA)** — unsupervised linear projection that
  retains a configured fraction of explained variance (95% by default).
  Fractional targets require the full SVD solver; `svd_solver: auto` is
  resolved to `full` and recorded rather than silently changing the target.
- **Linear Discriminant Analysis (LDA)** — supervised linear projection that
  maximizes class separability, with at most `n_classes - 1` components.

Fitting scopes are enforced by the implementation and verified by the Phase 4
script:

| Method | Fitted on | Reused across proportions | Test data |
|---|---|---|---|
| Baseline | No fitting | Yes | Never fitted on; unchanged features |
| PCA | Complete training pool (labelled + unlabelled) | Yes, one fit per dataset | Never fitted on; transformed only |
| LDA | Labelled training rows and labels only, per proportion | No, refitted at 10% / 25% / 50% | Never fitted on; transformed only |

Measured with seed 42 (variance retained and component counts are recorded in
`results/dimensionality_reduction/metadata/`):

| Dataset | Original | PCA components (variance) | LDA components per proportion |
|---|---|---|---|
| Breast Cancer | 30 | 10 (95.2069%) | 1 / 1 / 1 |
| MNIST | 784 | 318 (95.0256%) | 9 / 9 / 9 |
| HAR | 561 | 102 (95.0756%) | 5 / 5 / 5 |

## Classification algorithms

Phase 5 uses three supervised classifiers with fixed, untuned settings:

| Classifier | Configuration | Notes |
|---|---|---|
| Logistic Regression | `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)` | Handles binary and multiclass targets |
| SVM (linear) | `LinearSVC(C=1.0, loss="squared_hinge", dual=True, tol=1e-3, max_iter=2000, random_state=42)` | Deliberately **linear** for tractability on MNIST; hard labels only |
| K-Nearest Neighbors | `KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2, algorithm="brute")` | Consistent Euclidean distance; no tuning |

Each experiment trains on the labelled subset only and records the model-fitting
time (loading, preprocessing, and dimensionality-reduction times are excluded).
`LinearSVC` has no `predict_proba`. Phase 6 therefore wraps it in a
sigmoid-calibrated classifier for confidence estimation; final test predictions
still come from the Phase 5 `LinearSVC` configuration refitted on all labelled
plus pseudo-labelled rows.

## Semi-supervised learning approach

### Supervised versus semi-supervised experiments

Phase 5 is the **supervised** reference: each classifier sees only the labelled
training rows of its proportion. Phase 6 is **semi-supervised**: the same
classifier additionally sees the unlabelled training rows through
pseudo-labels, but never their true labels. Both phases use the same fixed
Phase 3 split, the same Phase 4 feature representation, the same classifier
configuration, and the same held-out test rows, so the only difference is the
self-training loop.

### Algorithm

For each of the 81 conditions:

1. Fit the classifier on the original labelled rows only.
2. Predict class probabilities for the remaining unlabelled training rows.
3. Accept rows whose maximum predicted probability is at least **0.90**.
4. Assign the predicted class as the pseudo-label, remove those rows from the
   unlabelled pool, and append them to the training set.
5. Refit on the original labelled rows plus all accumulated pseudo-labelled
   rows.
6. Repeat up to **3** iterations (`max_iterations` in `config/config.yaml`),
   stopping early when a round accepts fewer than **1** new sample or the pool
   is exhausted. A 10-round run is preserved under
   `results/self_training_max10_verified/`.

The seed is **42**, and the threshold is fixed for every experiment. It is
never tuned on test performance or on hidden unlabelled labels.

### Confidence estimation per classifier

| Classifier | Confidence source |
|---|---|
| Logistic Regression | `predict_proba`; maximum class probability |
| KNN | `predict_proba`; maximum class probability (distance-weighted neighbour vote) |
| Linear SVM | Sigmoid (Platt) calibration fitted only on a class-stratified holdout of the **original labelled rows**; maximum calibrated class probability |

The calibrated SVM uses `FrozenEstimator` to keep the Phase 5 `LinearSVC`
configuration fixed while calibration folds are fitted with `StratifiedKFold`.
Calibration rows are never pseudo-labelled, and the calibration subset is small
for Breast Cancer at 10%, so its probabilities are approximate. **Confidence
scores from different classifiers are not on identical scales and are not
treated as directly comparable.**

### Stopping conditions

A run stops at the first condition that holds:

- no unlabelled sample reaches the confidence threshold;
- fewer than the minimum number of new pseudo-labels are accepted;
- the maximum iteration count is reached;
- the unlabelled pool is empty.

The reason is stored as `stopping_reason` in every result row.

### Leakage prevention

1. Only labelled training rows and labelled targets fit the initial model.
2. Unlabelled true labels are never read by the self-training code.
3. Test rows never enter fitting, pseudo-labelling, or stopping decisions.
4. Test labels never select pseudo-labels or determine when training stops.
5. No preprocessing or DR transformation is fitted on test data.
6. MNIST uses only the 42,000 competition-training rows; the competition
   `test.csv` is never used.
7. HAR keeps the official subject-disjoint train/test split.

The self-training module receives `X_unlabelled` but no unlabelled labels.
Pseudo-label positions always refer to the original unlabelled pool, are
unique, and are removed from the pool once accepted.

### Feature representations

- **Baseline** — the existing Phase 3 standardised features.
- **PCA** — the Phase 4 PCA fitted once per dataset on the complete training
  pool. The same projection transforms labelled and unlabelled rows.
- **LDA** — the Phase 4 LDA fitted once per labelled proportion on labelled
  rows only. The same projection transforms the unlabelled pool and the test
  set; it is never refitted.

### Pseudo-label management

The Phase 3 split is never modified. Pseudo-labels are stored separately from
the original labels, and every accepted row keeps its original unlabelled
position and dataset index. There are no duplicates, no row is both labelled
and pseudo-labelled, and no row is forced into the training set to reach a
target count. The threshold is never lowered automatically.

### Recorded diagnostics

Every experiment stores per-iteration logs (added, cumulative, remaining,
mean/min/max accepted confidence, training-set size, pool considered), a
pseudo-label audit (position, train index, dataset index, pseudo-label,
confidence, iteration, source), and the final stopping reason. A
`numerically_degenerate_fit` flag marks final fits whose probabilities are
indistinguishable from uniform. This occurs for some MNIST LDA + Logistic
Regression conditions because the Phase 4 LDA projection produces extreme
out-of-sample values; the measured metrics are still reported and flagged
rather than hidden or silently replaced.

## Phase 5 results (supervised baseline)

All **81/81** experiments completed with no failures. Full tables, the results
report, per-experiment test predictions, and 15 figures are under
`results/classifier_evaluation/`.

Best configuration per dataset by macro F1:

| Dataset | Best configuration | Accuracy | Macro F1 |
|---|---|---|---|
| Breast Cancer | Baseline + Logistic Regression, 25% labelled | 0.9912 | 0.9905 |
| MNIST | PCA + KNN, 50% labelled | 0.9344 | 0.9337 |
| HAR | LDA + KNN, 25% labelled | 0.9620 | 0.9618 |

Mean macro F1 across all experiments: baseline 0.9157, PCA 0.9161, LDA 0.8870.
Performance rises with labelled data (macro F1 0.8754 at 10%, 0.9155 at 25%,
0.9279 at 50%). Logistic Regression has the highest mean macro F1 (0.9230),
followed by KNN (0.9035) and the linear SVM (0.8923). Sixteen linear-SVM fits
(8 on MNIST, 8 on HAR) reached the 2,000-iteration cap and raised
`ConvergenceWarning`; they still produced valid predictions and metrics, and the
warnings are recorded per experiment in the results CSV.

## Phase 6 results (self-training)

All **81/81** experiments completed with no failed runs: 27 per dataset across
10/25/50% labelled data, three representations, and three classifiers. Full
tables, the matched comparison, an 18-section report, 81 test-prediction
files, 81 per-iteration logs, 81 pseudo-label audits, and 18 figures are under
`results/self_training/`.

This run uses the speed setting **`max_iterations = 3`** with the unchanged
0.90 confidence threshold. A previous verified run with 10 rounds is preserved
under `results/self_training_max10_verified/` for comparison.

Self-training mean macro F1 is **0.8826** versus **0.9063** for the matched
supervised runs: mean improvement **-0.0237**, median **0.0000**. It improved
38 conditions, left 16 unchanged, and regressed 27. The negative mean is
dominated by three MNIST LDA + Logistic Regression fits whose final optimizer
state is numerically degenerate (flagged in the results table; see below).
Excluding those three, the mean improvement is **+0.0038** macro F1, which is
essentially neutral.

Best configuration per dataset by macro F1:

| Dataset | Best configuration | Accuracy | Macro F1 |
|---|---|---|---|
| Breast Cancer | Baseline + SVM, 10% labelled | 0.9825 | 0.9812 |
| MNIST | PCA + KNN, 50% labelled | 0.9339 | 0.9331 |
| HAR | LDA + KNN, 25% labelled | 0.9610 | 0.9610 |

Mean self-training macro F1 by factor: baseline 0.9164, PCA 0.9158, LDA
0.8156; KNN 0.9014, SVM 0.9048, LR 0.8416; 10% 0.8570, 25% 0.8956, 50%
0.8952.

Pseudo-labelling accepted 521,861 samples in total (65.13% of all pooled
unlabelled rows). Mean acceptance is 84.8% for KNN and 85.0% for logistic
regression, but 25.6% for the calibrated SVM because its probability scale is
different. The average run completed 2.69 rounds; 13 runs stopped when no
remaining sample reached the 0.90 threshold and 68 reached the three-round
cap.

Mean matched improvements by factor (macro F1, all 81 conditions): baseline
+0.0007, PCA -0.0003, LDA -0.0714; SVM +0.0125, KNN -0.0020, LR -0.0814;
Breast Cancer -0.0040, HAR +0.0003, MNIST -0.0673. Removing the three
degenerate fits leaves overall **+0.0038**, so the apparent LDA and Logistic
Regression penalty is a numerical artifact rather than a property of
self-training.

The largest genuine improvement is MNIST 10% LDA + SVM (macro F1 0.4923 to
0.7593, +0.2670). The largest regression is the degenerate MNIST 50% LDA + LR
fit (0.8740 to 0.0181).

Three MNIST LDA + Logistic Regression conditions collapsed: the Phase 4 LDA
projection produces out-of-sample values up to 2.7e15, the optimizer returns
coefficients near 1e-13, and predicted probabilities become uniform (0.1).
Those runs are marked `numerically_degenerate_fit` with a stored diagnostic,
and their measured metrics are reported as observed rather than fabricated or
replaced. No Phase 4 or Phase 5 artifact is changed and no feature rescaling
is applied in Phase 6.

Mean training time per run is 0.04 s (Breast Cancer), 3.83 s (HAR), and
211.92 s (MNIST); mean test-prediction time is 0.003 s, 0.058 s, and 1.052 s
respectively. Training time includes every self-training refit. The full
three-round wall time was 5,899.9 s. The MNIST linear-SVM fits dominate
(5,318.8 s), because each experiment still performs one initial fit, up to
three augmented refits, and a final full-data fit; liblinear convergence time
varies sharply with the pseudo-labelled set.

## Phase 7 results (final analysis)

The Phase 7 analysis joins the 81 Phase 5 rows to the 81 Phase 6 rows on the
full condition key and recomputes every comparison from the saved tables. It
makes no new measurements.

- **Overall**: mean accuracy **0.90894 → 0.88836** (−0.02058) and mean
  macro-F1 **0.90626 → 0.88261** (−0.02365). On macro-F1, 38 conditions
  improved, 27 regressed, and 16 were unchanged.
- **Sensitivity**: excluding the three numerically degenerate MNIST LDA +
  Logistic Regression fits leaves 78 conditions with **+0.00379** macro-F1
  (0.90843 → 0.91222), so the negative aggregate sign is driven by those fits.
- **Largest genuine gain**: MNIST, 10% labelled, LDA + Linear SVM,
  macro-F1 0.4923 → 0.7593 (**+0.26698**).
- **Factors** (mean macro-F1 change): Baseline +0.0007, PCA −0.0003, LDA
  −0.0714; Linear SVM +0.0125, KNN −0.0020, Logistic Regression −0.0814.
- **Cost**: supervised training totalled 767.3 s and self-training 5,826.3 s
  (**7.59×**); MNIST linear SVM alone used 91.3% of self-training time. Phase 5
  did not record prediction time.
- **Pseudo-labels**: 521,861 of 801,216 rows accepted (65.13% pooled; 73.39%
  mean per condition); 68 runs hit the 3-iteration cap and 13 stopped at the
  confidence threshold.
- **Statistics**: one fixed split and seed, so the analysis reports descriptive
  paired differences and Cohen's d_z only — no p-values or confidence intervals.

Best configurations by macro-F1:

| Dataset | Supervised | Self-training |
|---|---|---|
| Breast Cancer | Baseline + LR, 25% — 0.9905 | Baseline + SVM, 10% — 0.9812 |
| MNIST | PCA + KNN, 50% — 0.9337 | PCA + KNN, 50% — 0.9331 |
| HAR | LDA + KNN, 25% — 0.9618 | LDA + KNN, 25% — 0.9610 |

## Phase 8 submission package

The final package reuses the verified Phase 7 artifacts and adds:

| File | Purpose |
|---|---|
| `results/final_report/final_research_report.md` / `final_research_report.pdf` | Full 19-section research report (PDF renders the complete report) |
| `final_project_presentation.pptx` | 15-slide project presentation with existing result figures |
| `viva_questions_and_answers.md` | 50 viva questions with concise, factual answers |
| `project_demonstration_guide.md` | 10–12 minute demonstration script and file map |
| `final_submission_checklist.md` | Deliverable, quality, integrity, and sign-off gates |
| `scripts/verify_final_project.py` | Verifies the complete package and writes `final_verification_report.txt` |

No model was retrained, no dataset was reloaded for analysis, and no Phase 1–6
artifact was modified while building the package.

## Evaluation metrics

Both phases report test-set metrics for every experiment:

- **Accuracy** — fraction of correctly classified test samples.
- **Macro F1** — unweighted mean of per-class F1 scores. Macro averaging gives
  each class equal influence, which matters when class frequencies differ.

Phase 5 records the time spent in `model.fit` only. Phase 6 records
`training_time_seconds` for the whole self-training loop (initial fit plus
every refit) and `prediction_time_seconds` separately for the final test
prediction. Phase 6 also reports the number of pseudo-labels, the number of
remaining unlabelled rows, the number of iterations, the pseudo-label
acceptance rate, and the stopping reason.

Metrics are computed on the held-out test set that is never used for training,
pseudo-labelling, or dimensionality-reduction fitting. Multiclass targets are
handled by scikit-learn's multiclass accuracy and macro-F1 implementation.

## Project workflow

1. **Data acquisition** — download and stage the Breast Cancer, MNIST, and HAR
   files under `data/raw/`, then load and validate them with
   `src/data_loader.py`. *(Phases 1–2 complete; datasets are not re-downloaded.)*
2. **Preprocessing** — clean, validate, and scale features; create stratified
   train/test partitions. *(Phase 3 complete.)*
3. **Labelled/unlabelled partitioning** — create labelled subsets at 10%, 25%,
   and 50% of the non-test data; the remainder is the unlabelled pool. *(Phase 3
   complete.)*
4. **Dimensionality reduction** — fit each DR configuration on the permitted
   training data and transform train, unlabelled, and test sets. *(Phase 4
   complete.)*
5. **Baseline supervised training** — train Logistic Regression, linear SVM, and
   KNN on labelled data only. *(Phase 5 complete: 81 experiments.)*
6. **Self-training** — run iterative high-confidence pseudo-labelling and
   retraining. *(Phase 6 complete: 81 experiments.)*
7. **Evaluation** — compute the metric suite on the held-out test set for every
   dataset × DR method × classifier × labelled proportion × training regime.
   *(Phases 5–6 complete: 81 + 81 runs.)*
8. **Analysis and reporting** — aggregate metrics into tables and figures,
   generate the research report and PDF, and package the submission. *(Phases
   7–8 complete.)* The Streamlit dashboard remains an optional placeholder.

## Repository structure

```text
ml_project/
├── data/
│   ├── raw/                 # staged source datasets (git-ignored)
│   ├── processed/           # cached processed arrays (git-ignored)
│   └── README.md            # dataset provenance and placement instructions
├── notebooks/               # exploratory analysis and scratch work
├── scripts/
│   ├── check_datasets.py    # load and verify all three datasets
│   ├── verify_preprocessing.py  # splits, scaling scope, leakage checks
│   ├── verify_dimensionality_reduction.py  # 27 DR configurations + artifacts
│   ├── verify_classifiers.py    # 81 supervised experiments + leakage checks
│   ├── run_self_training.py     # Phase 6 grid entry point
│   ├── verify_self_training.py  # 81 self-training experiments + audit checks
│   ├── run_final_analysis.py    # Phase 7 comparison tables + snapshot
│   ├── generate_final_report.py # Phase 7 figures, documents, PDF
│   ├── verify_phase7.py         # Phase 7 metric/document/immutability checks
│   └── verify_final_project.py  # Phase 8 package verification
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # dataset loading, validation, and summaries
│   ├── preprocessing.py     # scaling, splitting, labelled/unlabelled splits
│   ├── dimensionality_reduction.py  # PCA / LDA / baseline handling
│   ├── classifiers.py       # classifier construction and calibrated SVM
│   ├── self_training.py     # confidence-based self-training loop
│   ├── self_training_runner.py  # Phase 6 orchestration and artifacts
│   ├── comparison.py        # Phase 5 vs Phase 6 matched comparison
│   ├── evaluation.py        # metrics, aggregation, persistence
│   ├── experiment_runner.py # configuration loading and Phase 5 grid
│   ├── visualization.py     # plots for figures and reports
│   ├── final_analysis.py    # Phase 7 join, validation, summary statistics
│   └── final_reporting.py   # Phase 7 figures, tables, documents, PDF
├── config/
│   └── config.yaml          # experiment configuration
├── results/
│   ├── metrics/             # metric tables (git-ignored)
│   ├── figures/             # plots (git-ignored)
│   ├── reports/             # generated summaries (git-ignored)
│   ├── dimensionality_reduction/
│   │   ├── metadata/        # 27-configuration JSON/CSV, PCA/LDA detail tables
│   │   ├── figures/         # five Phase 4 reduction figures
│   │   └── reports/         # Phase 4 verification report
│   ├── classifier_evaluation/
│   │   ├── tables/          # classifier_results.csv and .json
│   │   ├── reports/         # Phase 5 results and verification reports
│   │   ├── predictions/     # per-experiment test predictions (81 CSVs)
│   │   └── figures/         # 15 Phase 5 comparison figures
│   ├── self_training/
│   │   ├── tables/          # results, comparison CSV/JSON
│   │   ├── reports/         # Phase 6 results and verification reports
│   │   ├── predictions/     # 81 per-experiment test predictions
│   │   ├── logs/            # iteration logs, pseudo-label audits, run log,
│   │   │                    # Phase 1-5 immutability manifest
│   │   └── figures/         # 18 Phase 6 comparison and diagnostic figures
│   ├── self_training_max10_verified/  # preserved 10-iteration reference run
│   └── final_report/
│       ├── final_research_report.md/.pdf
│       ├── executive_summary.txt
│       ├── final_results_comparison.csv/.json
│       ├── limitations_and_future_work.md
│       ├── phase7_verification_report.txt
│       ├── final_verification_report.txt
│       ├── report_manifest.json
│       ├── figures/         # 11 final-report figures
│       ├── tables/          # 7 summary tables
│       └── provenance/      # Phase 1-6 immutability snapshot
├── dashboard/
│   └── app.py               # Streamlit dashboard (placeholder)
├── tests/
│   ├── __init__.py
│   └── test_project.py      # structural and configuration smoke tests
├── final_project_presentation.pptx
├── viva_questions_and_answers.md
├── project_demonstration_guide.md
├── final_submission_checklist.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The analysis and verification scripts only need the packages already listed in
`requirements.txt`. PDF generation uses a locally installed Microsoft Edge or
Google Chrome in headless mode; no extra Python package is required. `pypdf` is
optional and only used for PDF inspection.

## Usage

```bash
# Verify the staged datasets (no training)
python scripts/check_datasets.py

# Verify preprocessing, scaling scope, and semi-supervised splits
python scripts/verify_preprocessing.py

# Verify all 27 baseline/PCA/LDA configurations and write Phase 4 artifacts
python scripts/verify_dimensionality_reduction.py

# Run all 81 supervised classifier experiments (writes tables, report,
# predictions, and figures under results/classifier_evaluation/)
python -m src.experiment_runner --config config/config.yaml

# Verify coverage, metrics, predictions, dimensions, and leakage controls
python scripts/verify_classifiers.py

# Run all 81 Phase 6 self-training experiments (writes tables, comparison,
# iteration logs, pseudo-label audits, report, and figures under
# results/self_training/). Checkpoints are written after every experiment;
# --resume continues an interrupted run and refuses mismatched run settings.
python scripts/run_self_training.py --config config/config.yaml --resume

# Verify self-training coverage, metrics, pseudo-labels, iteration logs,
# comparison matching, and Phase 1-5 artifact immutability
python scripts/verify_self_training.py

# Phase 7: rebuild the comparison CSV/JSON, tables, figures, Markdown report,
# executive summary, limitations document, and PDF from the saved results
python scripts/run_final_analysis.py
python scripts/generate_final_report.py

# Phase 7: recompute every reported metric and check Phase 1-6 immutability
python scripts/verify_phase7.py

# Phase 8: verify the complete submission package (report, PDF, PPTX,
# figures, README, viva files, 162 configurations, immutability)
python scripts/verify_final_project.py

# Structural tests
pytest tests/

# Launch the dashboard (placeholder)
streamlit run dashboard/app.py
```

None of the Phase 7/8 commands trains a model or modifies a Phase 1–6 result.
They only read the saved tables and metadata.

## Reproducibility

- The global random seed is defined in `config/config.yaml` (currently 42).
- Train/test splits and labelled/unlabelled sampling use that seed.
- `data/processed/preprocessing_splits.json` records dataset sizes, class
  counts, the scaler fit scope, HAR subject IDs, and SHA-256 hashes of the
  split indices. Unlabelled true labels are not written to this file.
- `results/dimensionality_reduction/metadata/reduction_configurations.json`
  records all 27 measured configurations, fitting scopes, component counts,
  explained variance, reduction percentages, and fit/transform times.
- The verification script reruns every reduction with the same seed and
  compares SHA-256 content fingerprints of the transformed matrices and fitted
  transformer parameters (timings excluded). Identical fingerprints confirm
  same-seed reproducibility.
- Every Phase 5 experiment stores SHA-256 fingerprints of its fitting features
  and labels. `scripts/verify_classifiers.py` reconstructs the labelled-only
  inputs from the fixed split and confirms the fingerprints match, which proves
  that no test row and no unlabelled true label entered classifier fitting.
- Phase 5 writes one test-prediction CSV per experiment, so every reported
  metric can be recomputed independently.
- Before the first Phase 6 experiment, the runner writes
  `results/self_training/logs/phase1_phase5_artifact_manifest.json`, which
  records SHA-256, size, and modification time for every Phase 1-5 output
  file. The verifier re-hashes those files and confirms that no Phase 1-5
  result was modified, added, or deleted.
- Every Phase 6 experiment stores SHA-256 fingerprints of its labelled
  features, labelled targets, and unlabelled features. The verifier
  reconstructs the same views from the fixed split and compares the
  fingerprints, which proves that only the intended labelled and unlabelled
  training rows entered self-training. Randomized PCA and the LDA SVD solver
  change their last bits when BLAS partitions work across a different number
  of threads, so the verifier accepts a fingerprint that is bit-identical
  under either of the two thread settings Phase 6 used (the default pool and a
  pinned two-thread pool); row membership and fit scope are verified
  independently in both cases.
- Phase 6 writes one test-prediction CSV, one per-iteration log, and one
  pseudo-label audit per experiment. Audits record sample indices and
  predicted pseudo-labels only; they never contain the hidden unlabelled
  ground truth.
- The supervised-versus-self-training comparison stores both source files and
  recomputes every improvement as `self-training - supervised`; the JSON
  companion is written with `allow_nan=False`, so incomplete matches are
  explicit rather than silently written as invalid numbers.
- All experiment parameters are centralized in `config/config.yaml`.
