# Project Demonstration Guide

**Project:** Impact of Dimensionality Reduction on the Performance of Subsequent Supervised Classifiers in a Semi-Supervised Setting

This guide walks through a 10–12 minute demonstration using the saved results. **Do not retrain any model during the demonstration.** Every artifact listed here already exists and has been verified.

---

## 1. Before the demonstration

### 1.1 Check the environment

```bash
cd D:/ml_project
python --version
python -c "import pandas, sklearn, matplotlib, pypdf; print('core imports OK')"
```

Python 3.12 was used. The project needs `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `PyYAML`, and `joblib`. `pypdf` is only needed to inspect the PDF.

### 1.2 Confirm the key artifacts exist

```bash
ls results/final_report
ls results/final_report/figures
ls results/final_report/tables
```

Expected: the Markdown report, PDF, executive summary, comparison CSV/JSON, limitations document, verification report, 11 figures, and 7 summary tables.

### 1.3 Open these windows in advance

1. `final_research_report.md` (or the PDF) in a viewer.
2. `results/final_report/figures/` in a file browser.
3. `results/final_report/final_results_comparison.csv` in a spreadsheet.
4. `final_project_presentation.pptx` in presentation mode.

---

## 2. Ten-minute demonstration script

### Minute 0–1 — Framing

Open slide 1 of `final_project_presentation.pptx`.

> “This project asks whether PCA or LDA before classification changes how much self-training helps when labels are scarce. I ran a fully crossed design: 3 datasets, 3 label budgets, 3 representations, and 3 classifiers, for 81 supervised and 81 matched self-training conditions.”

### Minute 1–2 — Data and design

Advance to slides 3–5.

> “Breast Cancer is small and binary, MNIST is high-dimensional with 10 classes, and HAR uses a subject-disjoint split. Every condition shares the same split, seed, scaling scope, and test rows.”

### Minute 2–3 — PCA and LDA

Advance to slides 7–8.

> “PCA keeps 95% of variance and reduces 30→10, 784→318, and 561→102 features. LDA is fitted per labelled proportion and is capped at `n_classes − 1`, giving 1, 9, and 5 components.”

Mention the risk explicitly: LDA produced extreme out-of-sample values and three MNIST LDA + Logistic Regression fits became numerically degenerate.

### Minute 3–4 — Self-training procedure

Advance to slides 10–11.

> “Each round fits on labelled rows, scores the unlabelled pool, accepts predictions at or above 0.90, adds them, and refits. The maximum is three rounds. LR and KNN use `predict_proba`; the linear SVM uses a Platt calibration fitted on a labelled-only 20% holdout.”

Show `fig09_pseudo_label_acceptance.png`:
> “Overall acceptance was 65.13% of the pooled unlabelled rows; the mean per-condition rate was 73.39%.”

### Minute 4–6 — Results

Open `fig01_accuracy_supervised_vs_self_training.png` and then `fig02_macro_f1_supervised_vs_self_training.png`.

> “Mean accuracy moved from 0.90894 to 0.88836, and mean macro-F1 from 0.90626 to 0.88261. On macro-F1, 38 conditions improved, 27 regressed, and 16 were unchanged.”

Then open the comparison CSV and filter `numerically_degenerate_fit == True`.

> “The three MNIST LDA + LR fits collapsed. Excluding them, the remaining 78 conditions improve by +0.00379 macro-F1, so the aggregate negative sign is a numerical artifact, not a general property of self-training.”

### Minute 6–7 — Best configurations

Advance to slide 12 or open Section 11 of the report.

| Dataset | Best supervised | Best self-training |
|---|---|---|
| Breast Cancer | Baseline + LR, 25% — 0.9905 macro-F1 | Baseline + SVM, 10% — 0.9812 |
| MNIST | PCA + KNN, 50% — 0.9337 | PCA + KNN, 50% — 0.9331 |
| HAR | LDA + KNN, 25% — 0.9618 | LDA + KNN, 25% — 0.9610 |

### Minute 7–8 — Cost

Advance to the runtime section or open `fig07_training_time.png`.

> “Supervised training took 767.3 seconds; self-training took 5,826.3 seconds, 7.59 times more. MNIST linear-SVM fits alone used 91.3% of that time. Phase 5 did not record prediction time, so that comparison is not claimed.”

### Minute 8–9 — Limitations

Advance to slide 13 and open `limitations_and_future_work.md`.

State the four must-mention limits:
1. Three degenerate MNIST LDA + LR fits.
2. 16 Phase 5 and 17 Phase 6 convergence warnings.
3. Three-iteration cap: 68 cap hits and 13 confidence stops.
4. One fixed split and seed: descriptive statistics only, no p-values.

### Minute 9–10 — Verification and close

Open `results/final_report/phase7_verification_report.txt`.

> “Phase 7 verification passed 1,769 of 1,769 checks, including matched configurations, recomputed metrics, and Phase 1–6 immutability over 692 files.”

Close on slide 15: “The practical conclusion is that self-training was roughly neutral overall, PCA did not systematically break it, and LDA plus a linear probabilistic model is the combination to treat with caution.”

---

## 3. What to show if asked for evidence

| Question | File to open |
|---|---|
| “Show me the raw numbers.” | `results/final_report/final_results_comparison.csv` |
| “Show me the full analysis.” | `results/final_report/final_research_report.md` |
| “Show me the best configurations.” | `results/final_report/tables/summary_best_configurations.csv` |
| “Show me the warnings.” | `results/final_report/tables/summary_warnings.csv` |
| “Show me the degenerate fits.” | `results/final_report/tables/summary_degenerate_fits.csv` |
| “Show me the runtime.” | `results/final_report/tables/summary_runtime.csv` |
| “Show me the pseudo-label statistics.” | `results/final_report/tables/summary_pseudo_labels.csv` |
| “Show me that nothing changed.” | `results/final_report/provenance/phase1_phase6_snapshot.json` |

---

## 4. Optional live checks (safe, read-only)

These commands do not train models and do not rewrite experimental results.

```bash
# Re-run the Phase 7 verifier (reads existing files, regenerates only temporary documents)
python scripts/verify_phase7.py

# Quick count check
python -c "import pandas as pd; d=pd.read_csv('results/final_report/final_results_comparison.csv'); print(len(d), d.numerically_degenerate_fit.sum())"

# Inspect the report headings
grep "^## " results/final_report/final_research_report.md
```

Do **not** run `src.experiment_runner`, `scripts/run_self_training.py`, or any preprocessing script during the demonstration.

---

## 5. Troubleshooting

- **PDF does not open:** use `final_research_report.html` as a fallback; it is the print source for the PDF.
- **PPTX does not open:** the file was saved and re-opened by the presentation engine; if the viewer is old, open it in PowerPoint or LibreOffice. The Markdown report contains the same content.
- **Figures look small:** each PNG is 300 DPI and roughly 3,000 pixels wide; zoom in rather than resizing the window.
- **A number is questioned:** open the comparison CSV and recompute it directly; every improvement is `self-training − supervised`.
- **Time is short:** show slides 1, 11, 12, 13, and 15 only.

---

## 6. Hand-off checklist

- [ ] PPTX opens on the presentation machine.
- [ ] PDF or HTML report opens.
- [ ] `results/final_report/figures/` is reachable.
- [ ] Comparison CSV opens in a spreadsheet.
- [ ] Verification report is open in a background window.
- [ ] No training command is running.
