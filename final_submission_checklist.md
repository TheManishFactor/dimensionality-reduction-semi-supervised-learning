# Final Submission Checklist

**Project:** Impact of Dimensionality Reduction on the Performance of Subsequent Supervised Classifiers in a Semi-Supervised Setting

Use this checklist before handing in the project or sitting the viva. Commands are read-only unless stated otherwise.

---

## 1. Required deliverables

| # | Deliverable | Path | Required |
|---|---|---|---|
| 1 | Final research report (Markdown) | `results/final_report/final_research_report.md` | Yes |
| 2 | Final research report (PDF) | `results/final_report/final_research_report.pdf` | Yes |
| 3 | Executive summary | `results/final_report/executive_summary.txt` | Yes |
| 4 | Comparison table (CSV) | `results/final_report/final_results_comparison.csv` | Yes |
| 5 | Comparison table (JSON) | `results/final_report/final_results_comparison.json` | Yes |
| 6 | Limitations and future work | `results/final_report/limitations_and_future_work.md` | Yes |
| 7 | Phase 7 verification report | `results/final_report/phase7_verification_report.txt` | Yes |
| 8 | Final project presentation | `final_project_presentation.pptx` | Yes |
| 9 | Viva questions and answers | `viva_questions_and_answers.md` | Yes |
| 10 | Demonstration guide | `project_demonstration_guide.md` | Yes |
| 11 | Final verification report | `results/final_report/final_verification_report.txt` | Yes |
| 12 | Project README | `README.md` | Yes |

---

## 2. Verification gates

Run these before submission:

```bash
cd D:/ml_project

# Phase 7 artifacts: 81 + 81 configurations, recomputed metrics, documents, immutability
python scripts/verify_phase7.py

# Final package: report, PDF, PPTX, figures, README, viva files, 162 configurations
python scripts/verify_final_project.py
```

Both scripts must print a PASS status. They do not train models or modify Phase 1–6 results.

- [ ] `phase7_verification_report.txt` ends with `STATUS: PASS`.
- [ ] `final_verification_report.txt` ends with `STATUS: PASS`.
- [ ] Phase 5 has 81 completed rows and Phase 6 has 81 completed rows.
- [ ] 162 experiment IDs are accounted for (81 per phase, 81 shared condition IDs).
- [ ] No Phase 1–6 artifact was added, removed, or modified.

---

## 3. Report quality gates

- [ ] All 19 report sections are present and in order.
- [ ] The report uses only saved results; no estimated or fabricated metric appears.
- [ ] Figures 1–11 are referenced and present in `results/final_report/figures/`.
- [ ] Tables 1–23 are present and numbered in reading order.
- [ ] The three MNIST LDA + Logistic Regression numerical failures are disclosed.
- [ ] The 16 Phase 5 and 17 Phase 6 convergence warnings are disclosed.
- [ ] The 3-iteration cap, 68 cap hits, and 13 confidence stops are disclosed.
- [ ] The MNIST SVM computational cost (91.3% of self-training time) is disclosed.
- [ ] The report states that Phase 5 did not record prediction time.
- [ ] The report states that no p-values or confidence intervals are reported.
- [ ] All references are real publications or official dataset sources.

---

## 4. Presentation gates

- [ ] `final_project_presentation.pptx` opens in PowerPoint or LibreOffice.
- [ ] Exactly 15 slides are present.
- [ ] Slides cover: title, problem, objectives, datasets, methodology, preprocessing, PCA, LDA, supervised, semi-supervised, results, comparative analysis, limitations, conclusion, future scope.
- [ ] At least four existing figures from `results/final_report/figures/` appear in the deck.
- [ ] Every number in the deck matches the saved results.
- [ ] No slide has overflowing or overlapping text (the deck passed the geometry audit).

---

## 5. Viva readiness

- [ ] `viva_questions_and_answers.md` contains at least 30 questions (50 are provided).
- [ ] You can explain, without notes: the experimental grid, the 0.90 threshold, the 3-iteration cap, and the calibrated SVM.
- [ ] You can explain why the overall mean is negative and why it reverses after excluding three fits.
- [ ] You can name the best configuration for each dataset.
- [ ] You can state the runtime ratio (7.59×) and the MNIST SVM share (91.3%).
- [ ] You can explain why no significance test is reported.
- [ ] You can run the demonstration guide end to end in under 12 minutes.

---

## 6. Integrity and preservation

- [ ] `results/classifier_evaluation/` is untouched.
- [ ] `results/self_training/` is untouched.
- [ ] `results/dimensionality_reduction/` is untouched.
- [ ] `data/raw/` and `data/processed/` were not re-downloaded or rewritten.
- [ ] `config/config.yaml` and the Phase 6 shard configs are unchanged.
- [ ] No Phase 5 or Phase 6 experiment was rerun during packaging.
- [ ] The Phase 1–6 snapshot still lists 692 files and 274.6 MiB.
- [ ] Phase 7 and Phase 8 outputs live only under `results/final_report/` and the project root documents.

---

## 7. File-size and portability checks

| Item | Expected |
|---|---|
| PDF | ~2.6 MiB, 17 pages, 11 embedded images |
| PPTX | 15 slides, opens without repair |
| Report Markdown | ~60 KB |
| Comparison CSV | 81 rows × 40 columns |
| Figures | 11 PNGs, 300 DPI |
| Tables | 7 CSVs |

- [ ] The submission folder contains no absolute paths that break on another machine.
- [ ] Large raw datasets are excluded or submitted separately according to course rules.
- [ ] `requirements.txt` lists the packages needed to inspect and verify the results.

---

## 8. Final sign-off

| Check | Owner | Date | Result |
|---|---|---|---|
| Phase 7 verification PASS |  |  |  |
| Final project verification PASS |  |  |  |
| Report read end to end |  |  |  |
| Presentation rehearsed |  |  |  |
| Viva questions rehearsed |  |  |  |
| Submission archive created |  |  |  |

**Submission statement:** All reported metrics are derived from the saved Phase 5 and Phase 6 result files. No experiment was rerun, no metric was fabricated, and no Phase 1–6 artifact was modified during Phase 7–8 packaging.
