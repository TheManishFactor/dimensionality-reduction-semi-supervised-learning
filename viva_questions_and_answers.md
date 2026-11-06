# Viva Questions and Answers

**Project:** Impact of Dimensionality Reduction on the Performance of Subsequent Supervised Classifiers in a Semi-Supervised Setting

All numbers below come from the saved Phase 5 and Phase 6 results and the Phase 7 analysis. They refer to the completed **3-iteration** self-training run with confidence threshold **0.90** and seed **42**. No result is estimated.

---

## A. Project framing

**1. What is the research question?**
Does applying dimensionality reduction (PCA or LDA) before classification change how much semi-supervised self-training improves accuracy and macro-F1, across datasets, classifiers, and labelled-data proportions?

**2. Why is semi-supervised learning relevant here?**
Labels are expensive, but unlabelled data are abundant. Self-training can use the unlabelled pool, but its benefit may depend on the feature representation that precedes it.

**3. What is the experimental design?**
A fully crossed design: 3 datasets × 3 labelled proportions × 3 representations × 3 classifiers = **81 conditions** per training regime. Phase 5 runs the supervised versions and Phase 6 runs the matched self-training versions, for **162 experiment runs** in total.

**4. What is the supervised baseline?**
The same 81 conditions trained only on the labelled rows, with the same split, same features, and same classifier settings. It isolates the effect of adding pseudo-labels.

**5. Why not treat the 81 conditions as 81 independent samples?**
They share rows, test samples, and transformations. They are repeated configurations on one fixed split, not independent replicates, so no significance tests or confidence intervals are reported.

## B. Data and preprocessing

**6. Which datasets are used and why?**
- **Breast Cancer Wisconsin (Diagnostic):** small, low-dimensional, binary — a sanity check.
- **MNIST:** large, high-dimensional, 10 classes — the expensive regime.
- **UCI HAR:** medium-sized, 6 classes, subject-disjoint split — tests generalisation to unseen people.

**7. What are the dataset sizes?**

| Dataset | Rows | Features | Classes | Train / Test |
|---|---:|---:|---:|---:|
| Breast Cancer | 569 | 30 | 2 | 455 / 114 |
| MNIST | 42,000 | 784 | 10 | 33,600 / 8,400 |
| HAR | 10,299 | 561 | 6 | 7,352 / 2,947 |

**8. How were the train/test splits created?**
Stratified 80/20 holdouts for Breast Cancer and MNIST. HAR uses its official subject-disjoint split (21 training subjects, 9 test subjects).

**9. How was scaling handled?**
`StandardScaler` was fit on the training partition only, then applied to the training, unlabelled, and test rows. Test statistics never influenced the scaler.

**10. How are labelled and unlabelled rows defined?**
For each dataset, 10%, 25%, or 50% of the non-test rows are labelled. The remaining non-test rows form the unlabelled pool. The unlabelled ground truth is hidden from every model and from threshold selection.

**11. How is leakage prevented?**
Test rows are never used for fitting, dimensionality-reduction fitting, pseudo-labelling, or threshold choice. SHA-256 fingerprints of fit features, fit labels, and unlabelled features are stored and verified, and every metric is recomputed from saved test predictions.

## C. PCA and LDA

**12. What does PCA do in this project?**
It projects the scaled features onto orthogonal components that retain **95%** of the training variance. It is unsupervised, so labels never enter the projection.

**13. Where is PCA fitted?**
Once per dataset, on the complete scaled training pool (455, 33,600, and 7,352 rows). The same fitted projection is applied unchanged to labelled, unlabelled, and test rows.

**14. What are the resulting PCA dimensions?**
- Breast Cancer: **30 → 10** components, 95.2069% variance.
- MNIST: **784 → 318** components, 95.0256% variance.
- HAR: **561 → 102** components, 95.0756% variance.

**15. What does LDA do?**
LDA is a supervised projection that maximises between-class separation relative to within-class spread, using the known labels.

**16. Where is LDA fitted and how many components are kept?**
It is fitted separately for each labelled proportion, using only that condition's labelled rows. Components are capped at `min(n_features, n_classes − 1)`, giving **1** for Breast Cancer, **9** for MNIST, and **5** for HAR.

**17. Why is LDA riskier than PCA here?**
LDA uses labels and can be unstable when the labelled subset is small. On MNIST it produced extreme out-of-sample values, and three LDA + Logistic Regression fits became numerically degenerate.

## D. Classifiers

**18. Which classifiers are compared and why?**
- **Logistic Regression** — linear probabilistic model.
- **Linear SVM** (`LinearSVC`) — large-margin linear model with a different loss.
- **KNN (k = 5)** — non-parametric, local decision rule.

They represent three different inductive biases, so the interaction with pseudo-labels is not tied to one model family.

**19. What hyperparameters were used?**
- Logistic Regression: lbfgs, `C = 1`, `max_iter = 2000`.
- Linear SVM: squared hinge, dual formulation, `C = 1`, `tol = 1e-3`, `max_iter = 2000`.
- KNN: `k = 5`, distance weights, Euclidean metric, brute-force search.

No hyperparameter tuning was performed.

**20. Why is KNN training time not directly comparable?**
KNN fitting only stores the training data; most computation happens at prediction time. Its fit time is therefore not a like-for-like measure of model training.

**21. Which metrics are used?**
Accuracy and macro-F1. Macro-F1 gives each class equal weight, which matters for the class imbalance in Breast Cancer and HAR.

## E. Self-training

**22. Describe the self-training loop.**
Fit on labelled rows → predict the unlabelled pool → accept predictions at or above the confidence threshold → add them to the training set → refit → repeat. It stops after 3 iterations or when no new sample is accepted.

**23. What confidence threshold and iteration limit were used?**
Threshold **0.90**, maximum **3 iterations**, minimum **1 new pseudo-label** per round, seed **42**.

**24. How is confidence obtained for each classifier?**
- Logistic Regression and KNN: maximum `predict_proba`.
- Linear SVM: `LinearSVC` has no probabilities, so a Platt sigmoid calibration is fitted on a class-stratified **20% holdout of the labelled rows only**, using `CalibratedClassifierCV`, `FrozenEstimator`, and `StratifiedKFold`. Final test predictions come from the plain `LinearSVC`.

**25. Why does the SVM need calibration?**
The 0.90 threshold is meaningless on raw decision scores, which are not probabilities. Calibration maps scores to probabilities so the same acceptance rule applies to all classifiers.

**26. Why is calibration done only on labelled rows?**
Using test rows or unlabelled ground truth would leak information and inflate acceptance. The calibration holdout is drawn only from labelled training rows.

**27. Was the threshold tuned on test performance?**
No. It was fixed at 0.90 before the experiments and used for all 81 conditions.

**28. How often did self-training stop for each reason?**
**68** runs reached the 3-iteration cap and **13** stopped because no remaining sample met the 0.90 threshold. All 81 runs completed.

**29. What was the pseudo-label acceptance?**
521,861 of 801,216 pooled unlabelled rows were accepted (**65.13%**). The mean per-condition acceptance was **73.39%**: Breast Cancer 75.07%, MNIST 61.29%, HAR 83.82%. Mean iterations completed: 2.69.

## F. Results and interpretation

**30. What was the overall effect of self-training?**
Mean accuracy fell from **0.90894** to **0.88836** (−0.02058); mean macro-F1 fell from **0.90626** to **0.88261** (−0.02365). On macro-F1, 38 conditions improved, 27 regressed, and 16 were unchanged.

**31. Why is the overall mean negative?**
Three MNIST LDA + Logistic Regression fits became numerically degenerate and produced near-uniform predictions. Excluding them, the remaining **78** conditions improve slightly: macro-F1 **0.90843 → 0.91222 (+0.00379)** and accuracy +0.00376.

**32. What was the largest genuine improvement?**
MNIST, 10% labelled, LDA + Linear SVM: macro-F1 **0.4923 → 0.7593 (+0.26698)**, accuracy +0.24524.

**33. Did self-training help most when labels were scarcest?**
No. Mean macro-F1 change was negative at every proportion: −0.0183 at 10%, −0.0199 at 25%, and −0.0327 at 50%. Self-training did not close the low-label gap overall.

**34. How did the representation affect the change?**
Mean macro-F1 change: Baseline **+0.0007**, PCA **−0.0003**, LDA **−0.0714**. The LDA figure is dominated by the three degenerate MNIST fits; Baseline and PCA are effectively neutral.

**35. How did the classifier affect the change?**
Linear SVM was the only classifier with a positive mean change (**+0.0125** macro-F1). Logistic Regression was −0.0814 (again dominated by the degenerate fits) and KNN was −0.0020.

**36. What are the best configurations per dataset?**

| Dataset | Best supervised (macro-F1) | Best self-training (macro-F1) |
|---|---|---|
| Breast Cancer | Baseline + LR, 25% — 0.9905 | Baseline + SVM, 10% — 0.9812 |
| MNIST | PCA + KNN, 50% — 0.9337 | PCA + KNN, 50% — 0.9331 |
| HAR | LDA + KNN, 25% — 0.9618 | LDA + KNN, 25% — 0.9610 |

**37. How large was the runtime cost?**
Supervised training totalled **767.3 s**; self-training totalled **5,826.3 s**, a ratio of **7.59×** (median condition ratio 10.12×). MNIST linear-SVM fits alone used **5,318.8 s**, or **91.3%** of self-training training time.

**38. Why is there no supervised prediction-time comparison?**
Phase 5 did not record prediction time. The gap is reported honestly rather than reconstructed; only Phase 6 prediction time (30.04 s total) is available.

## G. Warnings, failures, and statistics

**39. What is the difference between a warning, a limitation, and a failure?**
- A **warning** is emitted by an optimiser but the run still produces valid predictions.
- A **limitation** is a design constraint that qualifies the conclusions.
- A **failure** would be a run without valid results. There were **no failed runs** in Phase 5 or Phase 6.

**40. Which warnings were emitted?**
Phase 5: **16** `ConvergenceWarning`s (8 MNIST, 8 HAR). Phase 6: **17** LinearSVC convergence warnings and **3** `NumericalStabilityWarning`s. All are recorded per experiment and disclosed in the report.

**41. Why are the three degenerate fits kept in the results?**
Removing them silently would bias the comparison. They are flagged, reported, and included in the primary statistics; a sensitivity analysis excluding them is also reported.

**42. What statistical test was used?**
None. With one fixed split and seed, the conditions are not independent replicates. The analysis reports descriptive paired differences, medians, quartiles, and Cohen's d_z (mean paired difference divided by its standard deviation) without p-values or confidence intervals.

**43. What does the overall Cohen's d_z mean?**
It is a descriptive standardised effect size: −0.162 across all 81 macro-F1 pairs, and +0.119 for the 78 non-degenerate pairs. It is not a hypothesis test.

**44. What is the biggest threat to validity?**
The 3-iteration cap truncates the self-training search in 68 of 81 runs, and the single fixed split means split-to-split variability is unknown.

**45. Can the results be generalised to other seeds or splits?**
Only with caution. The results are conditional on the recorded split and seed. Replication across seeds and repeated splits is the main future-work item.

## H. Engineering and reproducibility

**46. How can the results be reproduced without retraining?**
All Phase 5/6 tables, predictions, pseudo-label audits, iteration logs, figures, and reports are saved. The Phase 7 scripts rebuild the comparison, report, and figures from those files only; the final verifier rechecks them.

**47. What verification was performed?**
Phase 4: **78/78** checks. Phase 5: **66/66**. Phase 6: **86/86**. Phase 7: **1,769/1,769**, including matched configurations, recomputed metrics, figure metadata, document regeneration, and Phase 1–6 immutability.

**48. How is Phase 1–6 integrity protected?**
A SHA-256 snapshot of 692 files (274.6 MiB) is compared before and after Phase 7. No Phase 1–6 artifact may be added, removed, or modified.

**49. What would you do differently with more compute?**
Run multiple seeds and repeated splits with a mixed-effects model, repeat self-training with a higher iteration cap, add shrinkage LDA or reject degenerate projections, log supervised prediction time, and compare co-training or label propagation under the same splits.

**50. What is the one-sentence takeaway?**
With a fixed split and a 3-round budget, PCA and the baseline representation are roughly neutral for self-training, LDA is the risky combination, and the apparent overall decline is driven by three numerical failures rather than by self-training itself.
