# Project Challenges and Mitigations

*Reliability Assessment of Saliency-Based XAI Methods for Cardiomegaly Detection in Chest X-rays Using a Composite Trustworthiness Index (CTI)*

This document records the major problems encountered during the project and how each was resolved. It maps issues to the project modules and is intended as a "Challenges & Mitigations" reference for the manuscript, thesis, or viva.

---

## 1. The single most critical problem — Patient data leakage (Module 1, Dataset)

The original train/validation/test split used a **random, image-level** `sklearn.train_test_split`, stratified only by class. Because NIH ChestX-ray14 contains multiple radiographs per patient, this placed the same patient in both training and test sets.

**Measured leakage (before fix):**
- 5,092 patients appeared in both train and test.
- **70.9% of test images (8,953 / 12,634)** belonged to a patient also seen in training.

On NIH ChestX-ray14 this is a well-known, automatic-rejection flaw for Q1/Q2 journals: the reported AUCs were optimistically biased and scientifically invalid.

**Fix — strict patient-wise split.** Patient IDs were partitioned 70/10/20 (stratified on each patient's Cardiomegaly status); all images of a patient were assigned to a single fold.

**Verification (after fix):**
- Patient overlap train↔test = 0, train↔val = 0, val↔test = 0.
- Test images whose patient is also in train = **0 / 12,903 = 0.0%**.
- Assertion `len(train_patients ∩ test_patients) == 0` — **PASSED**.
- 63,136 images across 25,492 unique patients; split 44,017 / 6,216 / 12,903 (Cardiomegaly 1,995 / 259 / 522).

Because the split changed, **all three models were retrained** and every downstream artifact (XAI maps, CTI, figures) regenerated.

---

## 2. Major problems by module

| # | Module | Problem | Impact | Resolution |
|---|--------|---------|--------|------------|
| 1 | 1 — Dataset | Patient data leakage (random image-level split; 70.9% test-image overlap) | Inflated/invalid AUC; certain journal rejection | Patient-wise split, 0% overlap (verified) |
| 2 | Infrastructure | Path/environment chaos: data scattered across `C:\cti_data`, `C:\cti_project`, and the real root; stale **leaky** duplicate split CSVs; conda not on PATH | Risk of reading wrong/leaky data; non-reproducible runs | Consolidated to one root `C:\Users\NMAMIT\cti_project`; removed stale CSVs; call env python by full path |
| 3 | 2 — Models | Shallow training (3-epoch stubs); EfficientNet-B4 AUC 0.68 (< 0.80 gate); models trained on the leaky split | Weak, invalid classifiers | Two-phase fine-tuning + early stopping; retrained — all AUC ≥ 0.83 |
| 4 | 4 / 5 | ~2-day runtime: model reloaded per image; stability/sanity regenerated all maps; Score-CAM looped over 1,024–2,048 channels | Pipeline practically un-runnable | Load-once XAI engine + pre-resized memmap cache (epoch 16 min → 1.7 min); batched top-64 Score-CAM |
| 5 | 4 — XAI | Score-CAM broken — all-zero / wrong-shape (7×7) maps; localization ≈ 0.19 | One of five XAI methods unusable | ReLU-before-combine, upsample to 224×224, batched — localization ≈ 0.5–0.66 |
| 6 | 4 / 5 | Incorrect methodology: "Layer-CAM" was Grad-CAM on a shallow layer; sanity check used Grad-CAM for every method | Mislabeled method; invalid sanity metric | True element-wise Layer-CAM; same-method, seeded random-model sanity check |
| 7 | 5 — CTI | Incomplete/throwaway CTI: partial CSVs, only 4 methods, `cross_arch = 0`, no checkpointing | Results incomplete and non-resumable | Full 5-metric pipeline, all 15 combinations, per-image checkpointing |
| 8 | 8 — Stats | No statistical rigor: missing Friedman/Wilcoxon; broken pathology correlation; 150-DPI figures | Below Q1/Q2 evidence bar | Friedman + Wilcoxon/Bonferroni + Cohen's d + bootstrap CIs; all figures at 300 DPI |
| 9 | 2 / 7 | Class imbalance → low precision/F1; accuracy misleading at ~96% imbalance | Headline F1 understates models | AUC-ROC as primary metric; threshold tuning (e.g. EfficientNet specificity 0.25 → 0.75); focal-loss ablation **completed** (Section 3.8) — confirms low F1 is a calibration artifact + prevalence-bound ceiling |
| 10 | 9 — App | Stale Streamlit app (hardcoded old CTI, no Score-CAM, wrong ranking); dual-server `ModuleNotFoundError: torch` (base env vs cti_project env) | Demo showed wrong numbers / crashed | Rewrote app to read live results; killed the base-env server; documented correct launch command |

---

## 3. Recurring root causes

Most issues traced back to two themes:

1. **Methodological shortcuts** that look fine at a glance but fail peer review — patient leakage, a mislabeled Layer-CAM, single-criterion (localization-only) evaluation, and reporting accuracy on a 96%-imbalanced test set.
2. **Naïve implementations that did not scale** — per-image model reloads and unbatched Score-CAM produced the "~2-day runtime" wall that originally stalled the project; solved with a load-once engine, a memmap image cache, and batched/limited Score-CAM.

---

## 4. Status summary

- **Resolved (10 of 10):** leakage, paths/env, training, runtime, Score-CAM, methodology (Layer-CAM + sanity), CTI completeness, statistical rigor, app, and low F1 (#9).
- **Low F1 (#9) — resolved as a reported finding (Section 3.8).** The focal-loss ablation (`focal_train.py` → `models_v2/`) was run: it nearly doubled F1@0.5 and restored a near-default optimal threshold (0.99 → 0.66–0.71), proving the low headline F1 was a *calibration artifact* of the dual imbalance correction; but best-achievable F1 rose only marginally (≈+0.01), proving the ceiling is *prevalence-bound* at ~4%. The original models are retained for CTI (marginally higher AUC); F1 is reported at the tuned operating point (~0.36–0.37). No retrain adoption was needed.

No Q1/Q2 blockers remain: the four classic rejection triggers — data leakage, missing statistical tests, incomplete XAI methods, and sub-threshold AUC — are all cleared.
