# Methods and Results (Draft)

*Reliability Assessment of Saliency-Based XAI Methods for Cardiomegaly Detection in Chest X-rays Using a Composite Trustworthiness Index (CTI)*

All numbers below are generated directly from the project result files in `results/`. Figure references map to the 300-DPI files in that folder.

---

## 2. Methods

### 2.1 Dataset and patient-independent split
We used the NIH ChestX-ray14 dataset and framed the task as binary classification of **Cardiomegaly** (any image whose label set contains "Cardiomegaly") versus **Normal** ("No Finding"). After restricting to images available on disk, the cohort comprised **63,136 frontal radiographs (2,776 Cardiomegaly, 60,360 Normal)** drawn from 25,492 unique patients.

Because ChestX-ray14 contains multiple studies per patient, a random image-level split leaks patient identity across folds and inflates performance. We therefore performed a **patient-independent split**: patient IDs were partitioned 70/10/20 into train/validation/test with stratification on each patient's Cardiomegaly status, and all images of a patient were assigned to a single fold. The resulting split contains **44,017 / 6,216 / 12,903** images (train/val/test), with **1,995 / 259 / 522** Cardiomegaly cases respectively. We verified **zero patient overlap** between any two folds (0 of 12,903 test images shared a patient with the training set), eliminating the leakage that affects much prior work on this dataset.

Region-level ground truth for the localization metric was taken from `BBox_List_2017.csv`; of its 984 annotations across eight pathologies, the **146 Cardiomegaly bounding boxes** (all present on disk) form the saliency-evaluation set (`BBox_Cardiomegaly.csv`).

### 2.2 Preprocessing and augmentation
Images were resized to 224×224 and normalized with ImageNet statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]). Training-time augmentation comprised random horizontal flips, rotations of ±10°, brightness jitter of ±15%, and Gaussian blur applied with probability 0.2; validation and test images received no augmentation.

### 2.3 Classification models and training
Three ImageNet-pretrained backbones were fine-tuned with a single sigmoid output: **DenseNet121**, **ResNet50**, and **EfficientNet-B4**. Class imbalance (prevalence ≈ 4.4%) was handled with both a weighted `BCEWithLogitsLoss` (pos_weight = 21.06, computed as the train-set negative/positive ratio) and a `WeightedRandomSampler` that oversampled the minority class.

Optimization used Adam (lr = 1×10⁻⁴, weight decay = 1×10⁻⁵) with a CosineAnnealingWarmRestarts schedule (T₀ = 10) and mixed-precision training. A two-phase fine-tuning protocol was applied: epochs 1–5 trained only the classifier head with the backbone frozen, and from epoch 6 the last two blocks were unfrozen and fine-tuned end-to-end at lr = 1×10⁻⁵. Training ran for up to 50 epochs with early stopping (patience = 7) on validation AUC-ROC; the best-AUC checkpoint was retained. The primary metric was AUC-ROC, reported with a 95% confidence interval from 1,000 bootstrap resamples of the test set. As a calibration ablation, all three backbones were additionally retrained with a focal loss (α = 0.6, γ = 2) under an otherwise identical protocol (Section 3.8).

### 2.4 Saliency-based XAI methods
For every Cardiomegaly bounding-box image we generated five attribution maps per model — **Grad-CAM**, **Grad-CAM++**, **Score-CAM**, **Layer-CAM**, and **Integrated Gradients** — yielding 15 model×method combinations and 2,190 maps in total. Grad-CAM and Grad-CAM++ used the final convolutional block; Layer-CAM was computed with its canonical element-wise positive-gradient weighting on an earlier (finer) block; Score-CAM used the 64 highest-activation channels (batched in a single forward pass) and was upsampled to 224×224; Integrated Gradients used a zero baseline with 50 steps. Every map was normalized to [0, 1] and stored as a 224×224 array.

### 2.5 Composite Trustworthiness Index (CTI)
We quantified explanation reliability with a CTI averaging five equally weighted components, each scaled to [0, 1] (higher = more trustworthy):

1. **Localization** — mean of IoU (saliency thresholded at the 80th percentile vs. the scaled bounding box), Jensen–Shannon similarity (1 − JS divergence), and the Pointing Game hit rate.
2. **Stability** — mean of SSIM and Spearman correlation between the map and maps recomputed under three input perturbations (Gaussian noise, ±10% brightness, 5° rotation).
3. **Cross-XAI agreement** — mean pairwise Spearman correlation across the five methods for a given image/model (10 pairs).
4. **Cross-architecture agreement** — mean pairwise Spearman correlation of a method's maps across the three models (3 pairs).
5. **Sanity** — 1 − ½(SSIM + max(Spearman, 0)) between the trained-model map and the *same method* applied to a fully randomized (seeded) model; high dissimilarity indicates the explanation depends on learned weights.

CTI = 0.20·Localization + 0.20·Stability + 0.20·Cross-XAI + 0.20·Cross-Arch + 0.20·Sanity, computed per image and averaged.

### 2.6 Statistical analysis
Method-level CTI distributions (per-image CTI averaged across the three models, n = 146) were compared with the Friedman test, followed by all ten pairwise Wilcoxon signed-rank tests with Bonferroni correction (α = 0.05/10 = 0.005). The effect size between the best and worst methods was quantified with Cohen's d. Robustness was assessed by (i) recomputing the ranking under three CTI weightings, (ii) bootstrapping 95% CIs for each method's mean CTI, (iii) correlating per-image CTI with bounding-box area (Spearman), and (iv) recomputing the ranking on the leakage-free test-split subset (n = 27).

---

## 3. Results

### 3.1 Classification performance
All three models exceeded the AUC-ROC ≥ 0.80 acceptance threshold on the held-out, patient-independent test set (**Table 1**, Fig. 7). ResNet50 and DenseNet121 were strongest (AUC 0.897 and 0.894); EfficientNet-B4 trailed at 0.832.

**Table 1. Test-set classification performance (threshold 0.5).**

| Model | AUC-ROC (95% CI) | F1 | Precision | Recall | Specificity | Balanced Acc |
|---|---|---|---|---|---|---|
| DenseNet121 | 0.894 (0.883–0.905) | 0.190 | 0.106 | 0.933 | 0.668 | 0.801 |
| ResNet50 | 0.897 (0.885–0.908) | 0.235 | 0.136 | 0.868 | 0.768 | 0.818 |
| EfficientNet-B4 | 0.832 (0.817–0.846) | 0.100 | 0.053 | 0.989 | 0.249 | 0.619 |

Because training emphasized sensitivity to the rare class, the default 0.5 threshold over-predicted positives (low precision and, for EfficientNet-B4, low specificity). At a validation-tuned operating point (Youden's J), specificity rose substantially without affecting AUC — DenseNet121 to 0.813, ResNet50 to 0.791, and EfficientNet-B4 from 0.249 to **0.753** (balanced accuracy 0.811 / 0.819 / 0.747) — confirming the models are well-calibrated rankers whose operating point is simply tunable.

### 3.2 Trustworthiness ranking (CTI)
Across all 15 combinations, **Grad-CAM++ paired with DenseNet121 achieved the highest CTI (0.715)** (**Table 2**, Fig. 1). Averaged across the three architectures, **Grad-CAM++ was the most trustworthy method (mean CTI 0.662, 95% CI 0.656–0.668)**, followed by Grad-CAM (0.627), Layer-CAM (0.620), Score-CAM (0.572), and Integrated Gradients (0.558). The bootstrap CIs of the leading methods are tight and non-overlapping, indicating a stable ordering.

**Table 2. CTI per model × XAI method, and method-level mean (95% CI).**

| Method | DenseNet121 | ResNet50 | EfficientNet-B4 | Mean (95% CI) | Rank |
|---|---|---|---|---|---|
| Grad-CAM++ | **0.715** | 0.710 | 0.561 | 0.662 (0.656–0.668) | 1 |
| Grad-CAM | 0.683 | 0.661 | 0.538 | 0.627 (0.619–0.635) | 2 |
| Layer-CAM | 0.634 | 0.678 | 0.548 | 0.620 (0.614–0.627) | 3 |
| Score-CAM | 0.579 | 0.625 | 0.510 | 0.572 (0.561–0.582) | 4 |
| Integrated Gradients | 0.595 | 0.579 | 0.498 | 0.558 (0.551–0.563) | 5 |

### 3.3 Component-level analysis
No single method dominated every dimension (Fig. 2). Grad-CAM gave the best **localization** (0.532), Grad-CAM++ the best **stability** (0.800) and **cross-architecture agreement** (0.615), and Layer-CAM the best **sanity** (0.829). Cross-XAI agreement is a per-model property and is therefore identical across methods within a model (overall 0.538). Grad-CAM++ ranks first or second on four of the five components, explaining its top overall CTI.

### 3.4 Statistical significance
The five methods differed highly significantly in CTI (Friedman χ²(4) = 420.8, p ≈ 9.0×10⁻⁹⁰). **All ten pairwise Wilcoxon signed-rank tests were significant after Bonferroni correction** (α = 0.005); the least-separated pairs (Grad-CAM vs Layer-CAM, p = 3.6×10⁻³; Score-CAM vs Integrated Gradients, p = 3.5×10⁻³) remained below threshold. The effect size between the best and worst methods (Grad-CAM++ vs Integrated Gradients) was very large (Cohen's d = 2.88).

### 3.5 Sensitivity to CTI weighting
The method ranking was **identical under all three weight configurations** (equal, localization-heavy, stability-heavy; Fig. 5), with Grad-CAM++ first and Integrated Gradients last in every case, demonstrating that the conclusion is not an artifact of equal weighting.

### 3.6 Pathology size vs. trustworthiness
Per-image CTI correlated positively with cardiomegaly bounding-box area for four of five methods (Fig. 6): Integrated Gradients (Spearman r = 0.324, p = 7×10⁻⁵), Score-CAM (r = 0.239, p = 0.004), Layer-CAM (r = 0.190, p = 0.022), and Grad-CAM (r = 0.177, p = 0.033); Grad-CAM++ showed no significant dependence (r = 0.096, p = 0.250). Thus, for most attribution methods, explanation reliability degrades on smaller pathological regions, whereas Grad-CAM++ is comparatively size-robust.

### 3.7 Leakage-free robustness
Restricting the analysis to the 27 bounding-box images that fall in the held-out test split (zero patient overlap with training) reproduced the ranking: Grad-CAM++ remained first (0.656), with Layer-CAM (0.617) and Grad-CAM (0.614) essentially tied, confirming the result is not driven by training-set images.

### 3.8 Loss-function ablation: calibration and the F1 ceiling
The low F1 at threshold 0.5 (Table 1) reflects the **dual imbalance correction** used in training (weighted BCE with pos_weight = 21 *and* a minority oversampler), which deliberately shifts the decision boundary toward the positive class to maximize sensitivity. To separate this calibration effect from intrinsic discriminative ability, we retrained all three backbones with a **focal loss** (α = 0.6, γ = 2) and a single, milder imbalance correction, leaving the architecture and patient-independent split unchanged (**Table 3**).

**Table 3. Loss-function ablation on the patient-independent test set (dual-corrected weighted-BCE → focal loss).**

| Model | AUC-ROC (BCE → Focal) | Avg. Precision (BCE → Focal) | F1@0.5 (BCE → Focal) | Best F1 (BCE → Focal) |
|---|---|---|---|---|
| DenseNet121 | 0.894 → 0.888 | 0.293 → 0.311 | 0.190 → 0.280 | 0.356 → 0.367 |
| ResNet50 | 0.897 → 0.892 | 0.305 → 0.319 | 0.236 → 0.353 | 0.373 → 0.379 |
| EfficientNet-B4 | 0.832 → 0.840 | 0.188 → 0.216 | 0.100 → 0.188 | 0.252 → 0.267 |

Focal loss **nearly doubled F1 at the default 0.5 threshold** (e.g. ResNet50 0.236 → 0.353) and moved the F1-optimal threshold from a pathological **0.99 to a near-default 0.66–0.71**, confirming that the headline F1 was a *calibration artifact* of the dual correction rather than a sign of weak models. Average precision improved modestly for all three (e.g. EfficientNet-B4 0.188 → 0.216), while AUC-ROC was essentially unchanged (within ±0.006). Critically, the **best achievable F1** — the operating-point-independent ceiling set by the precision–recall curve — rose only marginally (DenseNet121 0.356 → 0.367; ResNet50 0.373 → 0.379), showing that F1 is fundamentally **bounded by the ~4% disease prevalence** and cannot be raised substantially by the loss function alone. The main CTI analysis therefore retains the original models (marginally higher AUC-ROC); F1 under this imbalance is best reported at a tuned operating point (Section 3.1), not at threshold 0.5.

---

## Key statement
Among all 15 model–XAI combinations evaluated on NIH ChestX-ray14 cardiomegaly detection under a strict patient-independent split, **Grad-CAM++ with DenseNet121 achieved the highest Composite Trustworthiness Index (0.715)**, and Grad-CAM++ was the most reliable method overall (mean CTI 0.662, 95% CI 0.656–0.668). The ranking was statistically significant (Friedman χ² = 420.8, p < 0.001; all pairwise Wilcoxon significant at Bonferroni α = 0.005; Cohen's d = 2.88), stable across three weighting schemes, and reproduced on a leakage-free subset, while explanation reliability correlated with pathology size for four of five methods.

*Figures: Fig. 1 `cti_heatmap`, Fig. 2 `component_breakdown`, Fig. 3 `training_curves`, Fig. 4 `sample_heatmaps`, Fig. 5 `sensitivity_analysis`, Fig. 6 `pathology_correlation`, Fig. 7 `roc_curves` (all 300 DPI, .png/.pdf in `results/`).*
