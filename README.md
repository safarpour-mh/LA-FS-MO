# Unsupervised, Label-Agnostic Feature Selection for Biomedical Data

MATLAB implementation of a multi-objective, fully unsupervised feature selection framework for high-dimensional biomedical datasets.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17839764.svg)](https://doi.org/10.5281/zenodo.17839764)

---

## 📌 Abstract

This work presents a novel, label-agnostic, multi-objective feature selection framework for high-dimensional biomedical data. The method jointly optimizes two intrinsic properties:
- **Distributional informativeness** via discretized Shannon entropy ($K=5$ bins)
- **Structural coherence** via inverted Davies–Bouldin Index (1D k-means, $k=2$)

A fuzzy crowding-distance mechanism promotes diversity by prioritizing features in sparsely populated regions of the bi-objective space. The approach operates with linear complexity $\mathcal{O}(dn + d \log d)$ and requires no class labels, distributional assumptions, or iterative optimization.

---

## 📂 Repository Structure

├── main_evaluation.m # Main script: Full vs. Selected features (single dataset)
├── run_comparison_eight_datasets.m # Batch evaluation across 8 benchmark datasets
├── run_baseline_comparison.m # Comparison with unsupervised baselines (Laplacian, MCFS, UDFS)
├── run_statistical_validation.m # Wilcoxon signed-rank test implementation
├── dataset/ # Original .mat files (for MATLAB compatibility)
│ ├── wdbcSamples.mat, wdbcLabel.mat, ...
├── datasets_CSV/ # ✅ Human-readable CSV versions of all datasets
│ ├── wdbc_features.csv, wdbc_labels.csv, ...
├── results/ # Output folder for tables and figures (auto-created)
└── README.md # This file



---

## 🗄️ Dataset Information

All benchmark datasets are publicly available from the **UCI Machine Learning Repository**:
- **Source:** https://archive.ics.uci.edu/
- **Citation:** Kelly, M., Longjohn, R., & Nottingham, K. (2025). The UCI Machine Learning Repository.

| Dataset | Instances | Features | Domain |
|---------|-----------|----------|--------|
| crx | 690 | 15 | Credit scoring |
| australian | 690 | 14 | Credit approval |
| heart | 270 | 13 | Cardiac diagnosis |
| ionosphere | 351 | 34 | Radar signal |
| wpbc | 198 | 34 | Breast cancer prognosis |
| wdbc | 569 | 30 | Breast cancer diagnosis |
| segment | 2310 | 19 | Image segmentation |
| zoo | 101 | 16 | Animal taxonomy |

### Data Formats
All datasets are provided in **two formats** for maximum accessibility:
- **MATLAB `.mat` files** (`[name]Samples.mat`, `[name]Label.mat`): For direct use with the provided MATLAB scripts.
- **Human-readable CSV files** (`[name]_features.csv`, `[name]_labels.csv`): Located in the `datasets_CSV/` folder, these files enable inspection and reuse in Python, R, or other environments without MATLAB.

**CSV Structure:**
- `[name]_features.csv`: Feature matrix with columns `Feature_1, Feature_2, ..., Feature_d`
- `[name]_labels.csv`: Single-column label vector (`Label`)
- Both files have matching row ordering (sample-wise alignment)

---

## 🔬 Methodology Summary

The proposed framework operates in three sequential, label-free steps:

1. **Feature Scoring**: Each feature $f_j$ is independently evaluated using two non-parametric objectives:
   - Discretized Shannon entropy ($K=5$ uniform bins) for distributional informativeness
   - Inverted Davies–Bouldin Index (1D k-means, $k=2$) for structural coherence

2. **Diversity-Aware Ranking**: A fuzzy crowding-distance score computes each feature's degree of "informative isolation" in the bi-objective space, promoting non-redundancy without pairwise comparisons.

3. **Subset Selection**: The top $r = \max(5, \lceil 0.2 \cdot d \rceil)$ features are selected, ensuring compactness while preserving predictive utility.

*No class labels are accessed during steps 1–3.*


---

## 💻 Code Information

| File | Purpose |
|------|---------|
| `main_evaluation.m` | Reproduce classification results for a single dataset (Accuracy, Precision, Recall, F1, ROC) |
| `run_comparison_eight_datasets.m` | Batch evaluation across all 8 datasets; generates combined ROC figure |
| `run_baseline_comparison.m` | Compare proposed method against Laplacian Score, MCFS, UDFS (F1-score) |
| `run_statistical_validation.m` | Wilcoxon signed-rank test with Bonferroni correction |

**Key Implementation Details:**
- All scripts are fully unsupervised: class labels are never accessed during feature scoring or ranking.
- Stratified 5-fold cross-validation is used for downstream evaluation.
- Random seed is fixed (`rng(42)`) for reproducibility.
- Selected feature indices per dataset are hardcoded based on the method's output.

---

## 🚀 Usage Instructions

### Prerequisites
1. MATLAB R2023a or later (Statistics and Machine Learning Toolbox recommended)
2. Clone or download this repository
3. Create a `dataset/` subfolder and place all `.mat` files inside

### Step-by-Step
```matlab
% 1. Navigate to the repository root in MATLAB
cd /path/to/repository

% 2. Run single-dataset evaluation (e.g., wdbc)
%    Edit main_evaluation.m to load your desired dataset
run main_evaluation.m

% 3. Run batch evaluation across all 8 datasets
run run_comparison_eight_datasets.m

% 4. Compare with unsupervised baselines (heart, wpbc, wdbc)
run run_baseline_comparison.m

% 5. Perform statistical validation
run run_statistical_validation.m
```

---

## Output: Results are saved to the results/ folder:
*_Results.xlsx: Classification metrics (Acc, Prec, Rec, F1)
ROC_*.pdf/png: ROC curves
comparison_results.tex: LaTeX table for baseline comparison
wilcoxon_table.tex: LaTeX table for statistical tests


---

## ⚙️ Requirements
MATLAB Version: R2023a or later
Toolboxes: Statistics and Machine Learning Toolbox (for fitcsvm, fitcknn, TreeBagger, cvpartition, perfcurve)
No third-party dependencies: All code uses native MATLAB functions

---

## 📜 License & Contributions
License: MIT License — see LICENSE file for details.
Contributions: Bug reports and suggestions are welcome via GitHub Issues.
Reproducibility: All random seeds are fixed; results should be identical across runs on the same MATLAB version.

