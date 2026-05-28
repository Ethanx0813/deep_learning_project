# Deep Learning for Structural Stress Prediction using CNN

**Group 3 — Question 7**  
**Ankit Yadav** (2025MEM1047) &nbsp;|&nbsp; **Nitin Kumar** (2025MEM1051)

---

## Problem Statement

Given the **thickness profile** of a structural component (226 thickness values per sample), predict the **maximum stress** experienced by that component. This is a high-dimensional regression and classification problem drawn from computational structural mechanics, where running full FEM simulations for 5000+ configurations is expensive — a deep learning surrogate model can drastically reduce that cost.

---

## Dataset

| Property | Value |
|---|---|
| Total samples | 5,000 |
| Input features | 226 (thickness profile values per sample) |
| Target (regression) | Maximum von Mises stress (`max_stress`) |
| Target (classification) | 4 percentile-based stress classes |
| Source | `DL_data.zip` → `prob1_data/output.xlsx` + `stress/*.txt` files |

**Stress class labels** are defined using training-set percentiles:

| Class | Range |
|---|---|
| 0 | 0 – 10th percentile |
| 1 | 11th – 25th percentile |
| 2 | 26th – 50th percentile |
| 3 | 51st – 100th percentile |

---

## Methodology

Two parallel approaches are explored and compared on the same dataset and architecture.

### Approach 1 — Regression CNN → Binning
Train a CNN to **predict the exact stress value**, then bin the prediction into the four percentile classes for comparison.

### Approach 2 — Direct Classification CNN
Train a CNN to **directly predict the stress class** (4-class softmax output).

---

## Architecture

Both models share a common **CNN backbone** that extracts spatial features from the 1-D thickness profile.

```
Input: (N, 226)
  │
  ▼
unsqueeze → (N, 1, 226)          ← treat as 1-D signal
  │
  ├── Conv1d(1 → 64, k=5)
  ├── BatchNorm1d + ReLU
  ├── MaxPool1d(2)
  │
  ├── Conv1d(64 → 128, k=3)
  ├── BatchNorm1d + ReLU
  ├── MaxPool1d(2)
  │
  ├── Conv1d(128 → 256, k=3)
  ├── BatchNorm1d + ReLU
  ├── AdaptiveAvgPool1d(1)       ← collapses spatial dim to 1
  │
  ├── Flatten → Linear(256→256) → ReLU → Dropout(0.3)
  └── Linear(256→128) → ReLU → Dropout(0.3)
                                          (128-dim feature vector)
```

**Regression head:** `Linear(128 → 1)` — predicts a single stress value.  
**Classification head:** `Linear(128 → 4)` — predicts logits for 4 classes.

---

## Data Preprocessing

### Features (X)
- Applied `StandardScaler` (zero mean, unit variance) fitted on training set only.
- Prevents data leakage by transforming val/test using training statistics.

### Regression Target (y)
- Applied **log transform** (`ln(stress)`) to compress the heavy-tailed stress distribution.
- Then **z-score normalized** using training set mean (`μ`) and std (`σ`).
- Predictions are back-transformed: `y_orig = exp(y_norm × σ + μ)`.

### Classification Target
- Converted stress values to 4 percentile-based class labels using `np.percentile`.
- Computed **class weights** (`compute_class_weight`) to handle class imbalance in the CrossEntropyLoss.

### Train / Validation / Test Split

| Split | Samples | Ratio |
|---|---|---|
| Train | 3,500 | 70% |
| Validation | 750 | 15% |
| Test | 750 | 15% |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Random seed | 42 |
| Epochs | 100 |
| Batch size | 64 |
| Optimizer | Adam (lr = 1e-3) |
| Regression loss | MSELoss |
| Classification loss | CrossEntropyLoss (with class weights) |
| Device | GPU if available, else CPU |

---

## Results

### Regression (Approach 1)

| Metric | Value |
|---|---|
| R² Score | 0.36 |
| RMSE | 2,956,389 |
| MAE | 2,106,605 |

### Classification (Approach 2)

| Metric | Value |
|---|---|
| Accuracy | 49.6% |

### Head-to-Head Comparison

| Approach | Bin Accuracy |
|---|---|
| Regression → Bin (Approach 1) | **56.5%** |
| Direct Classification (Approach 2) | 49.6% |

**Approach 1 outperforms direct classification by ~7 percentage points**, showing that learning the continuous stress value first provides a stronger signal than learning discrete boundaries directly.

### Data Efficiency Curve

Training set size was varied from 500 to 3500 samples. Approach 1 consistently outperforms Approach 2 across all training sizes, with the gap widening as data increases — indicating that regression-then-bin generalises better with more data.

---

## Repository Structure

```
deep_learning_project/
├── deep_learning_project.ipynb   # Full implementation with outputs
├── DLTORPR.pdf                   # Project report
├── Project_DL.pptx               # Presentation slides
└── README.md
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ethanx0813/deep_learning_project.git
   cd deep_learning_project
   ```

2. **Install dependencies**
   ```bash
   pip install torch numpy pandas matplotlib scikit-learn openpyxl
   ```

3. **Place the dataset**  
   Put `DL_data.zip` in the project root directory (not included in repo due to size).

4. **Run the notebook**  
   Open `deep_learning_project.ipynb` in Jupyter and run all cells.

---

## Tech Stack

| Library | Purpose |
|---|---|
| PyTorch 2.2 | Model definition, training, inference |
| scikit-learn | Preprocessing, metrics, class weights |
| NumPy | Numerical operations |
| pandas | Data loading from Excel |
| Matplotlib | Visualisation (loss curves, parity plot, confusion matrix) |

---

## Key Takeaways

- **Log-transforming skewed targets** before regression significantly stabilises training and improves convergence.
- **Shared backbone design** ensures both tasks benefit from the same spatial feature extractor, enabling a fair comparison.
- **Class-weighted loss** is critical for imbalanced classification — without it, the model collapses to predicting the majority class.
- The **regression surrogate outperforms direct classification**, suggesting continuous-valued supervision carries richer gradient information even when the final goal is categorical.
