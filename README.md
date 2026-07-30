# HelloML

**Handwritten digit OCR on the DIDA dataset — multi-model comparison**

Unified notebook with four classifiers, 5-fold CV, enhanced preprocessing, and **train/load modes** so you can skip retraining.

| Item | Detail |
|------|--------|
| **Task** | Multi-class digit classification (0–9) |
| **Dataset** | DIDA (1,000 images × 10 digits = **10,000**) |
| **Split** | 80% train / 20% test (stratified) |
| **Models** | GaussianNB · Linear Regression (OvA) · Logistic Regression · **MLP** |
| **Metrics** | Accuracy, Precision, Recall, F1 (macro), Confusion Matrix |
| **CV** | 5-fold + `GridSearchCV` |
| **Main notebook** | [`OCR_HelloML_Complete.ipynb`](OCR_HelloML_Complete.ipynb) |
| **Helpers** | [`helloml_pipeline.py`](helloml_pipeline.py) |

---

## Repository structure

```
HelloML/
├── OCR_HelloML_Complete.ipynb   # ★ Main notebook (train / load)
├── helloml_pipeline.py          # Preprocessing + save/load helpers
├── artifacts/                   # Created after first train run (*.joblib)
├── OCR_Enhanced_Final.ipynb     # Older MLP-only version
├── DIDA/                        # Raw images (folders 0–9)
└── README.md
```

---

## Train once, load later

At the top of the notebook:

```python
RUN_MODE = "train"          # first run — trains + saves artifacts
# RUN_MODE = "load"         # later runs — restores variables, skips GridSearchCV
ARTIFACTS_DIR = Path.cwd() / "artifacts"
FILES_PER_FOLDER = 1000
```

| Mode | Behaviour |
|------|-----------|
| `"train"` | Load DIDA → preprocess → GridSearchCV → evaluate → **write** `artifacts/*.joblib` |
| `"load"` | **Read** artifacts → evaluate / plot only (no training) |

Saved objects (via `joblib`):

| File | Contents |
|------|----------|
| `X_train.joblib`, `X_test.joblib`, `y_train.joblib`, `y_test.joblib` | Features & labels |
| `X_proc.joblib`, `y.joblib` | Preprocessed images & full labels |
| `best_estimators.joblib` | All four fitted models |
| `df_cv_best.joblib`, `df_cv_all.joblib`, `df_test.joblib` | Result tables |

You can also call manually:

```python
save_artifacts(ARTIFACTS_DIR, globals())
load_artifacts(ARTIFACTS_DIR, globals())
```

---

## Pipeline

```text
Load DIDA  →  Preprocess (denoise→KMeans→crop→scale→center 28×28)
           →  Normalize + flatten (784) + stratified split
           →  GridSearchCV (all 4 models, 5-fold)
           →  Confusion matrices + held-out test scores + charts
           →  Save artifacts (train mode)
```

---

## Preprocessing (detail)

| Step | Operation | Why |
|------|-----------|-----|
| 1 | Median blur 3×3 | Remove stain / salt-pepper noise |
| 2 | K-Means k=2 | Adaptive ink vs background |
| 3 | Bounding-box crop | Drop empty margins |
| 4 | Scale max side → 20 px | Size normalization |
| 5 | Center on 28×28 | Fixed spatial layout |
| 6 | `/255` + flatten | Features in [0,1], shape `(N, 784)` |

---

## Models

| Model | Tuned params | Typical CV acc |
|-------|--------------|----------------|
| **MLP** | `hidden_layer_sizes` | ~0.80–0.85 |
| **LogisticReg** | `C ∈ {1.0, 10.0}` | ~0.74–0.78 |
| **LinearReg_OvA** | — | ~0.64–0.68 |
| **NaiveBayes** | — | ~0.50–0.55 |

---

## Requirements

```bash
pip install numpy pandas scikit-learn opencv-python matplotlib seaborn joblib
```

## How to run

1. Clone the repo and place `DIDA/` (folders `0`…`9`) at the project root.  
2. Open **`OCR_HelloML_Complete.ipynb`**.  
3. Keep `RUN_MODE = "train"` for the first run; switch to `"load"` afterwards.  

> Quick dry-run: set `FILES_PER_FOLDER = 200`.

---

## Author

**Ahmed Magdy** ([@Sober-Migo](https://github.com/Sober-Migo))

## License

Educational use. Feel free to adapt for learning.
