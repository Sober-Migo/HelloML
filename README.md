# HelloML

**Handwritten digit OCR on the DIDA dataset — multi-model comparison**

Unified, cleaned notebook that covers the full course assignment: four classifiers, 5-fold CV, hyperparameter search, confusion matrices, held-out test evaluation, and an **enhanced preprocessing pipeline**.

| Item | Detail |
|------|--------|
| **Task** | Multi-class digit classification (0–9) |
| **Dataset** | DIDA (1,000 images × 10 digits = **10,000**) |
| **Split** | 80% train / 20% test (stratified) |
| **Models** | GaussianNB · Linear Regression (OvA) · Logistic Regression · **MLP** |
| **Metrics** | Accuracy, Precision, Recall, F1 (macro), Confusion Matrix |
| **CV** | 5-fold + `GridSearchCV` |
| **Main notebook** | [`OCR_HelloML_Complete.ipynb`](OCR_HelloML_Complete.ipynb) |

---

## Repository structure

```
HelloML/
├── OCR_HelloML_Complete.ipynb   # ★ Recommended — full multi-model pipeline
├── OCR_Enhanced_Final.ipynb     # Earlier MLP-only enhanced version
├── DIDA/                        # Raw images (folders 0–9)
└── README.md
```

---

## Pipeline

```text
Load DIDA  →  Preprocess (denoise→KMeans→crop→scale→center 28×28)
           →  Normalize + flatten (784) + stratified split
           →  GridSearchCV (all 4 models, 5-fold)
           →  Confusion matrices + held-out test scores + charts
```

---

## Preprocessing (detail)

Raw DIDA digits vary in size, position, thickness, and often have stains.

| Step | Operation | Why |
|------|-----------|-----|
| 1 | Median blur 3×3 | Remove stain / salt-pepper noise |
| 2 | K-Means k=2 | Adaptive ink vs background (better than fixed Otsu) |
| 3 | Bounding-box crop | Drop empty margins |
| 4 | Scale max side → 20 px | Size normalization with margin |
| 5 | Center on 28×28 | Fixed spatial layout |
| 6 | `/255` + flatten | Features in [0,1], shape `(N, 784)` |

```python
# ink = darker of the two K-Means centers
pixels = denoised.reshape(-1, 1).astype(float)
kmeans = KMeans(n_clusters=2, n_init=10, max_iter=300, random_state=42)
labels = kmeans.fit_predict(pixels)
ink_label = np.argmin(kmeans.cluster_centers_)
```

The notebook also plots **raw vs preprocessed** samples (one per digit).

---

## Models & hyperparameter search

| Model | What is tuned | Role |
|-------|---------------|------|
| **NaiveBayes** | — | Fast probabilistic baseline |
| **LinearReg_OvA** | — | OLS + One-vs-All (MSE loss) |
| **LogisticReg** | `C ∈ {1.0, 10.0}` | Strong linear baseline |
| **MLP** | `hidden_layer_sizes`: `(700,200)`, `(500,100,10)`, `(300,150,100,50,10)` | Non-linear capacity |

Typical ranking on this pipeline:

| Rank | Model | Typical CV accuracy |
|------|-------|---------------------|
| 1 | MLP | ~0.80–0.85 |
| 2 | Logistic Regression | ~0.74–0.78 |
| 3 | Linear Regression OvA | ~0.64–0.68 |
| 4 | GaussianNB | ~0.50–0.55 |

MLP wins because digit classes are **not linearly separable** in raw pixel space; depth/width below a capacity threshold collapses toward chance (~10%).

---

## What the notebook produces

1. CV table (all param combos + best per model)  
2. 2×2 grid of **CV confusion matrices**  
3. Full **classification reports** on the held-out test set  
4. Side-by-side bar charts (CV accuracy vs test accuracy)  
5. Accuracy vs training-time scatter  

---

## Requirements

```bash
pip install numpy pandas scikit-learn opencv-python matplotlib seaborn
```

Python ≥ 3.10 recommended.

## How to run

1. Clone the repo and place the `DIDA/` folder (subfolders `0`…`9`) at the project root.  
2. Open **`OCR_HelloML_Complete.ipynb`**.  
3. Run all cells top to bottom.  

> **Faster dry-run:** set `files_per_folder=200` in `load_dataset(...)`.  
> Full 10k + MLP grid search can take several minutes on CPU (`n_jobs=-1`).

---

## Key functions

| Function | Purpose |
|----------|---------|
| `load_dataset(root, files_per_folder=1000)` | Load grayscale images + labels |
| `binarize_center_resize(imgs)` | Denoise → K-Means → crop → scale → center |
| `normalize_flatten_split(imgs, y)` | [0,1] normalize, flatten, stratified split |
| `get_experiment_setup()` | Model dict for `GridSearchCV` |

---

## Author

**Ahmed Magdy** ([@Sober-Migo](https://github.com/Sober-Migo))  
Computer Science graduate — AI/ML & backend.

## License

Educational use. Feel free to adapt for learning.
