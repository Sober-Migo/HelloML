# HelloML

**First Machine Learning project — Optical Character Recognition (OCR) for handwritten digits**

This repository contains an end-to-end pipeline that classifies digits **0–9** from the **DIDA** dataset using a Multi-Layer Perceptron (MLP). Images are preprocessed (denoise → binarize with K-Means → center & resize) and fed into scikit-learn’s `MLPClassifier`, with hyperparameter search via `GridSearchCV`.

## Highlights

| Item | Detail |
|------|--------|
| **Task** | Multi-class digit classification (0–9) |
| **Dataset** | DIDA (folders `0` … `9`) |
| **Samples used** | 1,000 images per digit → **10,000** total |
| **Split** | 80% train / 20% test (stratified) |
| **Model** | `sklearn.neural_network.MLPClassifier` |
| **Best architecture** | Hidden layers `(700, 200)` |
| **CV accuracy** | ~81.3% (5-fold) |
| **Test accuracy** | **~81.7%** |

## Repository structure

```
HelloML/
├── OCR_Enhanced_Final.ipynb   # Main notebook (full pipeline + plots)
├── DIDA/                      # Raw digit images (folders 0–9)
├── Processed_Data/            # Optional preprocessed images
└── README.md
```

## Pipeline overview

```text
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌────────────┐
│ Load DIDA   │ -> │ Preprocess           │ -> │ Normalize &     │ -> │ GridSearchCV │ -> │ Evaluate   │
│ (10k imgs)  │    │ denoise→binarize→    │    │ flatten + split │    │ (MLP configs)│    │ test set   │
│             │    │ center→28×28         │    │ 784 features    │    │ 5-fold CV    │    │            │
└─────────────┘    └──────────────────────┘    └─────────────────┘    └──────────────┘    └────────────┘
```

1. **Load data** — Read grayscale images from `DIDA/0` … `DIDA/9` (1,000 per class, reproducible sampling with seed 42).
2. **Preprocess** — Denoise, binarize, crop, scale, and center each digit on a 28×28 canvas (details below).
3. **Normalize & split** — Scale pixels to `[0, 1]`, flatten to 784 features, stratified 80/20 split.
4. **Train & tune** — `GridSearchCV` over several MLP depth/width configs (accuracy + macro-F1).
5. **Evaluate** — CV metrics, confusion matrix, final held-out test accuracy.

---

## Preprocessing in detail

Raw DIDA digits vary in size, position, ink thickness, and often have background noise or stains. The goal of preprocessing is to produce clean, size-normalized, centered binary digits similar in spirit to MNIST-style inputs.

All of this is implemented in `binarize_center_resize()` in the notebook.

### Step 1 — Denoise (median blur)

```python
denoised = cv2.medianBlur(img, 3)
```

A **3×3 median filter** removes salt-and-pepper noise and stained pixels without heavily blurring the stroke edges. This stabilizes the next clustering step.

### Step 2 — Binarize with K-Means (k = 2)

Instead of a fixed global threshold (e.g. Otsu), each image is treated independently:

1. Flatten grayscale pixels and run **K-Means with 2 clusters**.
2. The cluster with the **darker center** is treated as **ink**; the other is background.
3. Build a binary mask: ink → `255`, background → `0`.

```python
pixels = denoised.reshape(-1, 1).astype(float)
kmeans = KMeans(n_clusters=2, n_init=10, max_iter=300)
labels = kmeans.fit_predict(pixels)
ink_label = np.argmin(kmeans.cluster_centers_)
binary_img = (labels == ink_label).reshape(denoised.shape).astype(np.uint8) * 255
```

**Why K-Means?** Illumination and paper texture differ across samples. Adaptive two-cluster separation is more robust than a single threshold for this dataset.

### Step 3 — Crop to the digit bounding box

```python
x, y, w, h = cv2.boundingRect(binary_img)
digit_crop = binary_img[y:y+h, x:x+w]
```

Only the ink region is kept. Empty (all-background) images fall back to a simple resize of the full binary mask.

### Step 4 — Scale longest side to 20 px

```python
f = 20.0 / max(w, h)
new_w, new_h = max(1, int(w * f)), max(1, int(h * f))
digit_resized = cv2.resize(digit_crop, (new_w, new_h))
```

The digit is scaled so its **longest side is 20 pixels**, leaving a margin when placed on a 28×28 canvas (similar to classic MNIST preprocessing).

### Step 5 — Center on a 28×28 canvas

```python
final_img = np.zeros((28, 28), dtype=np.uint8)
start_y = (28 - new_h) // 2
start_x = (28 - new_w) // 2
final_img[start_y:start_y+new_h, start_x:start_x+new_w] = digit_resized
```

The resized digit is pasted into the center of a black 28×28 image so the model always sees a consistent spatial layout.

### Step 6 — Normalize & flatten

```python
normalized = (imgs.astype(np.float32) / 255.0).reshape((-1, 784))
```

- Pixel range → `[0.0, 1.0]`
- Shape → `(N, 784)` for the MLP input layer

### Preprocessing summary

| Stage | Operation | Effect |
|-------|-----------|--------|
| 1 | Median blur (3×3) | Remove stain / salt-pepper noise |
| 2 | K-Means (k=2) | Separate ink from background per image |
| 3 | Bounding box crop | Discard empty margins |
| 4 | Scale max side → 20 | Normalize digit size |
| 5 | Center on 28×28 | Fixed spatial position |
| 6 | `/255` + flatten | MLP-ready feature vectors |

```text
Raw grayscale     Denoised        Binary (ink)     Cropped      Scaled+centered
   ┌─────┐         ┌─────┐         ┌─────┐        ┌───┐         ┌────────┐
   │ ░░3░ │   →    │  3  │   →    │ ███ │   →   │███│   →    │        │
   │ ░██░ │        │ ███ │        │ ███ │        │███│         │  ████  │
   │ ░░█░ │        │  █  │        │  █  │        │ █ │         │  ████  │
   └─────┘         └─────┘         └─────┘        └───┘         │   ██   │
                                                                │        │
                                                                └────────┘
                                                                  28 × 28
```

---

## Model & hyperparameter search

`GridSearchCV` (5-fold stratified CV) searched over four MLP architectures:

| Layers | Structure | Mean CV Accuracy | Mean F1 | Fit time (s) |
|--------|-----------|------------------|---------|--------------|
| **2** | **`(700, 200)`** | **0.81** | **0.81** | ~78 |
| 3 | `(500, 100, 10)` | 0.78 | 0.78 | ~86 |
| 6 | `(1000, 500, 200, 100, 50, 5)` | 0.77 | 0.77 | ~183 |
| 7 | `(350, 250, 150, 100, 70, 25, 10)` | 0.75 | 0.75 | ~79 |

**Winner:** `(700, 200)` — best accuracy with relatively low training time. Deeper nets did not help on this feature representation.

Default MLP settings (unless overridden by the search):

- Activation: ReLU  
- Solver: Adam  
- `max_iter=2000`, `random_state=42`

---

## Results

| Metric | Value |
|--------|-------|
| Best params | `hidden_layer_sizes=(700, 200)` |
| Mean CV accuracy | ≈ **0.813** |
| Final **test** accuracy | ≈ **0.817** |

### Confusion matrix (5-fold CV on train set)

The notebook plots a heatmap of the CV confusion matrix (`sns.heatmap`). Most digits are recognized reliably; residual errors tend to occur between visually similar pairs (e.g. **3↔8**, **4↔9**, **5↔6**), which is expected for stroke-based digit data after aggressive binarization.

> Open `OCR_Enhanced_Final.ipynb` and run the evaluation cells to view:
> - **CV confusion matrix** heatmap  
> - **Bar chart** of mean accuracy per architecture  
> - Training loss curves printed during MLP fitting  

### Architecture comparison (from notebook)

```text
Mean CV Accuracy by depth

(700, 200)           ████████████████████  0.81
(500, 100, 10)       ███████████████████   0.78
(1000…5) 6 layers   ███████████████████   0.77
(350…10) 7 layers   ██████████████████    0.75
```

---

## Requirements

```text
python >= 3.10
numpy
pandas
scikit-learn
opencv-python
matplotlib
seaborn
joblib
```

```bash
pip install numpy pandas scikit-learn opencv-python matplotlib seaborn joblib
```

## How to run

1. Clone the repo and ensure the `DIDA` folder is present at the project root.
2. Open `OCR_Enhanced_Final.ipynb` in Jupyter / VS Code / Google Colab.
3. Run cells in order. The notebook will:
   - Load and preprocess images  
   - Run `GridSearchCV` (several minutes on CPU)  
   - Show CV results, **confusion matrix**, accuracy **bar chart**, and test accuracy  

> **Tip:** For faster experiments, lower `files_per_folder` in `load_dataset()` (e.g. `200`). Hyperparameter search uses `n_jobs=-1` and is CPU-heavy.

## Key functions

| Function | Purpose |
|----------|---------|
| `load_dataset(root_path, files_per_folder=1000)` | Load images + labels from DIDA folders |
| `binarize_center_resize(imgs, target_size=(28, 28))` | Denoise → K-Means binarize → crop → scale → center |
| `normalize_flatten_split(imgs, target)` | Normalize to [0,1], flatten, stratified split |

## Author

**Ahmed Magdy** ([@Sober-Migo](https://github.com/Sober-Migo))  
Computer Science graduate specializing in AI/ML development and backend integration.

## License

This project is provided for educational purposes. Feel free to use and adapt it for learning.
