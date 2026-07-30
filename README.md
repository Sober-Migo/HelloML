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
├── OCR_Enhanced_Final.ipynb   # Main notebook (full pipeline)
├── DIDA/                      # Raw digit images (folders 0–9)
├── Processed_Data/            # Optional preprocessed images
└── README.md
```

## Pipeline overview

1. **Load data**  
   Read grayscale images from `DIDA/0` … `DIDA/9` (1,000 per class, reproducible sampling).

2. **Preprocess** (`binarize_center_resize`)
   - Median blur (denoise)
   - K-Means (k=2) to separate ink vs background
   - Crop to bounding box, scale longest side to 20 px, center on a 28×28 canvas

3. **Normalize & split**  
   Pixel values → `[0, 1]`, flatten to 784 features, stratified train/test split.

4. **Train & tune**  
   `GridSearchCV` over several MLP depth/width configs with 5-fold CV (accuracy + macro-F1).

5. **Evaluate**  
   Report best CV metrics, confusion matrix, and final held-out test accuracy.

### Architectures compared

| Layers | Structure | Mean CV Accuracy |
|--------|-----------|------------------|
| 2 | `(700, 200)` | **0.81** |
| 3 | `(500, 100, 10)` | 0.78 |
| 6 | `(1000, 500, 200, 100, 50, 5)` | 0.77 |
| 7 | `(350, 250, 150, 100, 70, 25, 10)` | 0.75 |

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

Install with:

```bash
pip install numpy pandas scikit-learn opencv-python matplotlib seaborn joblib
```

## How to run

1. Clone the repo and ensure the `DIDA` folder is present at the project root.
2. Open `OCR_Enhanced_Final.ipynb` in Jupyter / VS Code / Google Colab.
3. Run cells in order. The notebook:
   - Loads and preprocesses images
   - Runs GridSearchCV (can take several minutes)
   - Prints CV results, confusion matrix, and final test accuracy

> **Note:** Hyperparameter search uses `n_jobs=-1` and can be CPU-heavy. Reduce `files_per_folder` in `load_dataset()` for quicker experiments.

## Key functions

| Function | Purpose |
|----------|---------|
| `load_dataset(root_path, files_per_folder=1000)` | Load images + labels from DIDA folders |
| `binarize_center_resize(imgs, target_size=(28, 28))` | Denoise, binarize, center, resize |
| `normalize_flatten_split(imgs, target)` | Normalize, flatten, train/test split |

## Results summary

- **Best params:** `hidden_layer_sizes=(700, 200)`
- **Mean CV accuracy:** ≈ 0.813
- **Final test accuracy:** ≈ 0.8165

## Author

**Ahmed Magdy** ([@Sober-Migo](https://github.com/Sober-Migo))  
Computer Science graduate specializing in AI/ML development and backend integration.

## License

This project is provided for educational purposes. Feel free to use and adapt it for learning.
