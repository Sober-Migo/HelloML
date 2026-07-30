"""HelloML helpers: preprocessing, models, save/load."""
from pathlib import Path
import time
import warnings
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ARTIFACT_KEYS = [
    "X_train", "X_test", "y_train", "y_test",
    "X_proc", "y",
    "best_estimators",
    "df_cv_all", "df_cv_best", "df_test",
]


def save_artifacts(directory, g, keys=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    keys = keys or ARTIFACT_KEYS
    saved = []
    for name in keys:
        if name not in g or g[name] is None:
            print(f"  skip (missing): {name}")
            continue
        path = directory / f"{name}.joblib"
        joblib.dump(g[name], path)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  saved {name:20s} -> {path.name}  ({size_mb:.2f} MB)")
        saved.append(name)
    print(f"Done. Saved {len(saved)} artifact(s) to {directory}")
    return saved


def load_artifacts(directory, g, keys=None):
    directory = Path(directory)
    keys = keys or ARTIFACT_KEYS
    loaded, missing = [], []
    for name in keys:
        path = directory / f"{name}.joblib"
        if not path.exists():
            missing.append(name)
            continue
        g[name] = joblib.load(path)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  loaded {name:20s} <- {path.name}  ({size_mb:.2f} MB)")
        loaded.append(name)
    if missing:
        print(f"  missing: {missing}")
    print(f"Done. Loaded {len(loaded)} artifact(s) from {directory}")
    return loaded


def artifacts_exist(directory, required=None):
    directory = Path(directory)
    required = required or ["X_train", "X_test", "y_train", "y_test", "best_estimators"]
    return all((directory / f"{k}.joblib").exists() for k in required)


def load_dataset(root_path, files_per_folder=1000, seed=42):
    root_path = Path(root_path)
    print(f"Loading data from: {root_path}")
    t0 = time.time()
    X, y = [], []
    rng = np.random.default_rng(seed)
    for digit in range(10):
        folder = root_path / str(digit)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        all_files = [p for p in folder.iterdir() if p.is_file()]
        if len(all_files) < files_per_folder:
            raise ValueError(
                f"Folder '{digit}' has {len(all_files)} images, need >= {files_per_folder}."
            )
        chosen = sorted(rng.choice(all_files, size=files_per_folder, replace=False))
        for path in chosen:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: could not read {path}")
                continue
            X.append(img)
            y.append(digit)
    y = np.array(y, dtype=np.int64)
    print(f"Loaded {len(X)} images in {time.time() - t0:.1f}s")
    return X, y


def binarize_center_resize(imgs, target_size=(28, 28)):
    H, W = target_size
    out = np.zeros((len(imgs), H, W), dtype=np.uint8)
    for i, img in enumerate(imgs):
        denoised = cv2.medianBlur(img, 3)
        pixels = denoised.reshape(-1, 1).astype(np.float64)
        kmeans = KMeans(n_clusters=2, n_init=10, max_iter=300, random_state=42)
        labels = kmeans.fit_predict(pixels)
        ink_label = int(np.argmin(kmeans.cluster_centers_))
        binary = (labels == ink_label).reshape(denoised.shape).astype(np.uint8) * 255
        coords = np.column_stack(np.where(binary > 0))
        if coords.shape[0] > 0:
            x, y, w, h = cv2.boundingRect(binary)
            crop = binary[y:y+h, x:x+w]
            scale = 20.0 / max(w, h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((H, W), dtype=np.uint8)
            sy = (H - new_h) // 2
            sx = (W - new_w) // 2
            canvas[sy:sy+new_h, sx:sx+new_w] = resized
            out[i] = canvas
        else:
            out[i] = cv2.resize(binary, (W, H))
    print(f"Preprocessed images: {out.shape}")
    return out


def normalize_flatten_split(imgs, y, test_size=0.20, seed=42):
    X = (imgs.astype(np.float32) / 255.0).reshape(len(imgs), -1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_experiment_setup():
    return {
        "NaiveBayes": {"model": GaussianNB(), "params": {}},
        "LinearReg_OvA": {
            "model": OneVsRestClassifier(LinearRegression()),
            "params": {},
        },
        "LogisticReg": {
            "model": LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42),
            "params": {"C": [1.0, 10.0]},
        },
        "MLP": {
            "model": MLPClassifier(random_state=42, max_iter=2000),
            "params": {
                "hidden_layer_sizes": [
                    (700, 200),
                    (500, 100, 10),
                    (300, 150, 100, 50, 10),
                ],
            },
        },
    }


SCORING_METRICS = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
}
