"""
Shared evaluation utilities (ICT 4442 mini project).
EVERY model owner should import this file so all four models are scored identically.
Owner: Varaang Tiwari (230953378)
"""
import json
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)


def compute_metrics(y_true, y_pred):
    """Accuracy, precision, recall, F1 (macro AND per-class) and confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_c, r_c, f_c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),
        "f1_macro": float(f_m),
        "precision_per_class": [float(x) for x in p_c],   # [neg, pos]
        "recall_per_class": [float(x) for x in r_c],
        "f1_per_class": [float(x) for x in f_c],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_samples": int(len(y_true)),
    }


def accuracy_drop(clean_acc, shifted_acc):
    """Absolute drop in accuracy (percentage points) from clean to shifted/perturbed data.
    Used later for the cross-domain and perturbation experiments (Phase 3)."""
    return float((clean_acc - shifted_acc) * 100)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
