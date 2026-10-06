"""Shared helpers: load the processed dataset, honour its chronological split, metrics, results table."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

ROOT = Path(__file__).parent
DATA = ROOT / "data/processed/chicago_domestic_minimal.csv"
OUT_DIR = ROOT / "outputs"
RESULTS = OUT_DIR / "leakfree_results.csv"
TARGET = "Domestic"
FEATURES = ["Primary Type", "Location Description", "Beat"]   # all treated as categorical
SEED = 42


def load_splits():
    df = pd.read_csv(DATA)
    for c in FEATURES:
        df[c] = df[c].astype(str)                  # Beat is a code, not a quantity
    parts = []
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        parts.append((d[FEATURES].reset_index(drop=True), d[TARGET].to_numpy()))
    return parts                                   # (Xtr,ytr), (Xva,yva), (Xte,yte)


def best_threshold(y_val, p_val):
    """Threshold chosen on VALIDATION only: maximises min(accuracy, balanced accuracy)."""
    ts = np.linspace(0.05, 0.95, 91)
    sc = [min(accuracy_score(y_val, p_val >= t), balanced_accuracy_score(y_val, p_val >= t)) for t in ts]
    return float(ts[int(np.argmax(sc))])


def report(name, y, p, t):
    pred = (p >= t).astype(int)
    a, b = accuracy_score(y, pred), balanced_accuracy_score(y, pred)
    print(f"{name:<22} thr={t:.2f}  accuracy={a:.4f}  balanced_acc={b:.4f}  gap={abs(a-b):.4f}")
    return a, b


def metrics_row(name, y, p, t):
    """One row of the final results table (computed on the TEST set)."""
    pred = (p >= t).astype(int)
    return {"Model": name,
            "Features": " + ".join(FEATURES),
            "Train/Val/Test": "2023-Jun24 / Jul-Dec24 / 2025",
            "Test_rows": len(y),
            "Threshold": round(t, 2),
            "Accuracy": round(accuracy_score(y, pred), 4),
            "Balanced_Accuracy": round(balanced_accuracy_score(y, pred), 4),
            "Precision": round(precision_score(y, pred), 4),
            "Recall": round(recall_score(y, pred), 4),
            "F1": round(f1_score(y, pred), 4),
            "ROC_AUC": round(roc_auc_score(y, p), 4)}


def save_rows(rows):
    """Add/replace rows in outputs/leakfree_results.csv (so ML and DL scripts share one table)."""
    OUT_DIR.mkdir(exist_ok=True)
    new = pd.DataFrame(rows)
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        new = pd.concat([old[~old.Model.isin(new.Model)], new], ignore_index=True)
    new.to_csv(RESULTS, index=False)
    return new