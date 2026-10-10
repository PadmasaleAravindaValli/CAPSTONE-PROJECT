"""Builds the final results table + chart from the models already saved in models/."""
import joblib, matplotlib, numpy as np, pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_utils import (ROOT, OUT_DIR, RESULTS, FEATURES, load_splits, best_threshold, metrics_row, save_rows)

(_, _), (Xva, yva), (Xte, yte) = load_splits()
rows = []
for name in ["LogisticRegression", "RandomForest", "HistGradientBoosting", "XGBoost"]:
    path = ROOT / f"models/{name}.joblib"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Train and save the ML models first "
                         f"(they must be trained on the current features: {FEATURES}).")
    pipe = joblib.load(path)
    t = best_threshold(yva, pipe.predict_proba(Xva)[:, 1])          # threshold from VALIDATION
    rows.append(metrics_row(name, yte, pipe.predict_proba(Xte)[:, 1], t))   # scores on TEST
tbl = save_rows(rows)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\nFINAL RESULTS (test set = all of 2025)\n")
print(tbl.drop(columns=["Features", "Train/Val/Test"]).to_string(index=False))
print("\nFeatures used by every model:", " + ".join(FEATURES))
print("Saved:", RESULTS)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(tbl)); w = 0.38
b1 = ax.bar(x - w/2, tbl["Accuracy"], w, label="Accuracy")
b2 = ax.bar(x + w/2, tbl["Balanced_Accuracy"], w, label="Balanced accuracy")
ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
ax.axhline(0.8, color="red", ls="--", lw=1, label="0.80 target")
ax.set_xticks(x); ax.set_xticklabels(tbl["Model"], rotation=20, ha="right")
ax.set_ylim(0, 1.0)                                  # full axis: bars start at 0 and nothing is clipped
ax.set_ylabel("Score (test set, 2025)"); ax.set_title("Leakage-free results: predicting Domestic")
ax.legend(loc="lower right"); plt.tight_layout()
OUT_DIR.mkdir(exist_ok=True)
plt.savefig(OUT_DIR / "leakfree_accuracy_chart.png", dpi=150)
print("Saved:", OUT_DIR / "leakfree_accuracy_chart.png")