"""Builds the final results table + chart from the models already saved in models/."""
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_utils import *

(_, _), (Xva, yva), (Xte, yte) = load_splits()
rows = []
for name in ["LogisticRegression", "RandomForest", "HistGradientBoosting", "XGBoost"]:
    pipe = joblib.load(ROOT / f"models/{name}.joblib")
    t = best_threshold(yva, pipe.predict_proba(Xva)[:, 1])          # threshold from VALIDATION
    rows.append(metrics_row(name, yte, pipe.predict_proba(Xte)[:, 1], t))   # scores on TEST
tbl = save_rows(rows)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\nFINAL RESULTS (test set = all of 2025)\n")
print(tbl.drop(columns=["Features", "Train/Val/Test"]).to_string(index=False))
print("\nFeatures used by every model:", " + ".join(FEATURES))
print("Saved:", RESULTS)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(tbl)); w = 0.38
b1 = ax.bar(x - w/2, tbl.Accuracy, w, label="Accuracy")
b2 = ax.bar(x + w/2, tbl.Balanced_Accuracy, w, label="Balanced accuracy")
ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
ax.axhline(0.8, color="red", ls="--", lw=1, label="0.80 target")
ax.set_xticks(x); ax.set_xticklabels(tbl.Model, rotation=15); ax.set_ylim(0.7, 0.9)
ax.set_ylabel("Score (test set, 2025)"); ax.set_title("Leakage-free results: predicting Domestic")
ax.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "leakfree_accuracy_chart.png", dpi=150)
print("Saved:", OUT_DIR / "leakfree_accuracy_chart.png")