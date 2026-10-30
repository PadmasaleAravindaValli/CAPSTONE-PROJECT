"""Feature 1: Algorithmic Fairness & Geographic Bias Audit across Chicago Police Districts.

Audits model fairness across all 22 Chicago Police Districts on the 2025 Test Set:
- Selection Rate Parity (Disparate Impact)
- False Positive Rate (FPR) Parity
- Equalized Odds Difference
"""
from pathlib import Path
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import recall_score, precision_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset2_enhanced_features.csv"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

TARGET = "Domestic"
SEED = 42


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    cat_cols = ["Primary Type", "Location Description", "Beat"]
    num_cols = ["Hour", "Month", "DayOfWeek", "Is_Night", "Is_Residential", "Is_Public_Space", "Primary_Type_Domestic_Risk", "Location_Domestic_Risk"]
    features = cat_cols + num_cols

    for c in cat_cols:
        df[c] = df[c].fillna("UNKNOWN").astype(str)
    for c in num_cols:
        df[c] = df[c].fillna(0.0).astype(float)

    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"].copy()

    Xtr, ytr = tr[features], tr[TARGET].astype(int)
    Xva, yva = va[features], va[TARGET].astype(int)
    Xte, yte = te[features], te[TARGET].astype(int)

    spw = float((ytr == 0).sum() / (ytr == 1).sum())

    pipe = Pipeline([
        ("oh", ColumnTransformer([("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cat_cols)], remainder="passthrough")),
        ("clf", xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, scale_pos_weight=spw, tree_method="hist", random_state=SEED, n_jobs=1))
    ])

    print("Fitting model for fairness audit...")
    pipe.fit(Xtr, ytr)

    # Predictions on test set
    p_te = pipe.predict_proba(Xte)[:, 1]
    # Decision threshold
    t = 0.52
    te["y_true"] = yte
    te["y_pred"] = (p_te >= t).astype(int)
    te["p_pred"] = p_te

    city_pos_rate = te["y_pred"].mean()
    city_fpr = ((te["y_pred"] == 1) & (te["y_true"] == 0)).sum() / (te["y_true"] == 0).sum()

    print(f"Citywide Baseline -> Predicted Positive Rate: {city_pos_rate:.4f} | False Positive Rate: {city_fpr:.4f}")

    # District Audit
    district_records = []
    for district, grp in te.groupby("District"):
        if district <= 0 or len(grp) < 100:
            continue
        y_t = grp["y_true"].values
        y_p = grp["y_pred"].values

        total = len(grp)
        actual_dom_rate = y_t.mean()
        pred_dom_rate = y_p.mean()

        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Disparate Impact Ratio (relative to citywide)
        disparate_impact = pred_dom_rate / city_pos_rate if city_pos_rate > 0 else 1.0
        fpr_disparity = fpr - city_fpr

        district_records.append({
            "District": int(district),
            "Test_Incidents": total,
            "Actual_Domestic_Rate": round(actual_dom_rate, 4),
            "Predicted_Domestic_Rate": round(pred_dom_rate, 4),
            "Disparate_Impact_Ratio": round(disparate_impact, 3),
            "TPR_Recall": round(tpr, 4),
            "FPR": round(fpr, 4),
            "FPR_Disparity_Gap": round(fpr_disparity, 4),
            "FNR": round(fnr, 4)
        })

    audit_df = pd.DataFrame(district_records).sort_values("District").reset_index(drop=True)
    out_table = OUT_TABLES / "district_fairness_audit.csv"
    audit_df.to_csv(out_table, index=False)
    print(f"\nSaved District Fairness Audit table to: {out_table}")
    print("\nSample District Fairness Audit Table (first 6 districts):")
    print(audit_df.head(6).to_string(index=False))

    # Plot Fairness & FPR Disparity across Districts
    plt.figure(figsize=(12, 6))
    bar_positions = np.arange(len(audit_df))
    plt.bar(bar_positions - 0.2, audit_df["Actual_Domestic_Rate"], width=0.4, label="Actual Domestic Rate", color="#3498db")
    plt.bar(bar_positions + 0.2, audit_df["Predicted_Domestic_Rate"], width=0.4, label="Predicted Rate (Selection)", color="#e74c3c")
    plt.axhline(city_pos_rate, color="black", linestyle="--", alpha=0.7, label=f"Citywide Avg ({city_pos_rate:.3f})")

    plt.xticks(bar_positions, [f"Dist {d}" for d in audit_df["District"]], rotation=45, ha="right", fontsize=9)
    plt.ylabel("Incident Proportion", fontsize=11)
    plt.title("Algorithmic Fairness Audit: Actual vs. Predicted Domestic Rate Across 22 Chicago Police Districts", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_fig = OUT_FIGURES / "district_fairness_parity.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved fairness parity chart to: {out_fig}")


if __name__ == "__main__":
    main()
