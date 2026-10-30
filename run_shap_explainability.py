"""Model Explainability with SHAP (Explainable AI / XAI).

Computes Shapley values using TreeExplainer on XGBoost to explain:
1. Which features are the strongest overall drivers of Domestic predictions.
2. How individual feature values (e.g. Is_Residential=1 vs Is_Public_Space=1)
   push predictions toward or away from Domestic incidents.
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

from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import shap

PROJECT_ROOT = Path(__file__).resolve().parent
D2_PATH = PROJECT_ROOT / "data" / "processed" / "dataset2_enhanced_features.csv"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "Domestic"


def main():
    print(f"Loading Dataset 2 from {D2_PATH}...")
    df = pd.read_csv(D2_PATH)

    # Features for explanation
    cat_cols = ["Primary Type", "Location Description", "Beat", "Time_Period"]
    num_cols = [
        "Hour", "Month", "DayOfWeek", "IsWeekend",
        "Hour_Sin", "Hour_Cos", "Is_Night",
        "Is_Residential", "Is_Public_Space",
        "Primary_Type_Domestic_Risk", "Location_Domestic_Risk", "Beat_Domestic_Risk",
        "Beat_Historical_Incident_Volume"
    ]
    all_features = cat_cols + num_cols

    for c in cat_cols:
        df[c] = df[c].fillna("UNKNOWN").astype(str)
    for c in num_cols:
        df[c] = df[c].fillna(0.0).astype(float)

    # Ordinal encode categoricals for fast tree explanation
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = oe.fit_transform(df[cat_cols])

    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]

    # Sample for training and SHAP explanation
    tr_sample = tr.sample(min(80_000, len(tr)), random_state=SEED)
    te_sample = te.sample(min(2_500, len(te)), random_state=SEED)

    Xtr, ytr = tr_sample[all_features], tr_sample[TARGET].astype(int)
    Xte, yte = te_sample[all_features], te_sample[TARGET].astype(int)

    print("Training XGBoost for SHAP explanation...")
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.08,
        scale_pos_weight=spw, tree_method="hist", random_state=SEED, n_jobs=1
    )
    model.fit(Xtr, ytr)

    print("Computing TreeExplainer SHAP values on 2,500 test incidents...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(Xte)

    # 1. SHAP Beeswarm Summary Plot
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("SHAP Beeswarm: Impact of Features on Domestic Incident Prediction", fontsize=12, fontweight="bold")
    plt.tight_layout()
    beeswarm_path = OUT_FIGURES / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP Beeswarm to: {beeswarm_path}")

    # 2. SHAP Global Feature Importance Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title("SHAP Global Feature Importance (|mean SHAP value|)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    bar_path = OUT_FIGURES / "shap_feature_importance_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP Bar plot to: {bar_path}")

    # 3. Save Importance Table
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": all_features,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    table_path = OUT_TABLES / "shap_feature_importance.csv"
    importance_df.to_csv(table_path, index=False)
    print(f"Saved SHAP importance table to: {table_path}")
    print("\nTop 8 Most Influential Features According to SHAP:")
    print(importance_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
