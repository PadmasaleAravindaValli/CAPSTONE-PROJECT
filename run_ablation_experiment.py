"""Feature Ablation Experiment: Dataset 1 (Base 14 Features) vs Dataset 2 (Enhanced 29 Features).

Evaluates models under identical chronological train/val/test splits to quantify
the exact performance gain contributed by the 15 engineered features:
- Cyclic trigonometric transforms
- Semantic location flags (Is_Residential, Is_Public_Space)
- Interaction (Type @ Location)
- Bayesian empirical domestic risk rates
- Spatial incident volume
"""
from pathlib import Path
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent
D1_PATH = PROJECT_ROOT / "data" / "processed" / "dataset1_important_features.csv"
D2_PATH = PROJECT_ROOT / "data" / "processed" / "dataset2_enhanced_features.csv"

OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "Domestic"


def tune_threshold(y_val, p_val):
    ts = np.linspace(0.05, 0.95, 91)
    scores = [min(accuracy_score(y_val, p_val >= t), balanced_accuracy_score(y_val, p_val >= t)) for t in ts]
    return float(ts[int(np.argmax(scores))])


def evaluate(y_true, p_pred, threshold):
    y_pred = (p_pred >= threshold).astype(int)
    return {
        "Threshold": round(threshold, 3),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Balanced_Acc": round(balanced_accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1_Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, p_pred), 4)
    }


def run_experiment_on_dataset(df, cat_cols, num_cols, dataset_label):
    print(f"\n=======================================================")
    print(f"EVALUATING MODELS ON {dataset_label.upper()}")
    print(f"Features: {len(cat_cols) + len(num_cols)} ({len(cat_cols)} cat, {len(num_cols)} num)")
    print(f"=======================================================")

    for c in cat_cols:
        df[c] = df[c].fillna("UNKNOWN").astype(str)
    for c in num_cols:
        df[c] = df[c].fillna(0.0).astype(float)

    feature_cols = cat_cols + num_cols

    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]

    Xtr, ytr = tr[feature_cols], tr[TARGET].astype(int).to_numpy()
    Xva, yva = va[feature_cols], va[TARGET].astype(int).to_numpy()
    Xte, yte = te[feature_cols], te[TARGET].astype(int).to_numpy()

    spw = float((ytr == 0).sum() / (ytr == 1).sum())

    onehot = lambda: ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cat_cols)
    ], remainder="passthrough")

    target_enc = lambda: ColumnTransformer([
        ("te", TargetEncoder(target_type="binary", cv=5, random_state=SEED), cat_cols)
    ], remainder="passthrough")

    models = {
        "Logistic Regression": Pipeline([
            ("p", onehot()),
            ("m", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=SEED, n_jobs=1))
        ]),
        "HistGradientBoosting": Pipeline([
            ("p", target_enc()),
            ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, l2_regularization=1.0,
                                                class_weight="balanced", early_stopping=True, random_state=SEED))
        ]),
        "XGBoost": Pipeline([
            ("p", onehot()),
            ("m", xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.08, subsample=0.8,
                                    colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                                    n_jobs=1, random_state=SEED, eval_metric="logloss"))
        ])
    }

    results = []
    for name, pipe in models.items():
        print(f"Fitting {name}...")
        pipe.fit(Xtr, ytr)
        p_va = pipe.predict_proba(Xva)[:, 1]
        t = tune_threshold(yva, p_va)
        p_te = pipe.predict_proba(Xte)[:, 1]
        m = evaluate(yte, p_te, t)
        m["Model"] = name
        m["Dataset"] = dataset_label
        results.append(m)
        print(f" -> [{name}] Accuracy: {m['Accuracy']:.4f} | Balanced Acc: {m['Balanced_Acc']:.4f} | F1: {m['F1_Score']:.4f} | ROC-AUC: {m['ROC_AUC']:.4f}")

    return results


def main():
    print(f"Loading Dataset 1 from {D1_PATH}...")
    df1 = pd.read_csv(D1_PATH)
    d1_cat = ["Primary Type", "Location Description", "Beat", "District", "Community Area"]
    d1_num = ["Hour", "Month", "DayOfWeek", "IsWeekend", "Latitude", "Longitude"]
    res1 = run_experiment_on_dataset(df1, d1_cat, d1_num, "Dataset 1 (Base 14 Features)")

    print(f"\nLoading Dataset 2 from {D2_PATH}...")
    df2 = pd.read_csv(D2_PATH)
    d2_cat = [
        "Primary Type", "Location Description", "Beat", "District", "Community Area",
        "Time_Period", "Type_Location_Interaction"
    ]
    d2_num = [
        "Hour", "Month", "DayOfWeek", "IsWeekend", "Latitude", "Longitude",
        "Hour_Sin", "Hour_Cos", "Month_Sin", "Month_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos",
        "Is_Night", "Is_Residential", "Is_Public_Space",
        "Primary_Type_Domestic_Risk", "Location_Domestic_Risk", "Beat_Domestic_Risk",
        "Beat_Historical_Incident_Volume"
    ]
    res2 = run_experiment_on_dataset(df2, d2_cat, d2_num, "Dataset 2 (Enhanced 29 Features)")

    # Combine into comparison dataframe
    all_res = res1 + res2
    comp_df = pd.DataFrame(all_res)

    out_csv = OUT_TABLES / "ablation_d1_vs_d2.csv"
    comp_df.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print("FINAL ABLATION COMPARISON (2025 LOCKED TEST PERIOD)")
    print("=" * 80)
    print(comp_df[["Model", "Dataset", "Balanced_Acc", "F1_Score", "ROC_AUC", "Accuracy"]].to_string(index=False))
    print(f"\nSaved comparison table to: {out_csv}")

    # Plot comparison bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=comp_df, x="Model", y="Balanced_Acc", hue="Dataset", palette="Set2")
    plt.title("Ablation Study: Base Features (Dataset 1) vs. Enhanced Features (Dataset 2)", fontsize=13, fontweight="bold")
    plt.ylabel("Balanced Accuracy (Validation Tuned)", fontsize=11)
    plt.xlabel("Model Architecture", fontsize=11)
    plt.ylim(0.70, 0.90)
    for p in plt.gca().patches:
        h = p.get_height()
        if h > 0:
            plt.gca().annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                               ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')
    plt.tight_layout()
    chart_path = OUT_FIGURES / "ablation_d1_vs_d2.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved figure to: {chart_path}")


if __name__ == "__main__":
    main()
