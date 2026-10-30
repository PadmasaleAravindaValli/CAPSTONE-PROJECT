"""Step 9: Comprehensive Feature Ablation and Feature Comparison Study.

Conducts:
1. Feature Comparison Ranking (Correlation, Mutual Info, XGBoost Gain Importance)
2. Cumulative Step-by-Step Feature Ablation (Temporal -> Spatial -> Categories -> Cyclical -> Venue -> Bayesian Priors -> Interaction)
3. Leave-One-Group-Out (LOGO) Ablation
Outputs saved to outputs/tables/ and outputs/figures/.
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

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset2_enhanced_features.csv"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

TARGET = "Domestic"
SEED = 42


def tune_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [balanced_accuracy_score(y_true, y_prob >= t) for t in thresholds]
    return float(thresholds[np.argmax(scores)])


def evaluate_preds(y_true, y_prob, threshold, step_name):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Configuration": step_name,
        "Threshold": round(threshold, 3),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Balanced_Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1_Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, y_prob), 4)
    }


def main():
    print(f"Loading Dataset 2 from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    cat_cols_all = ["Primary Type", "Location Description", "Beat", "District", "Community Area", "Time_Period", "Type_Location_Interaction"]
    num_cols_all = [
        "Hour", "Month", "DayOfWeek", "IsWeekend", "Latitude", "Longitude",
        "Hour_Sin", "Hour_Cos", "Month_Sin", "Month_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos",
        "Is_Night", "Is_Residential", "Is_Public_Space",
        "Primary_Type_Domestic_Risk", "Location_Domestic_Risk", "Beat_Domestic_Risk",
        "Beat_Historical_Incident_Volume"
    ]

    for c in cat_cols_all:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str)
    for c in num_cols_all:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median()).astype(float)

    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"].copy()

    # Subsample for mutual information and fast feature importance
    sub_sample = tr.sample(min(80_000, len(tr)), random_state=SEED)
    spw = float((tr[TARGET] == 0).sum() / (tr[TARGET] == 1).sum())

    # =========================================================================
    # PART 1: FEATURE COMPARISON & RANKING
    # =========================================================================
    print("\n--- Part 1: Computing Feature Comparison & Ranking Metrics ---")
    numeric_for_mi = [c for c in num_cols_all if c in df.columns]
    X_mi = sub_sample[numeric_for_mi]
    y_mi = sub_sample[TARGET].astype(int)

    mi_scores = mutual_info_classif(X_mi, y_mi, random_state=SEED)
    correlations = [sub_sample[c].corr(sub_sample[TARGET]) for c in numeric_for_mi]

    feature_comp = pd.DataFrame({
        "Feature": numeric_for_mi,
        "Correlation_with_Domestic": np.round(correlations, 4),
        "Abs_Correlation": np.round(np.abs(correlations), 4),
        "Mutual_Information": np.round(mi_scores, 4)
    })

    # Add feature category classification
    def get_category(f):
        if "Risk" in f: return "Bayesian Risk Prior"
        if "Sin" in f or "Cos" in f: return "Cyclical Temporal"
        if "Is_" in f or "Time_Period" in f: return "Venue / Shift Flag"
        if f in ["Hour", "Month", "DayOfWeek", "IsWeekend"]: return "Raw Temporal"
        if f in ["Beat", "District", "Community Area", "Latitude", "Longitude", "Beat_Historical_Incident_Volume"]: return "Spatial / Volume"
        return "Categorical / Interaction"

    feature_comp["Category"] = feature_comp["Feature"].apply(get_category)
    feature_comp = feature_comp.sort_values(by="Mutual_Information", ascending=False).reset_index(drop=True)

    out_comp_csv = OUT_TABLES / "feature_comparison_ranking.csv"
    feature_comp.to_csv(out_comp_csv, index=False)
    print(f"Saved Feature Comparison table to: {out_comp_csv}")

    # Plot Feature Comparison Ranking
    plt.figure(figsize=(10, 6))
    palette = {"Bayesian Risk Prior": "#e74c3c", "Venue / Shift Flag": "#f39c12", "Cyclical Temporal": "#3498db", "Raw Temporal": "#9b59b6", "Spatial / Volume": "#2ecc71"}
    sns.barplot(data=feature_comp, x="Mutual_Information", y="Feature", hue="Category", palette=palette, dodge=False)
    plt.title("Feature Comparison: Mutual Information with Domestic Violence Target", fontsize=12, fontweight="bold")
    plt.xlabel("Mutual Information Score", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_comp_png = OUT_FIGURES / "feature_comparison_ranking.png"
    plt.savefig(out_comp_png, dpi=300)
    plt.close()
    print(f"Saved Feature Comparison plot to: {out_comp_png}")

    # =========================================================================
    # PART 2: CUMULATIVE STEP-BY-STEP FEATURE ABLATION
    # =========================================================================
    print("\n--- Part 2: Running Cumulative Feature Ablation Progression ---")
    feature_stages = [
        ("1. Raw Temporal", ["Hour", "Month", "DayOfWeek", "IsWeekend"], []),
        ("2. + Spatial Baseline", ["Hour", "Month", "DayOfWeek", "IsWeekend", "Latitude", "Longitude"], ["Beat", "District", "Community Area"]),
        ("3. + Raw Crime Categories (Dataset 1)", ["Hour", "Month", "DayOfWeek", "IsWeekend", "Latitude", "Longitude"], ["Beat", "District", "Community Area", "Primary Type", "Location Description"]),
        ("4. + Cyclical Trigonometric Time", ["Hour", "Month", "DayOfWeek", "IsWeekend", "Latitude", "Longitude", "Hour_Sin", "Hour_Cos", "Month_Sin", "Month_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos"], ["Beat", "District", "Community Area", "Primary Type", "Location Description"]),
        ("5. + Venue & Shift Indicators", ["Hour", "Month", "DayOfWeek", "IsWeekend", "Latitude", "Longitude", "Hour_Sin", "Hour_Cos", "Month_Sin", "Month_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos", "Is_Night", "Is_Residential", "Is_Public_Space"], ["Beat", "District", "Community Area", "Primary Type", "Location Description", "Time_Period"]),
        ("6. + Bayesian Risk Priors & Volume", ["Hour", "Month", "DayOfWeek", "IsWeekend", "Latitude", "Longitude", "Hour_Sin", "Hour_Cos", "Month_Sin", "Month_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos", "Is_Night", "Is_Residential", "Is_Public_Space", "Primary_Type_Domestic_Risk", "Location_Domestic_Risk", "Beat_Domestic_Risk", "Beat_Historical_Incident_Volume"], ["Beat", "District", "Community Area", "Primary Type", "Location Description", "Time_Period"]),
        ("7. Full Model + Interaction (Dataset 2)", ["Hour", "Month", "DayOfWeek", "IsWeekend", "Latitude", "Longitude", "Hour_Sin", "Hour_Cos", "Month_Sin", "Month_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos", "Is_Night", "Is_Residential", "Is_Public_Space", "Primary_Type_Domestic_Risk", "Location_Domestic_Risk", "Beat_Domestic_Risk", "Beat_Historical_Incident_Volume"], ["Beat", "District", "Community Area", "Primary Type", "Location Description", "Time_Period", "Type_Location_Interaction"])
    ]

    ablation_results = []
    base_auc = None
    base_f1 = None

    for name, num_f, cat_f in feature_stages:
        print(f"Evaluating {name} ({len(num_f)} num, {len(cat_f)} cat)...")
        transformers = []
        if cat_f:
            transformers.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cat_f))
        if num_f:
            transformers.append(("num", StandardScaler(), num_f))

        prep = ColumnTransformer(transformers=transformers)
        cols_needed = cat_f + num_f

        X_tr_enc = prep.fit_transform(tr[cols_needed])
        X_va_enc = prep.transform(va[cols_needed])
        X_te_enc = prep.transform(te[cols_needed])

        clf = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, scale_pos_weight=spw, tree_method="hist", random_state=SEED, n_jobs=-1)
        clf.fit(X_tr_enc, tr[TARGET].astype(int))

        p_va = clf.predict_proba(X_va_enc)[:, 1]
        thr = tune_threshold(va[TARGET].astype(int), p_va)
        p_te = clf.predict_proba(X_te_enc)[:, 1]

        res = evaluate_preds(te[TARGET].astype(int), p_te, thr, name)
        if base_auc is None:
            base_auc = res["ROC_AUC"]
            base_f1 = res["F1_Score"]
        res["Cumulative_Delta_AUC"] = round(res["ROC_AUC"] - base_auc, 4)
        res["Cumulative_Delta_F1"] = round(res["F1_Score"] - base_f1, 4)
        ablation_results.append(res)
        print(f" -> Balanced Acc: {res['Balanced_Accuracy']:.4f} | F1: {res['F1_Score']:.4f} | ROC-AUC: {res['ROC_AUC']:.4f} | (Delta AUC: {res['Cumulative_Delta_AUC']:+.4f})")

    df_ablation = pd.DataFrame(ablation_results)
    out_abl_csv = OUT_TABLES / "feature_ablation_progression.csv"
    df_ablation.to_csv(out_abl_csv, index=False)
    print(f"\nSaved Feature Ablation Progression table to: {out_abl_csv}")

    # Plot Progression Bar Chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df_ablation))
    w = 0.35
    plt.bar(x - w/2, df_ablation["Balanced_Accuracy"], w, label="Balanced Accuracy", color="#3498db")
    plt.bar(x + w/2, df_ablation["ROC_AUC"], w, label="ROC-AUC", color="#2ecc71")
    plt.xticks(x, [f"Step {i+1}" for i in range(len(df_ablation))], fontsize=10)
    plt.ylim(0.55, 0.95)
    plt.title("Step-by-Step Cumulative Feature Ablation Progression (2025 Locked Test)", fontsize=13, fontweight="bold")
    plt.xlabel("Feature Progression Steps", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    for i in range(len(df_ablation)):
        plt.annotate(f"{df_ablation['ROC_AUC'].iloc[i]:.3f}", (x[i] + w/2, df_ablation["ROC_AUC"].iloc[i] + 0.01), ha="center", fontsize=9)
    plt.tight_layout()
    out_prog_png = OUT_FIGURES / "feature_ablation_progression.png"
    plt.savefig(out_prog_png, dpi=300)
    plt.close()
    print(f"Saved Feature Ablation Progression chart to: {out_prog_png}")

    # =========================================================================
    # PART 3: LEAVE-ONE-GROUP-OUT (LOGO) ABLATION
    # =========================================================================
    print("\n--- Part 3: Running Leave-One-Group-Out (LOGO) Ablation ---")
    full_cat = ["Primary Type", "Location Description", "Beat", "District", "Community Area", "Time_Period", "Type_Location_Interaction"]
    full_num = [
        "Hour", "Month", "DayOfWeek", "IsWeekend", "Latitude", "Longitude",
        "Hour_Sin", "Hour_Cos", "Month_Sin", "Month_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos",
        "Is_Night", "Is_Residential", "Is_Public_Space",
        "Primary_Type_Domestic_Risk", "Location_Domestic_Risk", "Beat_Domestic_Risk",
        "Beat_Historical_Incident_Volume"
    ]

    logo_configs = [
        ("Full Model (All Features)", full_num, full_cat),
        ("Drop: Bayesian Risk Priors", [c for c in full_num if "Risk" not in c and "Volume" not in c], full_cat),
        ("Drop: Cyclical Trigonometric", [c for c in full_num if "Sin" not in c and "Cos" not in c], full_cat),
        ("Drop: Venue / Shift Flags", [c for c in full_num if "Is_" not in c], [c for c in full_cat if c != "Time_Period"]),
        ("Drop: Spatial Coordinates & Beats", [c for c in full_num if c not in ["Latitude", "Longitude", "Beat_Historical_Incident_Volume"]], [c for c in full_cat if c not in ["Beat", "District", "Community Area"]]),
        ("Drop: High-Cardinality Interaction", full_num, [c for c in full_cat if c != "Type_Location_Interaction"])
    ]

    logo_results = []
    full_auc = None
    full_f1 = None

    for name, num_f, cat_f in logo_configs:
        print(f"Testing {name}...")
        transformers = []
        if cat_f:
            transformers.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cat_f))
        if num_f:
            transformers.append(("num", StandardScaler(), num_f))

        prep = ColumnTransformer(transformers=transformers)
        cols_needed = cat_f + num_f

        X_tr_enc = prep.fit_transform(tr[cols_needed])
        X_va_enc = prep.transform(va[cols_needed])
        X_te_enc = prep.transform(te[cols_needed])

        clf = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, scale_pos_weight=spw, tree_method="hist", random_state=SEED, n_jobs=-1)
        clf.fit(X_tr_enc, tr[TARGET].astype(int))

        p_va = clf.predict_proba(X_va_enc)[:, 1]
        thr = tune_threshold(va[TARGET].astype(int), p_va)
        p_te = clf.predict_proba(X_te_enc)[:, 1]

        res = evaluate_preds(te[TARGET].astype(int), p_te, thr, name)
        if full_auc is None:
            full_auc = res["ROC_AUC"]
            full_f1 = res["F1_Score"]
        res["Drop_Delta_AUC"] = round(res["ROC_AUC"] - full_auc, 4)
        res["Drop_Delta_F1"] = round(res["F1_Score"] - full_f1, 4)
        logo_results.append(res)
        print(f" -> ROC-AUC: {res['ROC_AUC']:.4f} | Drop Delta AUC: {res['Drop_Delta_AUC']:+.4f} | F1: {res['F1_Score']:.4f}")

    df_logo = pd.DataFrame(logo_results)
    out_logo_csv = OUT_TABLES / "leave_one_group_out_ablation.csv"
    df_logo.to_csv(out_logo_csv, index=False)
    print(f"\nSaved Leave-One-Group-Out table to: {out_logo_csv}")

    # Plot LOGO Impact Bar Chart
    plt.figure(figsize=(10, 5))
    df_logo_drops = df_logo.iloc[1:].copy()
    sns.barplot(data=df_logo_drops, x="Drop_Delta_AUC", y="Configuration", palette="Reds_r")
    plt.title("Performance Penalty (Delta ROC-AUC) When Dropping Feature Groups", fontsize=12, fontweight="bold")
    plt.xlabel("Change in ROC-AUC (Relative to Full 29-Feature Model)", fontsize=11)
    plt.ylabel("Ablation Group Removed", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_logo_png = OUT_FIGURES / "leave_one_group_out_impact.png"
    plt.savefig(out_logo_png, dpi=300)
    plt.close()
    print(f"Saved LOGO impact plot to: {out_logo_png}")


if __name__ == "__main__":
    main()
