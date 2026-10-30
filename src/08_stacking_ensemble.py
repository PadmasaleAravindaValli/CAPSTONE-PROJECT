"""Feature 3: Stacking Meta-Ensemble Classifier.

Combines heterogeneous base classifiers:
- Base 1: Logistic Regression
- Base 2: HistGradientBoosting
- Base 3: XGBoost
With a Meta-Learner (Logistic Regression) trained on out-of-fold cross-validation predictions.
Evaluated on the locked 2025 test set.
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

from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset2_enhanced_features.csv"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

TARGET = "Domestic"
SEED = 42


def tune_threshold(y_val, p_val):
    ts = np.linspace(0.05, 0.95, 91)
    scores = [min(accuracy_score(y_val, p_val >= t), balanced_accuracy_score(y_val, p_val >= t)) for t in ts]
    return float(ts[int(np.argmax(scores))])


def evaluate_model(y_true, p_pred, threshold, model_name):
    y_pred = (p_pred >= threshold).astype(int)
    return {
        "Model": model_name,
        "Threshold": round(threshold, 3),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Balanced_Acc": round(balanced_accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1_Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, p_pred), 4)
    }


def main():
    print(f"Loading Dataset 2 from {DATA_PATH}...")
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

    # Subsample training data for fast out-of-fold stacking cross-validation
    tr_sample = tr.sample(min(120_000, len(tr)), random_state=SEED)

    Xtr, ytr = tr_sample[features], tr_sample[TARGET].astype(int)
    Xva, yva = va[features], va[TARGET].astype(int)
    Xte, yte = te[features], te[TARGET].astype(int)

    spw = float((ytr == 0).sum() / (ytr == 1).sum())

    onehot_prep = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cat_cols)], remainder="passthrough")
    target_prep = ColumnTransformer([("te", TargetEncoder(target_type="binary", cv=3, random_state=SEED), cat_cols)], remainder="passthrough")

    base_estimators = [
        ("lr", Pipeline([("p", onehot_prep), ("m", LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED))])),
        ("hgb", Pipeline([("p", target_prep), ("m", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, l2_regularization=1.0, class_weight="balanced", random_state=SEED))])),
        ("xgb", Pipeline([("p", onehot_prep), ("m", xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, scale_pos_weight=spw, tree_method="hist", random_state=SEED, n_jobs=1))]))
    ]

    results = []

    # Evaluate individual base models first
    print("\n--- Training Individual Base Models ---")
    for name, pipe in base_estimators:
        print(f"Fitting {name}...")
        pipe.fit(Xtr, ytr)
        p_va = pipe.predict_proba(Xva)[:, 1]
        t = tune_threshold(yva, p_va)
        p_te = pipe.predict_proba(Xte)[:, 1]
        res = evaluate_model(yte, p_te, t, f"Base: {name.upper()}")
        results.append(res)
        print(f" -> [{name.upper()}] Accuracy: {res['Accuracy']:.4f} | Balanced Acc: {res['Balanced_Acc']:.4f} | F1: {res['F1_Score']:.4f} | ROC-AUC: {res['ROC_AUC']:.4f}")

    # Build and train Stacking Classifier
    print("\n--- Training Stacking Meta-Ensemble ---")
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, random_state=SEED),
        cv=3
    )
    stacking_clf.fit(Xtr, ytr)
    p_va_stack = stacking_clf.predict_proba(Xva)[:, 1]
    t_stack = tune_threshold(yva, p_va_stack)
    p_te_stack = stacking_clf.predict_proba(Xte)[:, 1]
    res_stack = evaluate_model(yte, p_te_stack, t_stack, "Stacked Meta-Ensemble (LR + HGB + XGB)")
    results.append(res_stack)
    print(f" -> [STACKED ENSEMBLE] Accuracy: {res_stack['Accuracy']:.4f} | Balanced Acc: {res_stack['Balanced_Acc']:.4f} | F1: {res_stack['F1_Score']:.4f} | ROC-AUC: {res_stack['ROC_AUC']:.4f}")

    res_df = pd.DataFrame(results)
    out_table = OUT_TABLES / "stacking_ensemble_results.csv"
    res_df.to_csv(out_table, index=False)
    print(f"\nSaved Stacking Results table to: {out_table}")

    # Plot Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=res_df, x="Model", y="Balanced_Acc", hue="Model", palette="crest", legend=False)
    plt.title("Model Stacking Meta-Ensemble vs. Individual Base Classifiers (2025 Test)", fontsize=12, fontweight="bold")
    plt.ylabel("Balanced Accuracy", fontsize=11)
    plt.xlabel("Model Configuration", fontsize=11)
    plt.ylim(0.78, 0.88)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()

    out_fig = OUT_FIGURES / "stacking_vs_individual_comparison.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved stacking comparison chart to: {out_fig}")


if __name__ == "__main__":
    main()
