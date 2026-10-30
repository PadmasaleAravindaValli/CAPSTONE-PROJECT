"""Feature 2: Probability Calibration and Reliability Diagram (Platt Scaling vs Isotonic Regression).

Evaluates probability calibration on the locked 2025 test set:
- Calculates Expected Calibration Error (ECE)
- Calculates Brier Score and Log Loss
- Plots Reliability Diagrams (Calibration Curves) against ideal 45-degree diagonal.
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

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None
from sklearn.metrics import brier_score_loss, log_loss
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset2_enhanced_features.csv"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

TARGET = "Domestic"
SEED = 42


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_count = mask.sum()
        if bin_count > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += (bin_count / len(y_true)) * np.abs(bin_acc - bin_conf)
    return ece


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

    ohe = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cat_cols)], remainder="passthrough")
    Xtr_enc = ohe.fit_transform(Xtr)
    Xva_enc = ohe.transform(Xva)
    Xte_enc = ohe.transform(Xte)

    # 1. Base Uncalibrated Model
    print("1. Training Base XGBoost model...")
    base_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, scale_pos_weight=spw, tree_method="hist", random_state=SEED, n_jobs=1)
    base_model.fit(Xtr_enc, ytr)
    p_uncal = base_model.predict_proba(Xte_enc)[:, 1]

    # 2. Platt Scaling (Sigmoid) Calibrated Model on Validation Set
    print("2. Calibrating with Platt Scaling (Sigmoid) on Validation Set...")
    estimator_sigmoid = FrozenEstimator(base_model) if FrozenEstimator is not None else base_model
    cal_sigmoid = CalibratedClassifierCV(estimator=estimator_sigmoid, method="sigmoid")
    cal_sigmoid.fit(Xva_enc, yva)
    p_sigmoid = cal_sigmoid.predict_proba(Xte_enc)[:, 1]

    # 3. Isotonic Regression Calibrated Model on Validation Set
    print("3. Calibrating with Isotonic Regression on Validation Set...")
    estimator_isotonic = FrozenEstimator(base_model) if FrozenEstimator is not None else base_model
    cal_isotonic = CalibratedClassifierCV(estimator=estimator_isotonic, method="isotonic")
    cal_isotonic.fit(Xva_enc, yva)
    p_isotonic = cal_isotonic.predict_proba(Xte_enc)[:, 1]

    # Compute Metrics
    methods = {
        "Uncalibrated Base Model": p_uncal,
        "Platt Scaling (Sigmoid)": p_sigmoid,
        "Isotonic Regression": p_isotonic
    }

    metrics = []
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal Diagonal)")

    colors = {"Uncalibrated Base Model": "red", "Platt Scaling (Sigmoid)": "blue", "Isotonic Regression": "green"}

    for name, p in methods.items():
        brier = brier_score_loss(yte, p)
        loss = log_loss(yte, p)
        ece = expected_calibration_error(yte.to_numpy(), p)
        metrics.append({
            "Method": name,
            "Brier_Score": round(brier, 4),
            "Expected_Calibration_Error_ECE": round(ece, 4),
            "Log_Loss": round(loss, 4)
        })
        print(f" -> [{name}] Brier Score: {brier:.4f} | ECE: {ece:.4f} | Log Loss: {loss:.4f}")

        # Calibration curve
        prob_true, prob_pred = calibration_curve(yte, p, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color=colors[name], label=f"{name} (ECE={ece:.3f})")

    metrics_df = pd.DataFrame(metrics)
    out_table = OUT_TABLES / "calibration_metrics.csv"
    metrics_df.to_csv(out_table, index=False)
    print(f"\nSaved Calibration Metrics table to: {out_table}")

    plt.title("Reliability Diagram: Model Probability Calibration (2025 Test Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Predicted Domestic Probability", fontsize=11)
    plt.ylabel("Observed Fraction of Domestic Incidents", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_fig = OUT_FIGURES / "model_calibration_curves.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved calibration curve plot to: {out_fig}")


if __name__ == "__main__":
    main()
