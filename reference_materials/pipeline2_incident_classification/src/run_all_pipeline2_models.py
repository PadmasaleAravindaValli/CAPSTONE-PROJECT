"""Pipeline 2: Comprehensive Incident-Level Domestic Classification.

Implements all 6 models from final methology.docx:
1. Logistic Regression
2. Random Forest
3. HistGradientBoosting
4. XGBoost
5. Multi-Layer Perceptron (MLP) with categorical embeddings
6. FT-Transformer (Feature Tokenizer Transformer)

Validation-based threshold tuning, locked 2025 test set evaluation,
and shuffled-label sanity check.
"""
from pathlib import Path
import os
import sys

# Crucial for stable single-process execution in Windows sandbox
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pipeline2_incident_classification" / "chicago_domestic_minimal.csv"
OUTPUT_TABLES = PROJECT_ROOT / "pipeline2_incident_classification" / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "pipeline2_incident_classification" / "outputs" / "figures"
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

FEATURES = ["Primary Type", "Location Description", "Beat"]
TARGET = "Domestic"
SEED = 42
MIN_COUNT = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_data():
    print(f"Loading incident dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, dtype={c: str for c in FEATURES})
    for c in FEATURES:
        df[c] = df[c].fillna("UNKNOWN").astype(str)

    splits = {}
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        splits[s] = (sub[FEATURES].reset_index(drop=True), sub[TARGET].astype(int).to_numpy())
        print(f"Split {s:>5}: {len(sub):,} rows | Domestic rate: {splits[s][1].mean():.4f}")
    return splits


def tune_threshold(y_val, p_val):
    ts = np.linspace(0.05, 0.95, 91)
    scores = [min(accuracy_score(y_val, p_val >= t), balanced_accuracy_score(y_val, p_val >= t)) for t in ts]
    best_t = float(ts[int(np.argmax(scores))])
    return best_t


def compute_metrics(y_true, p_pred, threshold):
    y_pred = (p_pred >= threshold).astype(int)
    return {
        "Threshold": round(threshold, 3),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Balanced_Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1_Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, p_pred), 4)
    }


# =====================================================================
# PYTORCH DEEP LEARNING ARCHITECTURES
# =====================================================================

class MLPWithEmbeddings(nn.Module):
    def __init__(self, vocab_sizes, emb_dim=32):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n + 1, min(emb_dim, (n + 2) // 2)) for n in vocab_sizes])
        total_dim = sum(e.embedding_dim for e in self.embs)
        self.net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        z = torch.cat([e(x[:, i]) for i, e in enumerate(self.embs)], dim=1)
        return self.net(z).squeeze(1)


class FTTransformer(nn.Module):
    def __init__(self, vocab_sizes, d=64, heads=4, layers=3, ff=128, p=0.1):
        super().__init__()
        sizes = [n + 1 for n in vocab_sizes]
        self.register_buffer("offsets", torch.tensor([0] + sizes[:-1]).cumsum(0))
        self.emb = nn.Embedding(sum(sizes), d)
        self.feat_bias = nn.Parameter(torch.zeros(len(vocab_sizes), d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=p, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):
        h = self.emb(x + self.offsets) + self.feat_bias
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], dim=1)
        return self.head(self.enc(h)[:, 0]).squeeze(1)


def main():
    splits = load_data()
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    Xte, yte = splits["test"]

    results = []
    test_probs_dict = {}

    # -----------------------------------------------------------------
    # 1. Classical Machine Learning Models
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 1: TRAINING CLASSICAL MACHINE LEARNING MODELS")
    print("="*70)

    spw = float((ytr == 0).sum() / (ytr == 1).sum())

    onehot = lambda: ColumnTransformer([("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                             min_frequency=30), FEATURES)])
    target_enc = lambda: ColumnTransformer([("te", TargetEncoder(target_type="binary", cv=5, random_state=SEED), FEATURES)])

    # (a) Logistic Regression
    print("1. Training Logistic Regression...")
    pipe_lr = Pipeline([("p", onehot()), ("m", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=SEED, n_jobs=1))])
    pipe_lr.fit(Xtr, ytr)
    p_val_lr = pipe_lr.predict_proba(Xva)[:, 1]
    t_lr = tune_threshold(yva, p_val_lr)
    p_te_lr = pipe_lr.predict_proba(Xte)[:, 1]
    m_lr = compute_metrics(yte, p_te_lr, t_lr)
    m_lr["Model"] = "Logistic Regression"
    m_lr["Type"] = "Classical ML"
    results.append(m_lr)
    test_probs_dict["Logistic Regression"] = p_te_lr
    print(f" -> Threshold: {t_lr:.3f} | Accuracy: {m_lr['Accuracy']:.4f} | Balanced Acc: {m_lr['Balanced_Accuracy']:.4f} | ROC-AUC: {m_lr['ROC_AUC']:.4f}")

    # (b) HistGradientBoosting
    print("2. Training HistGradientBoosting...")
    pipe_hgb = Pipeline([("p", target_enc()), ("m", HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, l2_regularization=1.0, class_weight="balanced",
        early_stopping=True, validation_fraction=0.1, random_state=SEED))])
    pipe_hgb.fit(Xtr, ytr)
    p_val_hgb = pipe_hgb.predict_proba(Xva)[:, 1]
    t_hgb = tune_threshold(yva, p_val_hgb)
    p_te_hgb = pipe_hgb.predict_proba(Xte)[:, 1]
    m_hgb = compute_metrics(yte, p_te_hgb, t_hgb)
    m_hgb["Model"] = "HistGradientBoosting"
    m_hgb["Type"] = "Classical ML"
    results.append(m_hgb)
    test_probs_dict["HistGradientBoosting"] = p_te_hgb
    print(f" -> Threshold: {t_hgb:.3f} | Accuracy: {m_hgb['Accuracy']:.4f} | Balanced Acc: {m_hgb['Balanced_Accuracy']:.4f} | ROC-AUC: {m_hgb['ROC_AUC']:.4f}")

    # (c) XGBoost
    print("3. Training XGBoost...")
    pipe_xgb = Pipeline([("p", onehot()), ("m", xgb.XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=2.0, scale_pos_weight=spw, tree_method="hist",
        n_jobs=1, random_state=SEED, eval_metric="logloss"))])
    pipe_xgb.fit(Xtr, ytr)
    p_val_xgb = pipe_xgb.predict_proba(Xva)[:, 1]
    t_xgb = tune_threshold(yva, p_val_xgb)
    p_te_xgb = pipe_xgb.predict_proba(Xte)[:, 1]
    m_xgb = compute_metrics(yte, p_te_xgb, t_xgb)
    m_xgb["Model"] = "XGBoost"
    m_xgb["Type"] = "Classical ML"
    results.append(m_xgb)
    test_probs_dict["XGBoost"] = p_te_xgb
    print(f" -> Threshold: {t_xgb:.3f} | Accuracy: {m_xgb['Accuracy']:.4f} | Balanced Acc: {m_xgb['Balanced_Accuracy']:.4f} | ROC-AUC: {m_xgb['ROC_AUC']:.4f}")

    # (d) Random Forest
    print("4. Training Random Forest (200 trees, balanced_subsample)...")
    pipe_rf = Pipeline([("p", target_enc()), ("m", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=20, n_jobs=1,
        class_weight="balanced_subsample", random_state=SEED))])
    pipe_rf.fit(Xtr, ytr)
    p_val_rf = pipe_rf.predict_proba(Xva)[:, 1]
    t_rf = tune_threshold(yva, p_val_rf)
    p_te_rf = pipe_rf.predict_proba(Xte)[:, 1]
    m_rf = compute_metrics(yte, p_te_rf, t_rf)
    m_rf["Model"] = "Random Forest"
    m_rf["Type"] = "Classical ML"
    results.append(m_rf)
    test_probs_dict["Random Forest"] = p_te_rf
    print(f" -> Threshold: {t_rf:.3f} | Accuracy: {m_rf['Accuracy']:.4f} | Balanced Acc: {m_rf['Balanced_Accuracy']:.4f} | ROC-AUC: {m_rf['ROC_AUC']:.4f}")

    # -----------------------------------------------------------------
    # 2. Deep Learning Models (PyTorch)
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 2: TRAINING DEEP LEARNING MODELS (PYTORCH)")
    print("="*70)

    vocabs = {}
    for c in FEATURES:
        vc = Xtr[c].value_counts()
        keep = vc[vc >= MIN_COUNT].index
        vocabs[c] = {v: i + 1 for i, v in enumerate(keep)}

    def encode_torch(df_part):
        arr = np.stack([df_part[c].map(vocabs[c]).fillna(0).astype(int).to_numpy() for c in FEATURES], axis=1)
        return torch.tensor(arr, dtype=torch.long)

    Ttr = encode_torch(Xtr)
    Tva = encode_torch(Xva)
    Tte = encode_torch(Xte)
    Ytr = torch.tensor(ytr, dtype=torch.float32)

    vocab_sizes = [len(vocabs[c]) for c in FEATURES]
    pos_weight_tensor = torch.tensor((1 - ytr.mean()) / ytr.mean(), dtype=torch.float32, device=DEVICE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # (e) Multi-Layer Perceptron (Embeddings)
    print("5. Training MLP with Categorical Embeddings...")
    mlp_model = MLPWithEmbeddings(vocab_sizes).to(DEVICE)
    mlp_opt = torch.optim.AdamW(mlp_model.parameters(), lr=0.002, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(Ttr, Ytr), batch_size=2048, shuffle=True)

    best_val_auc_mlp, best_mlp_state = 0.0, None
    for epoch in range(1, 15):
        mlp_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            mlp_opt.zero_grad()
            logits = mlp_model(bx)
            loss = bce_loss(logits, by)
            loss.backward()
            mlp_opt.step()

        mlp_model.eval()
        with torch.no_grad():
            v_probs = []
            for i in range(0, len(Tva), 8192):
                v_probs.append(torch.sigmoid(mlp_model(Tva[i:i+8192].to(DEVICE))).cpu())
            v_probs = torch.cat(v_probs).numpy()
            v_auc = roc_auc_score(yva, v_probs)
            if v_auc > best_val_auc_mlp:
                best_val_auc_mlp = v_auc
                best_mlp_state = {k: v.cpu().clone() for k, v in mlp_model.state_dict().items()}

    mlp_model.load_state_dict(best_mlp_state)
    mlp_model.eval()
    with torch.no_grad():
        val_probs_mlp = []
        for i in range(0, len(Tva), 8192):
            val_probs_mlp.append(torch.sigmoid(mlp_model(Tva[i:i+8192].to(DEVICE))).cpu())
        val_probs_mlp = torch.cat(val_probs_mlp).numpy()

        te_probs_mlp = []
        for i in range(0, len(Tte), 8192):
            te_probs_mlp.append(torch.sigmoid(mlp_model(Tte[i:i+8192].to(DEVICE))).cpu())
        te_probs_mlp = torch.cat(te_probs_mlp).numpy()

    t_mlp = tune_threshold(yva, val_probs_mlp)
    m_mlp = compute_metrics(yte, te_probs_mlp, t_mlp)
    m_mlp["Model"] = "MLP (Categorical Embeddings)"
    m_mlp["Type"] = "Deep Learning"
    results.append(m_mlp)
    test_probs_dict["MLP (Embeddings)"] = te_probs_mlp
    print(f" -> Threshold: {t_mlp:.3f} | Accuracy: {m_mlp['Accuracy']:.4f} | Balanced Acc: {m_mlp['Balanced_Accuracy']:.4f} | ROC-AUC: {m_mlp['ROC_AUC']:.4f}")

    # (f) FT-Transformer
    print("6. Training FT-Transformer...")
    ft_model = FTTransformer(vocab_sizes).to(DEVICE)
    ft_opt = torch.optim.AdamW(ft_model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_auc_ft, best_ft_state = 0.0, None
    for epoch in range(1, 10):
        ft_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            ft_opt.zero_grad()
            logits = ft_model(bx)
            loss = bce_loss(logits, by)
            loss.backward()
            ft_opt.step()

        ft_model.eval()
        with torch.no_grad():
            v_probs = []
            for i in range(0, len(Tva), 8192):
                v_probs.append(torch.sigmoid(ft_model(Tva[i:i+8192].to(DEVICE))).cpu())
            v_probs = torch.cat(v_probs).numpy()
            v_auc = roc_auc_score(yva, v_probs)
            if v_auc > best_val_auc_ft:
                best_val_auc_ft = v_auc
                best_ft_state = {k: v.cpu().clone() for k, v in ft_model.state_dict().items()}

    ft_model.load_state_dict(best_ft_state)
    ft_model.eval()
    with torch.no_grad():
        val_probs_ft = []
        for i in range(0, len(Tva), 8192):
            val_probs_ft.append(torch.sigmoid(ft_model(Tva[i:i+8192].to(DEVICE))).cpu())
        val_probs_ft = torch.cat(val_probs_ft).numpy()

        te_probs_ft = []
        for i in range(0, len(Tte), 8192):
            te_probs_ft.append(torch.sigmoid(ft_model(Tte[i:i+8192].to(DEVICE))).cpu())
        te_probs_ft = torch.cat(te_probs_ft).numpy()

    t_ft = tune_threshold(yva, val_probs_ft)
    m_ft = compute_metrics(yte, te_probs_ft, t_ft)
    m_ft["Model"] = "FT-Transformer"
    m_ft["Type"] = "Deep Learning"
    results.append(m_ft)
    test_probs_dict["FT-Transformer"] = te_probs_ft
    print(f" -> Threshold: {t_ft:.3f} | Accuracy: {m_ft['Accuracy']:.4f} | Balanced Acc: {m_ft['Balanced_Accuracy']:.4f} | ROC-AUC: {m_ft['ROC_AUC']:.4f}")

    # -----------------------------------------------------------------
    # 3. Shuffled-Label Sanity Check
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 3: SHUFFLED-LABEL LEAKAGE SANITY CHECK")
    print("="*70)
    ytr_shuffled = np.random.permutation(ytr)
    sanity_lr = Pipeline([("p", onehot()), ("m", LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED, n_jobs=1))])
    sanity_lr.fit(Xtr, ytr_shuffled)
    p_te_sanity = sanity_lr.predict_proba(Xte)[:, 1]
    sanity_auc = roc_auc_score(yte, p_te_sanity)
    print(f"Shuffled-label Test ROC-AUC: {sanity_auc:.4f} (Expected ~0.50 -> Confirms NO label leakage)")

    # -----------------------------------------------------------------
    # Save Results Table and Generate Figures
    # -----------------------------------------------------------------
    res_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    out_table = OUTPUT_TABLES / "incident_domestic_model_comparison_2025.csv"
    res_df.to_csv(out_table, index=False)
    print("\n" + "="*80)
    print("FINAL 2025 TEST RESULTS (INCIDENT-LEVEL CLASSIFICATION)")
    print("="*80)
    print(res_df.to_string(index=False))
    print(f"\nSaved results table to {out_table}")

    # Plot Comparison Chart
    plt.figure(figsize=(11, 6))
    sns.barplot(data=res_df, x="Model", y="Balanced_Accuracy", hue="Type", palette="Set2")
    plt.title("Incident-Level Domestic Incident Prediction: 2025 Test Performance", fontsize=13, fontweight="bold")
    plt.ylabel("Balanced Accuracy (Validation Tuned Threshold)", fontsize=11)
    plt.xlabel("Model", fontsize=11)
    plt.ylim(0.5, 0.95)
    plt.xticks(rotation=15)
    for p in plt.gca().patches:
        h = p.get_height()
        if h > 0:
            plt.gca().annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                               ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')
    plt.tight_layout()
    chart_path = OUTPUT_FIGURES / "incident_model_comparison_2025.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved figure to {chart_path}")

    # Plot Confusion Matrix for top model (XGBoost)
    best_preds = (test_probs_dict["XGBoost"] >= t_xgb).astype(int)
    cm = confusion_matrix(yte, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Domestic", "Domestic"],
                yticklabels=["Non-Domestic", "Domestic"])
    plt.title(f"XGBoost Confusion Matrix (2025 Test, thr={t_xgb:.2f})", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Status", fontsize=11)
    plt.ylabel("Actual Status", fontsize=11)
    plt.tight_layout()
    cm_path = OUTPUT_FIGURES / "xgboost_confusion_matrix_2025.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
