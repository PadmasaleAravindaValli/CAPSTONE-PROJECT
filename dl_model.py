"""
STEP 3 - Deep-learning model: MLP with categorical embeddings (PyTorch).

Uses data_utils.py for data, split, threshold selection (validation only) and the shared results table.
Run:  python dl_model.py
Needs: pip install torch
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from data_utils import (FEATURES, SEED, load_splits, best_threshold, report, metrics_row, save_rows)

MODEL_NAME = "MLP (embeddings)"
MIN_COUNT = 20          # categories rarer than this in TRAIN share one "unknown" id
BATCH = 2048
MAX_EPOCHS = 30
PATIENCE = 4            # stop when validation AUC has not improved for this many epochs
LR = 2e-3

torch.manual_seed(SEED); np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- data -> integer ids (vocab from TRAIN only)
(Xtr, ytr), (Xva, yva), (Xte, yte) = load_splits()
vocabs = {}
for c in FEATURES:
    vc = Xtr[c].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    vocabs[c] = {v: i + 1 for i, v in enumerate(keep)}          # 0 = unknown / rare / unseen

def encode(X):
    return torch.tensor(np.stack([X[c].map(vocabs[c]).fillna(0).astype(int).to_numpy() for c in FEATURES], axis=1),
                        dtype=torch.long)

Ttr, Tva, Tte = encode(Xtr), encode(Xva), encode(Xte)
Ytr = torch.tensor(ytr, dtype=torch.float32)
print(f"device={device}  train/val/test = {len(Ttr):,}/{len(Tva):,}/{len(Tte):,}  "
      f"train positive rate = {ytr.mean():.3f}")

# ---------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n + 1, min(50, (n + 2) // 2)) for n in sizes])
        d = sum(e.embedding_dim for e in self.embs)
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1))
    def forward(self, x):
        z = torch.cat([e(x[:, i]) for i, e in enumerate(self.embs)], dim=1)
        return self.net(z).squeeze(1)

model = MLP([len(vocabs[c]) for c in FEATURES]).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# class imbalance: weight positives; the decision threshold is tuned on validation afterwards
pos_weight = torch.tensor((1 - ytr.mean()) / ytr.mean(), dtype=torch.float32, device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

@torch.no_grad()
def predict(T):
    model.eval()
    out = [torch.sigmoid(model(T[i:i + 8192].to(device))).cpu() for i in range(0, len(T), 8192)]
    return torch.cat(out).numpy()

# ---------------------------------------------------------------- train with early stopping on validation AUC
best_auc, best_state, bad = 0.0, None, 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Ttr))
    total = 0.0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        if len(idx) < 2:
            continue                                   # BatchNorm needs more than one row
        xb, yb = Ttr[idx].to(device), Ytr[idx].to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
        total += loss.item() * len(idx)
    auc = roc_auc_score(yva, predict(Tva))
    print(f"epoch {epoch:2d}  train_loss={total / len(perm):.4f}  val_AUC={auc:.4f}")
    if auc > best_auc + 1e-4:
        best_auc, bad = auc, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= PATIENCE:
            print("early stopping"); break
model.load_state_dict(best_state)

# ---------------------------------------------------------------- threshold on VALIDATION, report on TEST
p_va, p_te = predict(Tva), predict(Tte)
thr = best_threshold(yva, p_va)
print("\nVALIDATION"); report(MODEL_NAME, yva, p_va, thr)
print("TEST      "); report(MODEL_NAME, yte, p_te, thr)

table = save_rows([metrics_row(MODEL_NAME, yte, p_te, thr)])
print("\n", table.to_string(index=False))