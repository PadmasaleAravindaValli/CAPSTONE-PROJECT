"""STEP 3c - third deep learning model: FT-Transformer (PyTorch)."""
import copy, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from data_utils import (FEATURES, SEED, RESULTS, load_splits, best_threshold, report, metrics_row, save_rows)

MIN_COUNT = 20          # categories rarer than this in TRAIN share the "unknown" id 0 (so id 0 gets trained)

torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
(Xtr, ytr), (Xva, yva), (Xte, yte) = load_splits()

# vocabularies come from TRAIN only; rare and unseen categories map to index 0
vocab = {}
for c in FEATURES:
    vc = Xtr[c].value_counts()
    keep = sorted(vc[vc >= MIN_COUNT].index)
    vocab[c] = {v: i + 1 for i, v in enumerate(keep)}

def enc(X):
    return torch.tensor(np.stack([X[c].map(vocab[c]).fillna(0).astype(int).to_numpy()
                                  for c in FEATURES], 1), dtype=torch.long).to(dev)
Ctr, Cva, Cte = enc(Xtr), enc(Xva), enc(Xte)
Ytr = torch.tensor(ytr, dtype=torch.float32).to(dev)
print(f"device={dev}  train/val/test = {len(Ctr):,}/{len(Cva):,}/{len(Cte):,}  features={FEATURES}")

class FTTransformer(nn.Module):
    """Each categorical feature -> one token; a [CLS] token is prepended; output is read from [CLS]."""
    def __init__(self, d=64, heads=4, layers=3, ff=128, p=0.1):
        super().__init__()
        sizes = [len(vocab[c]) + 1 for c in FEATURES]
        self.register_buffer("offsets", torch.tensor([0] + sizes[:-1]).cumsum(0))
        self.emb = nn.Embedding(sum(sizes), d)                       # one shared table, per-feature offsets
        self.feat_bias = nn.Parameter(torch.zeros(len(FEATURES), d))  # per-feature bias
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=p, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))
    def forward(self, c):
        x = self.emb(c + self.offsets) + self.feat_bias               # (B, F, d)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)        # (B, F+1, d)
        return self.head(self.enc(x)[:, 0]).squeeze(1)

net = FTTransformer().to(dev)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / (ytr == 1).sum(),
                                                        dtype=torch.float32).to(dev))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)

def predict(C):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(C), 8192):
            out.append(torch.sigmoid(net(C[i:i + 8192])))
    return torch.cat(out).cpu().numpy()

best_auc, best_state, bad, BS = 0, None, 0, 1024
for ep in range(30):
    net.train(); perm = torch.randperm(len(Ctr), device=dev)
    for i in range(0, len(perm), BS):
        idx = perm[i:i + BS]
        opt.zero_grad()
        loss_fn(net(Ctr[idx]), Ytr[idx]).backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    auc = roc_auc_score(yva, predict(Cva))            # early stopping + LR schedule use VALIDATION only
    sched.step(auc)
    print(f"epoch {ep+1:02d}  val_auc={auc:.4f}")
    if auc > best_auc:
        best_auc, best_state, bad = auc, copy.deepcopy(net.state_dict()), 0
    else:
        bad += 1
        if bad >= 5:
            print("early stopping"); break

net.load_state_dict(best_state)
p_va = predict(Cva)
t = best_threshold(yva, p_va)                         # threshold from validation
p_te = predict(Cte)
print("\nVALIDATION"); report("FT-Transformer (DL)", yva, p_va, t)
print("TEST      "); report("FT-Transformer (DL)", yte, p_te, t)
print("test ROC-AUC:", round(roc_auc_score(yte, p_te), 4))
save_rows([metrics_row("FT-Transformer (DL)", yte, p_te, t)])
print("Saved FT-Transformer row to", RESULTS, "- now run: python evaluate_ml.py")