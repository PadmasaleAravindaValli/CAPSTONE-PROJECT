"""STEP 3b - second deep learning model: Wide & Deep network (PyTorch)."""
import copy, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from data_utils import *

torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
(Xtr, ytr), (Xva, yva), (Xte, yte) = load_splits()

# vocabularies come from TRAIN only; unseen categories map to index 0
vocab = {c: {v: i + 1 for i, v in enumerate(sorted(Xtr[c].unique()))} for c in FEATURES}
def enc(X):
    return torch.tensor(np.stack([X[c].map(vocab[c]).fillna(0).astype(int).to_numpy()
                                  for c in FEATURES], 1), dtype=torch.long).to(dev)
Ctr, Cva, Cte = enc(Xtr), enc(Xva), enc(Xte)
Ytr = torch.tensor(ytr, dtype=torch.float32).to(dev)

class ResBlock(nn.Module):
    def __init__(self, d, p=0.2):
        super().__init__()
        self.f = nn.Sequential(nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d))
    def forward(self, x):
        return x + self.f(x)

class WideDeep(nn.Module):
    def __init__(self):
        super().__init__()
        n = [len(vocab[c]) + 1 for c in FEATURES]
        self.wide = nn.ModuleList([nn.Embedding(k, 1) for k in n])
        self.embs = nn.ModuleList([nn.Embedding(k, min(48, k // 2 + 2)) for k in n])
        d_in = sum(e.embedding_dim for e in self.embs)
        self.inp = nn.Linear(d_in, 128)
        self.blocks = nn.Sequential(ResBlock(128), ResBlock(128))
        self.out = nn.Sequential(nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, c):
        wide = sum(w(c[:, i]) for i, w in enumerate(self.wide)).squeeze(1)
        deep = torch.cat([e(c[:, i]) for i, e in enumerate(self.embs)], 1)
        return wide + self.out(self.blocks(self.inp(deep))).squeeze(1)

net = WideDeep().to(dev)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / (ytr == 1).sum(),
                                                        dtype=torch.float32).to(dev))
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)

def predict(C):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(C), 8192):
            out.append(torch.sigmoid(net(C[i:i + 8192])))
    return torch.cat(out).cpu().numpy()

best_auc, best_state, bad, BS = 0, None, 0, 2048
for ep in range(30):
    net.train(); perm = torch.randperm(len(Ctr), device=dev)
    for i in range(0, len(perm), BS):
        idx = perm[i:i + BS]
        opt.zero_grad(); loss_fn(net(Ctr[idx]), Ytr[idx]).backward(); opt.step()
    auc = roc_auc_score(yva, predict(Cva))            # early stopping + LR schedule use VALIDATION only
    sched.step(auc)
    print(f"epoch {ep+1:02d}  val_auc={auc:.4f}")
    if auc > best_auc: best_auc, best_state, bad = auc, copy.deepcopy(net.state_dict()), 0
    else:
        bad += 1
        if bad >= 4: break

net.load_state_dict(best_state)
t = best_threshold(yva, predict(Cva))                 # threshold from validation
p_te = predict(Cte)
report("Wide&Deep (DL)", yte, p_te, t)
print("test ROC-AUC:", round(roc_auc_score(yte, p_te), 4))
save_rows([metrics_row("Wide&Deep (DL)", yte, p_te, t)])
print("Saved Wide&Deep row to", RESULTS, "- now run: python evaluate_ml.py")