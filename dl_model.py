"""STEP 3 - deep learning: MLP with embeddings (PyTorch). Vocabularies, early stopping and threshold use train/val only."""
import copy, numpy as np, torch, torch.nn as nn
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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(len(vocab[c]) + 1, min(32, len(vocab[c]) // 2 + 2))
                                   for c in FEATURES])
        d = sum(e.embedding_dim for e in self.embs)
        self.mlp = nn.Sequential(nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 1))
    def forward(self, c):
        return self.mlp(torch.cat([e(c[:, i]) for i, e in enumerate(self.embs)], 1)).squeeze(1)

net = Net().to(dev)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / (ytr == 1).sum(),
                                                        dtype=torch.float32).to(dev))
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)

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
    auc = roc_auc_score(yva, predict(Cva))            # early stopping uses VALIDATION only
    print(f"epoch {ep+1:02d}  val_auc={auc:.4f}")
    if auc > best_auc: best_auc, best_state, bad = auc, copy.deepcopy(net.state_dict()), 0
    else:
        bad += 1
        if bad >= 4: break

net.load_state_dict(best_state)
t = best_threshold(yva, predict(Cva))                 # threshold from validation
p_te = predict(Cte)
report("MLP (embeddings)", yte, p_te, t)
print("test ROC-AUC:", round(roc_auc_score(yte, p_te), 4))
save_rows([metrics_row("MLP (embeddings)", yte, p_te, t)])     # adds the MLP to outputs/leakfree_results.csv
print("Saved MLP row to", RESULTS, "- now run: python evaluate_ml.py")