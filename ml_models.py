"""STEP 2 - classical ML. Every encoder lives inside a Pipeline, so it is fitted on TRAIN only."""
import joblib, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from data_utils import *

(Xtr, ytr), (Xva, yva), (Xte, yte) = load_splits()
print(f"train/val/test = {len(Xtr):,}/{len(Xva):,}/{len(Xte):,}   positive rate (train) = {ytr.mean():.3f}")
spw = (ytr == 0).sum() / (ytr == 1).sum()

onehot = lambda: ColumnTransformer([("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                        min_frequency=30), FEATURES)])
target_enc = lambda: ColumnTransformer([("te", TargetEncoder(target_type="binary", cv=5, random_state=SEED), FEATURES)])
models = {
    "LogisticRegression": Pipeline([("p", onehot()), ("m", LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0))]),
    "RandomForest": Pipeline([("p", target_enc()), ("m", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=20, n_jobs=-1,
        class_weight="balanced_subsample", random_state=SEED))]),
    "HistGradientBoosting": Pipeline([("p", target_enc()), ("m", HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, l2_regularization=1.0, class_weight="balanced",
        early_stopping=True, validation_fraction=0.1, random_state=SEED))]),
    "XGBoost": Pipeline([("p", onehot()), ("m", XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=2.0, scale_pos_weight=spw, tree_method="hist",
        n_jobs=-1, random_state=SEED, eval_metric="logloss"))]),
}

results = {}
for name, pipe in models.items():
    pipe.fit(Xtr, ytr)
    t = best_threshold(yva, pipe.predict_proba(Xva)[:, 1])       # tuned on validation
    p_te = pipe.predict_proba(Xte)[:, 1]                          # test: report only
    a, b = report(name, yte, p_te, t)
    results[name] = dict(acc=a, bacc=b, auc=roc_auc_score(yte, p_te), thr=t, p=p_te)
    joblib.dump(pipe, f"models/{name}.joblib")

print("\nSUMMARY (test = all of 2025, never used for any decision)")
for k, r in results.items():
    print(f"{k:<22} acc={r['acc']:.4f}  bal_acc={r['bacc']:.4f}  roc_auc={r['auc']:.4f}")

best = max(results, key=lambda k: min(results[k]["acc"], results[k]["bacc"]))
pred = (results[best]["p"] >= results[best]["thr"]).astype(int)
print(f"\nBest model: {best}")
print(classification_report(yte, pred, digits=4)); print(confusion_matrix(yte, pred))

# ---- leakage sanity check: with shuffled labels a leak-free pipeline must drop to chance (AUC ~ 0.5)
rng = np.random.default_rng(SEED)
chk = Pipeline([("p", onehot()), ("m", LogisticRegression(max_iter=300))])
chk.fit(Xtr, rng.permutation(ytr))
print(f"\n[sanity] shuffled-label AUC on test = {roc_auc_score(yte, chk.predict_proba(Xte)[:, 1]):.3f} (should be ~0.5)")