"""
STEP 1 - Data processing + feature selection  ->  data/processed/chicago_<target>_minimal.csv

Run:  python build_dataset.py                 (target = Domestic, default)
      python build_dataset.py --target Arrest
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="Domestic", choices=["Domestic", "Arrest"])
TARGET = ap.parse_args().target
OTHER_LABEL = "Arrest" if TARGET == "Domestic" else "Domestic"

ROOT = Path(__file__).parent
RAW = ROOT / "data/raw/Crimes_-_2001_to_Present_20260902.csv"
OUT = ROOT / f"data/processed/chicago_{TARGET.lower()}_minimal.csv"
REPORTS = ROOT / "reports"
SEED = 42

# FIX 2: make sure output folders exist (to_csv does not create them)
REPORTS.mkdir(exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. load, parse, restrict dates, sort
df = pd.read_csv(RAW, low_memory=False)
print("raw rows:", len(df))
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p")

# FIX 1: keep only 2023-2025 so train/val/test match the "2023-Jun24 / Jul-Dec24 / 2025" design
# (drop the upper bound if you also want 2026 rows in the test set)
df = df[(df["Date"] >= "2023-01-01") & (df["Date"] < "2026-01-01")]
df = df.sort_values("Date", kind="stable").reset_index(drop=True)
print("rows in 2023-2025 window:", len(df))

# ---------------------------------------------------------------- 2. leakage audit
desc_dom = df["Description"].str.contains("DOMESTIC", na=False)
print(f"\n[audit] 'Description' contains the word DOMESTIC in {desc_dom.sum():,} rows; "
      f"Domestic rate in those rows = {df.loc[desc_dom,'Domestic'].mean():.3f}")
LEAKAGE_AUDIT = {
    "ID / Case Number": "row identifiers, no predictive meaning",
    "Updated On": "record-maintenance timestamp written AFTER the incident (future info)",
    "Block": "~35k unique street strings: identifier-like, lets models memorise places",
    "IUCR / Description / FBI Code": "finer-grained re-coding of the crime; Description literally contains "
                                     "'DOMESTIC ...' -> direct target leak for Domestic",
    "X/Y Coordinate, Location": "duplicates of Latitude/Longitude",
    "Year": "not a usable predictor for future years (extrapolation) - excluded",
    OTHER_LABEL: f"another outcome label - must not be a feature when predicting {TARGET}",
}
print("[audit] excluded columns:")
for k, v in LEAKAGE_AUDIT.items():          # FIX 7: plain loop instead of list comprehension
    print(f"   - {k}: {v}")

# ---------------------------------------------------------------- 3. cleaning (no row loss except exact duplicates)
before = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c not in ("ID", "Case Number", "Updated On")])
print(f"\nexact duplicate records removed: {before-len(df):,}  -> rows kept: {len(df):,}")

df["Hour"] = df.Date.dt.hour
df["Month"] = df.Date.dt.month
df["DayOfWeek"] = df.Date.dt.dayofweek
df["IsWeekend"] = (df.DayOfWeek >= 5).astype(int)
df["Location Description"] = df["Location Description"].fillna("UNKNOWN")
df["Primary Type"] = df["Primary Type"].fillna("UNKNOWN")
df["LatLon_Missing"] = df.Latitude.isna().astype(int)
for c in ["Ward", "Community Area"]:
    df[c] = df[c].astype("Int64")

# ---------------------------------------------------------------- 4. chronological split
df["split"] = np.select([df.Date < "2024-07-01", df.Date < "2025-01-01"], ["train", "val"], "test")
print("\nsplit sizes:\n", df.split.value_counts().to_string())
print(f"positive rate per split ({TARGET}):\n", df.groupby("split")[TARGET].mean().round(4).to_string())

# ---------------------------------------------------------------- 5. feature selection (train/val only)
CAT = ["Primary Type", "Location Description", "District", "Community Area", "Ward"]
NUM = ["Hour", "Month", "DayOfWeek", "IsWeekend", "Beat", "Latitude", "Longitude", "LatLon_Missing"]
CANDIDATES = CAT + NUM
tr, va = df[df.split == "train"], df[df.split == "val"]

# 5a. filter step: mutual information with the target, TRAIN only
# FIX 4: run on a sample (continuous features use a slow nearest-neighbour estimator)
tr_mi = tr.sample(min(200_000, len(tr)), random_state=SEED)
Xmi = tr_mi[CANDIDATES].copy()
for c in CAT:
    Xmi[c] = Xmi[c].astype(str).astype("category").cat.codes
mi = pd.Series(mutual_info_classif(Xmi.fillna(-1), tr_mi[TARGET],
                                   discrete_features=[c in CAT for c in CANDIDATES],
                                   random_state=SEED), index=CANDIDATES).sort_values(ascending=False)
print("\n[selection] mutual information (train sample):\n", mi.round(4).to_string())

# 5b. wrapper step: greedy forward selection, scored by validation ROC-AUC
def make(cols):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    cc = [c for c in cols if c in CAT]
    def prep(d, fit=False):
        X = d[cols].copy()                  # FIX 3: removed the no-op astype(float, errors="ignore")
        for c in cc:
            X[c] = X[c].astype(str)
        if cc:
            X[cc] = enc.fit_transform(X[cc]) if fit else enc.transform(X[cc])
        return X.astype(float)
    mask = [c in cc for c in cols]
    return prep, mask

sub = tr.sample(min(150_000, len(tr)), random_state=SEED)   # subsample only to speed up the search
def score(cols):
    prep, mask = make(cols)
    m = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.12, class_weight="balanced",
                                       categorical_features=mask if any(mask) else None, random_state=SEED)
    m.fit(prep(sub, True), sub[TARGET])
    return roc_auc_score(va[TARGET], m.predict_proba(prep(va))[:, 1])

chosen, remaining, history, best = [], CANDIDATES.copy(), [], 0.5
while remaining:
    s = {c: score(chosen + [c]) for c in remaining}
    c, auc = max(s.items(), key=lambda kv: kv[1])
    if auc - best < 0.003:
        break                               # stop when a new feature adds < 0.003 AUC
    chosen.append(c); remaining.remove(c); best = auc
    history.append({"step": len(chosen), "added": c, "val_auc": round(auc, 4)})
    print(f"   + {c:<22} val AUC = {auc:.4f}")
sel = pd.DataFrame(history)
sel.to_csv(REPORTS / f"feature_selection_{TARGET.lower()}.csv", index=False)
mi.rename("mutual_info").to_csv(REPORTS / f"mutual_info_{TARGET.lower()}.csv")
print("\nFINAL minimal feature set:", chosen)

# ---------------------------------------------------------------- 6. save
out = df[chosen + [TARGET, "split"]].copy()
out[TARGET] = out[TARGET].astype(int)
out.to_csv(OUT, index=False)
print(f"\nsaved {OUT}  shape={out.shape}")