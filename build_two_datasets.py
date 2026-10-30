"""Two-Stage Dataset Builder:
Stage 1: Extracts clean core important features from raw data -> dataset1_important_features.csv
Stage 2: Adds rich domain-engineered features to Dataset 1 -> dataset2_enhanced_features.csv

Leakage-Free:
- Target encodings and empirical risk rates are fitted strictly on the TRAINING split.
- Excludes leakage columns (Description containing 'DOMESTIC', row IDs, timestamps).
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "Crimes_-_2001_to_Present_20260902.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_DATASET1 = PROCESSED_DIR / "dataset1_important_features.csv"
OUT_DATASET2 = PROCESSED_DIR / "dataset2_enhanced_features.csv"

SEED = 42


def build_dataset_1():
    print("=" * 70)
    print("STAGE 1: CREATING DATASET 1 (IMPORTANT CORE FEATURES ONLY)")
    print("=" * 70)
    print(f"Reading raw data from: {RAW_CSV}...")
    
    # Read raw columns
    cols_to_read = [
        "Date", "Primary Type", "Location Description", "Beat",
        "District", "Community Area", "Latitude", "Longitude", "Domestic"
    ]
    df = pd.read_csv(RAW_CSV, usecols=cols_to_read, low_memory=False)
    print(f"Raw records loaded: {len(df):,}")

    # Parse date and restrict to 2023-2025 analysis window
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df[(df["Date"] >= "2023-01-01") & (df["Date"] < "2026-01-01")].copy()
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    print(f"Records in 2023-2025 window: {len(df):,}")

    # Drop exact duplicates
    before_dup = len(df)
    df = df.drop_duplicates(subset=["Date", "Primary Type", "Location Description", "Beat", "Domestic"])
    print(f"Duplicates removed: {before_dup - len(df):,} | Remaining: {len(df):,}")

    # Clean missing values in categorical fields
    df["Primary Type"] = df["Primary Type"].fillna("UNKNOWN").astype(str).str.strip()
    df["Location Description"] = df["Location Description"].fillna("UNKNOWN").astype(str).str.strip()
    df["Beat"] = df["Beat"].fillna(-1).astype(int)
    df["District"] = df["District"].fillna(-1).astype(int)
    df["Community Area"] = df["Community Area"].fillna(-1).astype(int)

    # Core temporal features
    df["Hour"] = df["Date"].dt.hour
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # Convert target to binary integer
    df["Domestic"] = df["Domestic"].astype(bool).astype(int)

    # Chronological Split (Train: 2023-Jun24, Val: Jul-Dec24, Test: 2025)
    df["split"] = np.select(
        [df["Date"] < "2024-07-01", df["Date"] < "2025-01-01"],
        ["train", "val"],
        "test"
    )

    # Select only the clean important core features
    dataset1_cols = [
        "Date", "Primary Type", "Location Description", "Beat", "District", "Community Area",
        "Hour", "Month", "DayOfWeek", "IsWeekend", "Latitude", "Longitude",
        "split", "Domestic"
    ]
    df1 = df[dataset1_cols].copy()
    
    df1.to_csv(OUT_DATASET1, index=False)
    print(f"-> SUCCESS: Dataset 1 saved to: {OUT_DATASET1}")
    print(f"   Shape: {df1.shape} ({df1.shape[0]:,} rows, {df1.shape[1]} columns)")
    print(f"   Columns: {list(df1.columns)}")
    print(f"   Split distribution:\n{df1['split'].value_counts().to_string()}\n")
    return df1


def build_dataset_2(df1):
    print("=" * 70)
    print("STAGE 2: CREATING DATASET 2 (DATASET 1 + EXTRA ENGINEERED FEATURES)")
    print("=" * 70)
    df2 = df1.copy()

    # 1. Cyclic Time Transformations (continuous circle for hours/months)
    print("Engineering Feature Group 1: Cyclic Time Features...")
    df2["Hour_Sin"] = np.sin(2 * np.pi * df2["Hour"] / 24.0).round(4)
    df2["Hour_Cos"] = np.cos(2 * np.pi * df2["Hour"] / 24.0).round(4)
    df2["Month_Sin"] = np.sin(2 * np.pi * (df2["Month"] - 1) / 12.0).round(4)
    df2["Month_Cos"] = np.cos(2 * np.pi * (df2["Month"] - 1) / 12.0).round(4)
    df2["DayOfWeek_Sin"] = np.sin(2 * np.pi * df2["DayOfWeek"] / 7.0).round(4)
    df2["DayOfWeek_Cos"] = np.cos(2 * np.pi * df2["DayOfWeek"] / 7.0).round(4)

    # 2. Time-of-day Indicators
    print("Engineering Feature Group 2: Temporal Shift Indicators...")
    df2["Is_Night"] = ((df2["Hour"] >= 21) | (df2["Hour"] < 6)).astype(int)
    
    def get_time_period(hour):
        if 5 <= hour < 12:
            return "MORNING"
        elif 12 <= hour < 17:
            return "AFTERNOON"
        elif 17 <= hour < 21:
            return "EVENING"
        else:
            return "NIGHT"
    df2["Time_Period"] = df2["Hour"].apply(get_time_period)

    # 3. Location Domain Indicators (Residential vs Public)
    print("Engineering Feature Group 3: Location Semantic Indicators...")
    residential_keywords = ["RESIDENCE", "APARTMENT", "HOME", "HOUSE", "YARD", "PORCH", "CONDO"]
    public_keywords = ["STREET", "SIDEWALK", "ALLEY", "PARK", "HIGHWAY", "TRANSIT", "VEHICLE"]
    
    loc_upper = df2["Location Description"].str.upper()
    df2["Is_Residential"] = loc_upper.apply(lambda x: int(any(k in x for k in residential_keywords)))
    df2["Is_Public_Space"] = loc_upper.apply(lambda x: int(any(k in x for k in public_keywords)))

    # 4. Interaction Feature: Primary Type x Location Description
    print("Engineering Feature Group 4: Interaction Features...")
    df2["Type_Location_Interaction"] = df2["Primary Type"] + " @ " + df2["Location Description"]

    # 5. Leak-Free Empirical Domestic Risk Rates (Fitted strictly on TRAIN)
    print("Engineering Feature Group 5: Empirical Historical Risk Rates (Train-fitted)...")
    train_mask = df2["split"] == "train"
    train_df = df2[train_mask]
    global_mean = float(train_df["Domestic"].mean())
    smoothing_weight = 20.0  # Bayesian m-estimate smoothing

    def compute_smoothed_rate(col_name):
        grouped = train_df.groupby(col_name)["Domestic"].agg(["count", "mean"])
        smoothed = (grouped["count"] * grouped["mean"] + smoothing_weight * global_mean) / (grouped["count"] + smoothing_weight)
        return smoothed.to_dict()

    type_risk_map = compute_smoothed_rate("Primary Type")
    loc_risk_map = compute_smoothed_rate("Location Description")
    beat_risk_map = compute_smoothed_rate("Beat")

    df2["Primary_Type_Domestic_Risk"] = df2["Primary Type"].map(type_risk_map).fillna(global_mean).round(4)
    df2["Location_Domestic_Risk"] = df2["Location Description"].map(loc_risk_map).fillna(global_mean).round(4)
    df2["Beat_Domestic_Risk"] = df2["Beat"].map(beat_risk_map).fillna(global_mean).round(4)

    # 6. Spatial Volume Indicator (Total historical incidents per Beat in training data)
    beat_volume_map = train_df["Beat"].value_counts().to_dict()
    df2["Beat_Historical_Incident_Volume"] = df2["Beat"].map(beat_volume_map).fillna(0).astype(int)

    # Reorder columns: Metadata & Target at the end
    feature_cols = [c for c in df2.columns if c not in ["Date", "split", "Domestic"]]
    final_cols = ["Date"] + feature_cols + ["split", "Domestic"]
    df2 = df2[final_cols]

    df2.to_csv(OUT_DATASET2, index=False)
    print(f"-> SUCCESS: Dataset 2 saved to: {OUT_DATASET2}")
    print(f"   Shape: {df2.shape} ({df2.shape[0]:,} rows, {df2.shape[1]} columns)")
    print(f"   Added {df2.shape[1] - df1.shape[1]} extra engineered features!")
    print(f"   All Columns: {list(df2.columns)}\n")
    return df2


if __name__ == "__main__":
    df1 = build_dataset_1()
    df2 = build_dataset_2(df1)
    print("=" * 70)
    print("ALL DATASETS CREATED SUCCESSFULLY!")
    print("=" * 70)
