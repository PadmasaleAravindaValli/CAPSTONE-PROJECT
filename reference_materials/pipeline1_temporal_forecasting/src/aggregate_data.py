"""Aggregate raw Chicago crime records into Beat-Month panel with 6 crime groups and 5 enhanced features.

Matches the architecture and schema of the Major-Project repository.
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Crimes_-_2001_to_Present_20260902.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "pipeline1_temporal_forecasting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping primary types to 6 core groups (identical concept to Major-Project)
CRIME_GROUP_MAPPING = {
    "HOMICIDE": "GROUP_HOMICIDE",
    "CRIMINAL SEXUAL ASSAULT": "GROUP_SEXUAL_VIOLENCE",
    "SEX OFFENSE": "GROUP_SEXUAL_VIOLENCE",
    "HUMAN TRAFFICKING": "GROUP_SEXUAL_VIOLENCE",
    "PROSTITUTION": "GROUP_SEXUAL_VIOLENCE",
    "OBSCENITY": "GROUP_SEXUAL_VIOLENCE",
    "PUBLIC INDECENCY": "GROUP_SEXUAL_VIOLENCE",
    "KIDNAPPING": "GROUP_KIDNAPPING_ABDUCTION",
    "ROBBERY": "GROUP_ROBBERY_DACOITY",
    "BATTERY": "GROUP_PUBLIC_VIOLENCE",
    "ASSAULT": "GROUP_PUBLIC_VIOLENCE",
    "ARSON": "GROUP_PUBLIC_VIOLENCE",
    "WEAPONS VIOLATION": "GROUP_PUBLIC_VIOLENCE",
    "PUBLIC PEACE VIOLATION": "GROUP_PUBLIC_VIOLENCE",
    "INTIMIDATION": "GROUP_PUBLIC_VIOLENCE",
    "STALKING": "GROUP_PUBLIC_VIOLENCE",
    "INTERFERENCE WITH PUBLIC OFFICER": "GROUP_PUBLIC_VIOLENCE",
    "THEFT": "GROUP_PROPERTY_CRIME",
    "BURGLARY": "GROUP_PROPERTY_CRIME",
    "MOTOR VEHICLE THEFT": "GROUP_PROPERTY_CRIME",
    "CRIMINAL DAMAGE": "GROUP_PROPERTY_CRIME",
    "CRIMINAL TRESPASS": "GROUP_PROPERTY_CRIME",
    "DECEPTIVE PRACTICE": "GROUP_PROPERTY_CRIME",
}

GROUPS = [
    "GROUP_HOMICIDE",
    "GROUP_SEXUAL_VIOLENCE",
    "GROUP_KIDNAPPING_ABDUCTION",
    "GROUP_ROBBERY_DACOITY",
    "GROUP_PUBLIC_VIOLENCE",
    "GROUP_PROPERTY_CRIME",
]


def shannon_entropy(row_vals):
    vals = np.array(row_vals, dtype=float)
    total = vals.sum()
    if total <= 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def main():
    print(f"Reading raw data from {RAW_DATA_PATH}...")
    cols = ["Date", "Beat", "District", "Primary Type"]
    df = pd.read_csv(RAW_DATA_PATH, usecols=cols, low_memory=False)
    print(f"Loaded {len(df):,} incidents.")

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["Date", "Beat", "Primary Type"])
    df["Beat"] = df["Beat"].astype(int)
    df["District"] = df["District"].fillna(-1).astype(int)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Period_Str"] = df["Date"].dt.to_period("M").astype(str)

    # Filter to 2023-2025 window
    df = df[(df["Year"] >= 2023) & (df["Year"] <= 2025)]
    print(f"Incidents in 2023-2025 window: {len(df):,}")

    # Map primary type to group
    df["Crime_Group"] = df["Primary Type"].map(CRIME_GROUP_MAPPING)
    # Filter out incidents not falling into the 6 major analytical groups (e.g. general narcotics/other)
    df_mapped = df.dropna(subset=["Crime_Group"]).copy()
    print(f"Incidents in 6 core groups: {len(df_mapped):,}")

    # Generate complete grid of (Beat, Period) so no months are missing for any active beat
    all_beats = sorted(df_mapped["Beat"].unique())
    all_periods = pd.period_range(start="2023-01", end="2025-12", freq="M").astype(str)
    grid = pd.MultiIndex.from_product([all_periods, all_beats], names=["PERIOD", "UNIT_NAME"]).to_frame().reset_index(drop=True)
    grid["YEAR"] = grid["PERIOD"].str.slice(0, 4).astype(int)
    grid["MONTH"] = grid["PERIOD"].str.slice(5, 7).astype(int)
    grid["STATE"] = "ILLINOIS"

    # Aggregate counts by Period, Beat, Crime_Group
    agg = df_mapped.groupby(["Period_Str", "Beat", "Crime_Group"]).size().unstack(fill_value=0).reset_index()
    agg = agg.rename(columns={"Period_Str": "PERIOD", "Beat": "UNIT_NAME"})

    for g in GROUPS:
        if g not in agg.columns:
            agg[g] = 0

    # Merge with full grid
    merged = pd.merge(grid, agg, on=["PERIOD", "UNIT_NAME"], how="left")
    for g in GROUPS:
        merged[g] = merged[g].fillna(0).astype(float)

    # Calculate Violent Crime Burden
    # Burden = Homicide + Sexual Violence + Kidnapping + Robbery + Public Violence
    merged["VIOLENT_CRIME_BURDEN"] = (
        merged["GROUP_HOMICIDE"]
        + merged["GROUP_SEXUAL_VIOLENCE"]
        + merged["GROUP_KIDNAPPING_ABDUCTION"]
        + merged["GROUP_ROBBERY_DACOITY"]
        + merged["GROUP_PUBLIC_VIOLENCE"]
    )

    # Threshold fitting on Training period (2023 through mid-2024: 2023-01 to 2024-06)
    # Frozen boundaries like Major-Project
    train_mask = (merged["PERIOD"] >= "2023-01") & (merged["PERIOD"] <= "2024-06")
    train_burden = merged.loc[train_mask, "VIOLENT_CRIME_BURDEN"]
    cutoffs = np.quantile(train_burden, [0.2, 0.4, 0.6, 0.8])
    print(f"Fitted frozen 5-level quintile boundaries on training period: {cutoffs}")

    # Assign TARGET_VIOLENT_LEVEL (1 to 5)
    def assign_class(b):
        if b <= cutoffs[0]:
            return 1
        elif b <= cutoffs[1]:
            return 2
        elif b <= cutoffs[2]:
            return 3
        elif b <= cutoffs[3]:
            return 4
        else:
            return 5

    merged["TARGET_VIOLENT_LEVEL"] = merged["VIOLENT_CRIME_BURDEN"].apply(assign_class)

    # Save reduced grouped dataset (matches Major-Project district_grouped_reduced.csv)
    reduced_cols = [
        "PERIOD", "YEAR", "MONTH", "STATE", "UNIT_NAME",
        "GROUP_HOMICIDE", "GROUP_SEXUAL_VIOLENCE", "GROUP_KIDNAPPING_ABDUCTION",
        "GROUP_ROBBERY_DACOITY", "GROUP_PUBLIC_VIOLENCE", "GROUP_PROPERTY_CRIME",
        "VIOLENT_CRIME_BURDEN", "TARGET_VIOLENT_LEVEL"
    ]
    reduced_df = merged[reduced_cols].sort_values(["UNIT_NAME", "PERIOD"]).reset_index(drop=True)
    reduced_csv = OUT_DIR / "chicago_grouped_reduced.csv"
    reduced_df.to_csv(reduced_csv, index=False)
    print(f"Saved {reduced_csv} with {len(reduced_df):,} rows.")

    # 5 Enhanced Features
    # 1. TOTAL_GROUPED_CRIME
    reduced_df["TOTAL_GROUPED_CRIME"] = reduced_df[GROUPS].sum(axis=1)

    # 2. ACTIVE_CRIME_GROUPS
    reduced_df["ACTIVE_CRIME_GROUPS"] = (reduced_df[GROUPS] > 0).sum(axis=1)

    # 3. CRIME_DIVERSITY_INDEX (Shannon entropy)
    reduced_df["CRIME_DIVERSITY_INDEX"] = reduced_df[GROUPS].apply(shannon_entropy, axis=1)

    # 4. PREV_PERIOD_TOTAL_CRIME (lag-1 per beat)
    reduced_df["PREV_PERIOD_TOTAL_CRIME"] = reduced_df.groupby("UNIT_NAME")["TOTAL_GROUPED_CRIME"].shift(1)

    # 5. PERIOD_OVER_PERIOD_TOTAL_CRIME_CHANGE
    reduced_df["YOY_TOTAL_CRIME_CHANGE"] = (
        reduced_df["TOTAL_GROUPED_CRIME"] - reduced_df["PREV_PERIOD_TOTAL_CRIME"]
    )

    # Save enhanced dataset (matches Major-Project district_enhanced_features.csv)
    enhanced_csv = OUT_DIR / "chicago_enhanced_features.csv"
    reduced_df.to_csv(enhanced_csv, index=False)
    print(f"Saved {enhanced_csv} with {len(reduced_df):,} rows.")
    print("Target distribution:\n", reduced_df["TARGET_VIOLENT_LEVEL"].value_counts().sort_index())


if __name__ == "__main__":
    main()
