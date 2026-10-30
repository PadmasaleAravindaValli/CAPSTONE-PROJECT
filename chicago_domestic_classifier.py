"""
Chicago Crime Domestic Classification
------------------------------------
Target: Domestic (True/False)
Data window: 2023-2025
Chronological split:
  Train: 2023-01-01 through 2024-06-30
  Validation: 2024-07-01 through 2024-12-31
  Test: 2025-01-01 through 2025-12-31

This script evaluates a genuine later-time test set. It does not guarantee
80% accuracy; it reports the measured test metrics.

Run:
    python chicago_domestic_classifier.py --csv "Crimes_-_2001_to_Present_20260902(1).csv"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

START_YEAR = 2023
END_YEAR = 2025
TARGET = "Domestic"
CAT_FEATURES = ["Primary Type", "Location Description", "Beat"]
NUM_FEATURES = ["Hour", "Month", "DayOfWeek", "IsWeekend"]
FEATURES = CAT_FEATURES + NUM_FEATURES
CSV_CHUNK_SIZE = 500_000


def load_chicago_data(csv_path: str, chunk_size: int = CSV_CHUNK_SIZE) -> pd.DataFrame:
    """Read only the columns and years needed, in chunks to reduce memory use."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Put the CSV in the same folder as this script or pass --csv with its path."
        )

    needed = [
        "Date", "Year", "Primary Type", "Location Description",
        "Beat", "Domestic"
    ]
    chunks = []
    print(f"Reading {csv_path} in chunks; keeping years {START_YEAR}-{END_YEAR}...")

    for chunk in pd.read_csv(csv_path, usecols=lambda c: c in needed,
                             chunksize=chunk_size, low_memory=False):
        # Use the Year column for an inexpensive first filter, then parse dates.
        year_numeric = pd.to_numeric(chunk["Year"], errors="coerce")
        chunk = chunk.loc[year_numeric.between(START_YEAR, END_YEAR)].copy()
        if chunk.empty:
            continue

        chunk["Date"] = pd.to_datetime(chunk["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        chunk = chunk.loc[chunk["Date"].notna()].copy()
        if chunk.empty:
            continue

        # Filter to the requested years using parsed timestamps too.
        chunk = chunk.loc[chunk["Date"].dt.year.between(START_YEAR, END_YEAR)].copy()
        if chunk.empty:
            continue

        chunks.append(chunk)

    if not chunks:
        raise ValueError(
            f"No valid records found for {START_YEAR}-{END_YEAR}. "
            "Check the CSV's Year and Date columns."
        )

    data = pd.concat(chunks, ignore_index=True)
    del chunks

    # Normalise target values safely.
    if data[TARGET].dtype == object:
        target_text = data[TARGET].astype(str).str.strip().str.lower()
        target_map = {"true": 1, "false": 0, "1": 1, "0": 0}
        data[TARGET] = target_text.map(target_map)
    else:
        data[TARGET] = data[TARGET].map({True: 1, False: 0, 1: 1, 0: 0})

    data = data.dropna(subset=[TARGET, "Date"]).copy()
    data[TARGET] = data[TARGET].astype("int8")

    # Derive prediction-time features from the incident date.
    data["Hour"] = data["Date"].dt.hour.astype("int8")
    data["Month"] = data["Date"].dt.month.astype("int8")
    data["DayOfWeek"] = data["Date"].dt.dayofweek.astype("int8")
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype("int8")

    for col in CAT_FEATURES:
        data[col] = data[col].fillna("Unknown").astype(str).str.strip()
        data.loc[data[col].eq(""), col] = "Unknown"

    data["Beat"] = data["Beat"].replace({"nan": np.nan, "None": np.nan})
    data["Beat"] = pd.to_numeric(data["Beat"], errors="coerce")

    # Keep only the columns used downstream.
    data = data[["Date"] + FEATURES + [TARGET]].copy()
    data = data.sort_values("Date").reset_index(drop=True)

    print(f"Rows retained: {len(data):,}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Domestic positive rate: {data[TARGET].mean():.3%}")
    return data


def chronological_split(data: pd.DataFrame):
    """Split by time; no random split is used."""
    train = data.loc[data["Date"] < "2024-07-01"].copy()
    val = data.loc[
        (data["Date"] >= "2024-07-01") & (data["Date"] < "2025-01-01")
    ].copy()
    test = data.loc[
        (data["Date"] >= "2025-01-01") & (data["Date"] < "2026-01-01")
    ].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(
            "At least one chronological split is empty. Confirm that the CSV "
            "contains records from 2023, 2024 and 2025."
        )

    print("\nChronological split")
    print(f"Train (through 2024-06-30): {len(train):,}")
    print(f"Validation (2024-07-01 to 2024-12-31): {len(val):,}")
    print(f"Test (2025): {len(test):,}")
    return train, val, test


def make_model() -> Pipeline:
    """Ordinal-encode categories and train a gradient-boosted tree classifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    )),
                ]),
                CAT_FEATURES,
            ),
            (
                "numeric",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                NUM_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def choose_threshold(y_val, p_val):
    """Choose a threshold using validation data only, not the locked test set."""
    thresholds = np.arange(0.20, 0.801, 0.01)
    best_threshold, best_score = 0.5, -1.0
    for threshold in thresholds:
        pred = (p_val >= threshold).astype(int)
        score = balanced_accuracy_score(y_val, pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    print(
        f"\nValidation-selected threshold: {best_threshold:.2f} "
        f"(balanced accuracy={best_score:.4f})"
    )
    return best_threshold


def evaluate_split(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    result = {
        "split": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_domestic": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_domestic": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_domestic": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        result["roc_auc"] = None

    print(f"\n{name} metrics")
    for key, value in result.items():
        if key != "split":
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nClassification report:")
    print(classification_report(
        y_true, predictions, labels=[0, 1],
        target_names=["Non-Domestic", "Domestic"], zero_division=0
    ))
    print("Confusion matrix (rows=actual, columns=predicted; labels 0,1):")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))
    return result, predictions


def run_experiment(csv_path: str, output_dir: str = "outputs/chicago_domestic"):
    data = load_chicago_data(csv_path)
    train, val, test = chronological_split(data)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    print("\nTraining HistGradientBoostingClassifier...")
    pipeline = make_model()
    pipeline.fit(X_train, y_train)

    p_val = pipeline.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val.to_numpy(), p_val)

    # Test is evaluated only after the threshold is chosen from validation.
    p_test = pipeline.predict_proba(X_test)[:, 1]
    test_result, test_pred = evaluate_split("TEST (2025)", y_test.to_numpy(), p_test, threshold)

    # Also show the default 0.50 threshold as a reference.
    default_result, _ = evaluate_split(
        "TEST (2025), default threshold 0.50",
        y_test.to_numpy(), p_test, 0.5
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = {
        "target": TARGET,
        "feature_columns": FEATURES,
        "categorical_features": CAT_FEATURES,
        "numeric_features": NUM_FEATURES,
        "train_period": "2023-01-01 through 2024-06-30",
        "validation_period": "2024-07-01 through 2024-12-31",
        "test_period": "2025-01-01 through 2025-12-31",
        "validation_selected_threshold": threshold,
        "test_metrics_validation_threshold": test_result,
        "test_metrics_default_threshold": default_result,
        "sklearn_version": sklearn.__version__,
        "note": (
            "Accuracy is an empirical result, not guaranteed. The target is "
            "Domestic incident classification, not NCRB five-class burden forecasting."
        ),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions = test[["Date"] + FEATURES + [TARGET]].copy()
    predictions["Domestic_Probability"] = p_test
    predictions["Predicted_Domestic"] = test_pred.astype("int8")
    predictions.to_csv(out / "test_predictions_2025.csv", index=False)

    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual_Non_Domestic", "Actual_Domestic"],
        columns=["Predicted_Non_Domestic", "Predicted_Domestic"],
    ).to_csv(out / "confusion_matrix_2025.csv")

    print(f"\nSaved metrics and predictions to: {out.resolve()}")
    print("\nLeakage note: Description, IUCR, FBI Code, ID, Case Number, Block, "
          "Updated On, coordinates, and Arrest are intentionally excluded.")
    print("Important: this predicts Domestic from incident attributes; it does not "
          "predict a district-year violent-crime burden class.")
    return pipeline, metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", default="Crimes_-_2001_to_Present_20260902(1).csv",
        help="Path to the Chicago crime CSV file."
    )
    parser.add_argument(
        "--output-dir", default="outputs/chicago_domestic",
        help="Directory where metrics and predictions will be saved."
    )
    args = parser.parse_args()
    run_experiment(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
