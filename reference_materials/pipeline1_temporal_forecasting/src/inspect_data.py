"""Quick inspect of raw dataset properties."""
import pandas as pd
from pathlib import Path

csv_path = Path("data/raw/Crimes_-_2001_to_Present_20260902.csv")

print("Checking header and sample rows...")
df = pd.read_csv(csv_path, nrows=50000)
print("Columns:", df.columns.tolist())
print("\nUnique Years in first 50k rows:", sorted(df["Year"].dropna().unique()))
print("\nTop Primary Types in sample:\n", df["Primary Type"].value_counts().head(10))
print("\nUnique Districts in sample:", df["District"].nunique())
print("Unique Community Areas in sample:", df["Community Area"].nunique())
