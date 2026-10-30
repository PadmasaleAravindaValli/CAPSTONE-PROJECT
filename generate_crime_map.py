"""Generate Interactive Geospatial Crime Heatmap of Chicago using Folium."""
from pathlib import Path
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset1_important_features.csv"
OUT_MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
OUT_MAP_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML = OUT_MAP_DIR / "chicago_domestic_heatmap.html"

# Chicago geographic center
CHICAGO_COORDS = [41.8781, -87.6298]


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, usecols=["Latitude", "Longitude", "Domestic", "Primary Type", "Location Description", "Community Area", "split"])
    
    # Filter 2025 Test incidents with valid coordinates
    test_df = df[(df["split"] == "test") & df["Latitude"].notna() & df["Longitude"].notna()].copy()
    print(f"Total 2025 test incidents with valid coordinates: {len(test_df):,}")

    domestic_incidents = test_df[test_df["Domestic"] == 1]
    print(f"Domestic incidents in 2025: {len(domestic_incidents):,}")

    # Base Folium Map centered on Chicago
    m = folium.Map(location=CHICAGO_COORDS, zoom_start=11, tiles="CartoDB dark_matter")

    # Sample for smooth browser rendering
    heat_sample = domestic_incidents.sample(min(15_000, len(domestic_incidents)), random_state=42)
    heat_data = [[row["Latitude"], row["Longitude"]] for _, row in heat_sample.iterrows()]

    # HeatMap layer
    HeatMap(
        heat_data,
        radius=13,
        blur=18,
        max_zoom=13,
        gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
    ).add_to(m)

    # Add markers for top 5 highest domestic volume community areas
    area_summary = domestic_incidents.groupby("Community Area").agg(
        Count=("Domestic", "count"),
        Lat=("Latitude", "mean"),
        Lon=("Longitude", "mean")
    ).sort_values("Count", ascending=False).head(5)

    for area_id, row in area_summary.iterrows():
        folium.CircleMarker(
            location=[row["Lat"], row["Lon"]],
            radius=10,
            popup=f"<b>Community Area #{area_id}</b><br>Total Domestic Incidents: {row['Count']:,}",
            tooltip=f"High Risk Hotspot: Area #{area_id}",
            color="#FF3333",
            fill=True,
            fill_color="#FF3333",
            fill_opacity=0.7
        ).add_to(m)

    m.save(str(OUT_HTML))
    print(f"\n-> SUCCESS: Interactive Heatmap saved to: {OUT_HTML}")
    print("   Open this HTML file in Chrome/Edge to view the zoomable interactive map!")


if __name__ == "__main__":
    main()
