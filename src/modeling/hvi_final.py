#!/usr/bin/env python3
"""
hvi_corrected_final.py
Corrected HVI calculation using pixel index to align CSV and raster data.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import xy
from shapely.geometry import Point
import matplotlib.pyplot as plt
import os

# ================================================================
# 1. CONFIGURATION
# ================================================================

RASTER_FILE = "delhi_urban_heat.tif"
GEOJSON_FILE = "delhi_1997-2012_district.json"
CSV_FILE = "delhi_urban_heat_with_osm_buildings.csv"

OUTPUT_CSV = "delhi_heat_with_districts_pop_poverty.csv"
OUTPUT_HVI_CSV = "heat_vulnerability_index_final.csv"
OUTPUT_VIZ = "heat_vulnerability_index_final.png"

# ================================================================
# 2. HARDCODE POVERTY DATA (MPI) from NFHS-5 (2019-21)
# ================================================================

poverty_mpi = {
    'Central': 0.016,
    'East': 0.012,
    'New Delhi': 0.019,
    'North': 0.026,
    'North East': 0.015,
    'North West': 0.008,
    'South': 0.007,
    'South West': 0.015,
    'West': 0.020,
}

# ================================================================
# 3. EXTRACT PIXEL COORDINATES, LST AND INDEX
# ================================================================

print("📂 Extracting pixel coordinates and LST from raster...")
with rasterio.open(RASTER_FILE) as src:
    lst_data = src.read(1)
    transform = src.transform
    rows, cols = np.indices(lst_data.shape)
    xs, ys = xy(transform, rows, cols)
    lons = np.array(xs).flatten()
    lats = np.array(ys).flatten()
    lst_flat = lst_data.flatten()

# Create DataFrame with index (pixel ID)
df_pixels = pd.DataFrame({
    'pixel_id': np.arange(len(lst_flat)),
    'longitude': lons,
    'latitude': lats,
    'LST': lst_flat
})
print(f"✅ Total pixels: {len(df_pixels)}")

# Drop NaN pixels (where LST is missing)
df_pixels = df_pixels.dropna(subset=['LST'])
print(f"   Valid pixels: {len(df_pixels)}")

# ================================================================
# 4. LOAD GEOJSON AND ASSIGN DISTRICTS
# ================================================================

print("\n📂 Loading GeoJSON...")
gdf_districts = gpd.read_file(GEOJSON_FILE)
gdf_districts = gdf_districts.to_crs('EPSG:4326')
print(f"   {len(gdf_districts)} districts")

print("\n📍 Assigning districts to pixels...")
gdf_points = gpd.GeoDataFrame(
    df_pixels,
    geometry=gpd.points_from_xy(df_pixels.longitude, df_pixels.latitude),
    crs='EPSG:4326'
)

gdf_joined = gpd.sjoin(gdf_points, gdf_districts, how='inner', predicate='within')
print(f"✅ {len(gdf_joined)} pixels inside Delhi districts")

# Extract district and keep pixel_id
gdf_joined['district'] = gdf_joined['DISTRICT']
df_dist = gdf_joined[['pixel_id', 'longitude', 'latitude', 'LST', 'district']].copy()

# ================================================================
# 5. LOAD ORIGINAL CSV AND MERGE USING PIXEL_ID
# ================================================================

print("\n📊 Loading original CSV...")
df_csv = pd.read_csv(CSV_FILE)
print(f"   Original CSV: {len(df_csv)} rows")

# Add pixel_id to CSV (same order as raster)
df_csv['pixel_id'] = np.arange(len(df_csv))

# Merge on pixel_id
df_merged = df_dist.merge(df_csv, on='pixel_id', how='left')
print(f"✅ Merged dataset: {len(df_merged)} rows")

# ================================================================
# 6. COMPUTE DISTRICT‑LEVEL AGGREGATES
# ================================================================

print("\n🌡️ Computing district‑level aggregates...")

districts = df_merged['district'].unique().tolist()
lst_by_district = df_merged.groupby('district')['LST_x'].mean().to_dict()
pop_by_district = df_merged.groupby('district')['POPULATION'].mean().to_dict()

# Build HVI DataFrame
df_hvi = pd.DataFrame({
    'district': districts,
    'lst_avg': [lst_by_district[d] for d in districts],
    'pop_avg': [pop_by_district[d] for d in districts],
    'poverty_mpi': [poverty_mpi.get(d, np.nan) for d in districts],
})

# ================================================================
# 7. NORMALIZE COMPONENTS
# ================================================================

def minmax_norm(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return np.zeros_like(series)
    return (series - min_val) / (max_val - min_val)

df_hvi['lst_norm'] = minmax_norm(df_hvi['lst_avg'])
df_hvi['pop_norm'] = minmax_norm(df_hvi['pop_avg'])
df_hvi['poverty_norm'] = minmax_norm(df_hvi['poverty_mpi'])

# ================================================================
# 8. COMPUTE HVI
# ================================================================

df_hvi['HVI'] = 0.40 * df_hvi['lst_norm'] + 0.35 * df_hvi['pop_norm'] + 0.25 * df_hvi['poverty_norm']
df_hvi['HVI'] = minmax_norm(df_hvi['HVI'])

# Rank
df_hvi = df_hvi.sort_values('HVI', ascending=False)
df_hvi['rank'] = range(1, len(df_hvi) + 1)

# ================================================================
# 9. PRINT RESULTS
# ================================================================

print("\n🏆 Delhi: Heat Vulnerability Index (with Real Population & Poverty MPI)")
print("=" * 80)
print(df_hvi[['rank', 'district', 'HVI', 'lst_norm', 'pop_norm', 'poverty_norm']].to_string(index=False))
print("=" * 80)

# ================================================================
# 10. SAVE RESULTS
# ================================================================

df_hvi.to_csv(OUTPUT_HVI_CSV, index=False, float_format='%.4f')
print(f"\n✅ HVI table saved to: {OUTPUT_HVI_CSV}")

# Save merged dataset (for reference)
df_merged.to_csv(OUTPUT_CSV, index=False, float_format='%.4f')
print(f"✅ Merged dataset saved to: {OUTPUT_CSV}")

# ================================================================
# 11. VISUALIZATION
# ================================================================

print("\n📈 Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ---- Bar chart: HVI ----
ax1 = axes[0]
colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(df_hvi)))[::-1]
bars = ax1.barh(df_hvi['district'], df_hvi['HVI'], color=colors)
ax1.set_xlabel('Heat Vulnerability Index (HVI)', fontsize=12)
ax1.set_title('Delhi: Heat Vulnerability by District\n(Higher = More Vulnerable)', fontsize=14)
ax1.grid(True, alpha=0.3)
for bar, val in zip(bars, df_hvi['HVI']):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=9)

# ---- Stacked components ----
ax2 = axes[1]
components = df_hvi[['district', 'lst_norm', 'pop_norm', 'poverty_norm']].set_index('district')
components.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis')
ax2.set_title('Vulnerability Components (Normalized)', fontsize=14)
ax2.set_xlabel('District', fontsize=12)
ax2.set_ylabel('Normalized Score', fontsize=12)
ax2.legend(['Heat (LST)', 'Population', 'Poverty (MPI)'], loc='upper left')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_VIZ, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: {OUTPUT_VIZ}")
plt.show()

print("\n✅ Analysis complete!")
print("   All three components are based on real data:")
print("   - Heat (LST) from satellite raster")
print("   - Population from your CSV")
print("   - Poverty (MPI) from NFHS-5 (2019-21)")