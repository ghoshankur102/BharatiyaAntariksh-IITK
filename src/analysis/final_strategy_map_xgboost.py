#!/usr/bin/env python3
"""
hackathon_submission_fixed.py
Uses the merged CSV (with coordinates and district) to generate priority map.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# ================================================================
# 1. CONFIGURATION – Use the merged CSV (has coordinates and district)
# ================================================================

PIXEL_CSV = "delhi_heat_with_districts_pop_poverty.csv"   # This has coordinates + district
MODEL_FILE = "urban_heat_model_final.pkl"
HVI_CSV = "heat_vulnerability_index_final.csv"

OUTPUT_PREDICTIONS = "lst_predictions_with_priority.csv"
OUTPUT_PRIORITY_MAP = "delhi_priority_strategy_map.png"

# ================================================================
# 2. LOAD DATA
# ================================================================

print("📊 Loading merged pixel data (with coordinates)...")
df = pd.read_csv(PIXEL_CSV)
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# ================================================================
# 3. LOAD MODEL
# ================================================================

print("📂 Loading XGBoost model...")
model = joblib.load(MODEL_FILE)

# ================================================================
# 4. FEATURE ENGINEERING (exact step17)
# ================================================================

print("🔧 Engineering features...")

# Ensure building density exists
if 'BUILDING_DENSITY_OSM' not in df.columns:
    if 'BUILDING_DENSITY' in df.columns:
        df['BUILDING_DENSITY_OSM'] = df['BUILDING_DENSITY']
    else:
        df['BUILDING_DENSITY_OSM'] = df['NDBI'] * 10

# Create engineered features (exact formulas)
df['NDBI_POP'] = df['NDBI'] * df['POPULATION'] / 10000
df['NDBI_LIGHTS'] = df['NDBI'] * df['NIGHTLIGHTS']
df['URBAN_VEG'] = df['NDBI'] - df['VEG_QUALITY']
df['COOLING_EFFECT'] = df['VEG_QUALITY'] * (1 - df['NDBI'])

feature_cols = [
    'NDBI', 'POPULATION', 'NIGHTLIGHTS', 'VEG_QUALITY', 'BUILDING_DENSITY_OSM',
    'NDBI_POP', 'NDBI_LIGHTS', 'URBAN_VEG', 'COOLING_EFFECT'
]

X = df[feature_cols].copy()

# ================================================================
# 5. PREDICT LST
# ================================================================

print("🌡️ Predicting LST...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_pred_lst = model.predict(X_scaled)

df['predicted_LST'] = y_pred_lst
print(f"   LST range: {y_pred_lst.min():.1f}°C – {y_pred_lst.max():.1f}°C")

# ================================================================
# 6. ASSIGN STRATEGIES FROM PREDICTED LST
# ================================================================

print("📋 Assigning cooling strategies...")

def strategy_from_lst(lst):
    if lst >= 45:   return 'Cool Roofs'
    elif lst >= 42: return 'Blue-Green Infrastructure'
    elif lst >= 38: return 'Tree Corridors'
    elif lst >= 35: return 'Perimeter Shading'
    else:           return 'Maintain Green'

df['strategy'] = df['predicted_LST'].apply(strategy_from_lst)

print(f"   Strategy distribution:\n{df['strategy'].value_counts()}")

# ================================================================
# 7. OVERLAY HVI (if not already present)
# ================================================================

print("📊 Overlaying HVI...")

if 'HVI' not in df.columns:
    df_hvi = pd.read_csv(HVI_CSV)
    df = df.merge(df_hvi[['district', 'HVI']], on='district', how='left')
else:
    print("   HVI already present in data.")

df['HVI'] = df['HVI'].fillna(0.0)

# ================================================================
# 8. FINAL PRIORITY ASSIGNMENT
# ================================================================

def assign_priority(row):
    hvi = row['HVI']
    strategy = row['strategy']
    
    if hvi > 0.7 and strategy in ['Cool Roofs', 'Blue-Green Infrastructure']:
        return '🚨 IMMEDIATE'
    elif hvi > 0.6 and strategy == 'Maintain Green':
        return '🛡️ EQUITY FOCUS'
    elif hvi > 0.5:
        return '🔴 HIGH PRIORITY'
    else:
        return '🟢 STANDARD'

df['priority'] = df.apply(assign_priority, axis=1)

print(f"\n   Priority distribution:\n{df['priority'].value_counts()}")

# ================================================================
# 9. SAVE CSV
# ================================================================

df.to_csv(OUTPUT_PREDICTIONS, index=False)
print(f"✅ Predictions saved to: {OUTPUT_PREDICTIONS}")

# ================================================================
# 10. GENERATE PRIORITY MAP (coordinates are available)
# ================================================================

print("🗺️ Generating priority map...")

# Create GeoDataFrame from longitude/latitude
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs='EPSG:4326'
)

# Sample for speed if too many points
sample_size = min(10000, len(gdf))
gdf_sample = gdf.sample(sample_size, random_state=42)

# Define colour map
color_map = {
    '🚨 IMMEDIATE': 'darkred',
    '🔴 HIGH PRIORITY': 'red',
    '🛡️ EQUITY FOCUS': 'orange',
    '🟢 STANDARD': 'green'
}

fig, ax = plt.subplots(1, 1, figsize=(14, 12))

for priority, color in color_map.items():
    subset = gdf_sample[gdf_sample['priority'] == priority]
    if len(subset) > 0:
        ax.scatter(subset.geometry.x, subset.geometry.y,
                   s=1, c=color, label=priority, alpha=0.6)

ax.set_title('Delhi: Urban Cooling Priority Map\n'
             'LST Prediction + HVI Overlay', fontsize=18, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(markerscale=20, fontsize=12, loc='lower left')
plt.tight_layout()
plt.savefig(OUTPUT_PRIORITY_MAP, dpi=300, bbox_inches='tight')
print(f"✅ Priority map saved to: {OUTPUT_PRIORITY_MAP}")

print("\n" + "="*60)
print("🏆 FINAL SUBMISSION SUMMARY")
print("="*60)
print("""
✅ Objective 1: Heat Hotspots Identified (LST maps)
✅ Objective 2: Drivers Analyzed (Feature importance: NDBI, Population, NDVI)
✅ Objective 3: Heat Dynamics Modelled (XGBoost, R² = 0.64)
✅ Objective 4: Cooling Scenarios Generated & Optimized
   - Strategies assigned from predicted LST
   - HVI overlaid for equity/urgency prioritization
   - Final map: 4 priority tiers (IMMEDIATE → STANDARD)
""")
print("📁 Output files:")
print(f"   - {OUTPUT_PREDICTIONS}")
print(f"   - {OUTPUT_PRIORITY_MAP}")
print("\n✅ Hackathon submission ready!")