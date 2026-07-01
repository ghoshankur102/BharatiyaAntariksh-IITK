#!/usr/bin/env python3
"""
plot_cooling_strategy_map_final.py
Generates a spatial map of Delhi with legend at top‑right and a spacious table at bottom‑right.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ================================================================
# 1. CONFIGURATION
# ================================================================

GEOJSON_FILE = "delhi_1997-2012_district.json"
HVI_CSV = "heat_vulnerability_index_final.csv"
OUTPUT_MAP = "delhi_cooling_strategy_map_final.png"

# ================================================================
# 2. LOAD DATA
# ================================================================

print("📂 Loading GeoJSON...")
gdf = gpd.read_file(GEOJSON_FILE)
gdf = gdf.to_crs('EPSG:4326')

print("📊 Loading HVI data...")
df_hvi = pd.read_csv(HVI_CSV)

# ================================================================
# 3. DEFINE DISTRICT INFO
# ================================================================

district_info = {
    'West': {
        'priority': 'Critical',
        'primary': 'Cool Roofs',
        'secondary': 'Reflective Pavements',
        'color': '#d73027',
        'icon': '🏠'
    },
    'Central': {
        'priority': 'Critical',
        'primary': 'Cool Roofs',
        'secondary': 'Blue-Green Infrastructure',
        'color': '#d73027',
        'icon': '🏠'
    },
    'South West': {
        'priority': 'Critical',
        'primary': 'Green Corridors',
        'secondary': 'Cool Roofs',
        'color': '#d73027',
        'icon': '🌳'
    },
    'North West': {
        'priority': 'High',
        'primary': 'Blue-Green Infrastructure',
        'secondary': 'Cool Roofs (slums)',
        'color': '#fc8d59',
        'icon': '💧'
    },
    'North East': {
        'priority': 'High',
        'primary': 'Tree Corridors',
        'secondary': 'Water Body Restoration',
        'color': '#fc8d59',
        'icon': '🌳'
    },
    'New Delhi': {
        'priority': 'Poverty-Driven',
        'primary': 'Cooling Centres',
        'secondary': 'Urban Forestry',
        'color': '#fee08b',
        'icon': '🏛️'
    },
    'North': {
        'priority': 'Poverty-Driven',
        'primary': 'Cool Roofs (equity)',
        'secondary': 'Water Access',
        'color': '#fee08b',
        'icon': '🏠'
    },
    'South': {
        'priority': 'Low',
        'primary': 'Maintain Green',
        'secondary': 'Monitoring',
        'color': '#91cf60',
        'icon': '🌿'
    },
    'East': {
        'priority': 'Low',
        'primary': 'Perimeter Shading',
        'secondary': 'Building Bylaws',
        'color': '#91cf60',
        'icon': '🌳'
    }
}

# Merge
gdf['color'] = gdf['DISTRICT'].map(lambda d: district_info.get(d, {}).get('color', '#cccccc'))
gdf['priority'] = gdf['DISTRICT'].map(lambda d: district_info.get(d, {}).get('priority', 'Unknown'))
gdf['icon'] = gdf['DISTRICT'].map(lambda d: district_info.get(d, {}).get('icon', ''))

# HVI and MPI from CSV
hvi_dict = df_hvi.set_index('district')['HVI'].to_dict()
mpi_dict = df_hvi.set_index('district')['poverty_mpi'].to_dict() if 'poverty_mpi' in df_hvi.columns else {}
gdf['HVI'] = gdf['DISTRICT'].map(hvi_dict)
gdf['MPI'] = gdf['DISTRICT'].map(mpi_dict)

# ================================================================
# 4. CREATE FIGURE WITH WIDE RIGHT MARGIN
# ================================================================

fig, ax = plt.subplots(1, 1, figsize=(20, 14))   # wider and taller
ax.set_aspect('equal')

# Plot districts
gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', linewidth=1.5, alpha=0.7)

# Labels and icons
for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid
    x, y = centroid.x, centroid.y
    ax.text(x, y, row['DISTRICT'], ha='center', va='center',
            fontsize=12, fontweight='bold', color='black',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
    if row['icon']:
        ax.text(x, y + 0.03, row['icon'], ha='center', va='center', fontsize=18)

# ================================================================
# 5. LEGEND – placed at TOP RIGHT
# ================================================================

legend_elements = [
    mpatches.Patch(color='#d73027', label='Critical – Act Now'),
    mpatches.Patch(color='#fc8d59', label='High – Major Intervention'),
    mpatches.Patch(color='#fee08b', label='Poverty-Driven – Equity Focus'),
    mpatches.Patch(color='#91cf60', label='Low – Maintain & Monitor'),
]
legend = ax.legend(handles=legend_elements,
                   loc='upper left',
                   bbox_to_anchor=(1.02, 0.95),
                   fontsize=12,
                   title='Priority Tier',
                   title_fontsize=14)

# ================================================================
# 6. INSET TABLE – placed at BOTTOM RIGHT with enough space
# ================================================================

table_df = gdf[['DISTRICT', 'HVI', 'MPI']].dropna(subset=['HVI']).sort_values('HVI', ascending=False)
if table_df.empty:
    table_df = gdf[['DISTRICT', 'HVI', 'MPI']].fillna({'HVI': 0, 'MPI': 0}).sort_values('HVI', ascending=False)

# Create table with larger columns
table = ax.table(cellText=table_df.values,
                 colLabels=['District', 'HVI', 'MPI'],
                 loc='center',
                 bbox=[1.02, 0.05, 0.35, 0.40],   # wider: 0.35, taller: 0.40
                 cellLoc='center',
                 colWidths=[0.2, 0.1, 0.1])       # District column wider
table.auto_set_font_size(False)
table.set_fontsize(9)   # smaller font to fit neatly
table.scale(1.2, 1.2)

# Color header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#d3d3d3')

# ================================================================
# 7. SCALE BAR & NORTH ARROW (inside map)
# ================================================================

x_min, x_max = gdf.total_bounds[0], gdf.total_bounds[2]
y_min, y_max = gdf.total_bounds[1], gdf.total_bounds[3]

# Scale bar
scale_x = x_min + 0.05 * (x_max - x_min)
scale_y = y_min + 0.05 * (y_max - y_min)
scale_length_deg = 0.09
ax.plot([scale_x, scale_x + scale_length_deg], [scale_y, scale_y],
        color='black', linewidth=3)
ax.text(scale_x + scale_length_deg/2, scale_y - 0.005, '~10 km',
        ha='center', va='top', fontsize=10)

# North arrow
x_arrow = x_max - 0.05 * (x_max - x_min)
y_arrow = y_max - 0.05 * (y_max - y_min)
ax.annotate('N', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - 0.02),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'),
            fontsize=14, fontweight='bold', ha='center', va='center')

# ================================================================
# 8. TITLE & FINISH
# ================================================================

ax.set_title('Delhi Urban Cooling Strategy Map\n'
             'Heat Vulnerability Index (HVI) + Recommended Interventions',
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xticks([])
ax.set_yticks([])

# Reserve the right 35% of the figure for legends and table
plt.subplots_adjust(right=0.65)

plt.savefig(OUTPUT_MAP, dpi=300, bbox_inches='tight')
print(f"✅ Map saved to: {OUTPUT_MAP}")
plt.show()

print("\n✅ Done! The table now has enough space for district names.")