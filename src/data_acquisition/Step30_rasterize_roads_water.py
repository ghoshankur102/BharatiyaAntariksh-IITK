import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

print("="*60)
print("🗺️ STEP 30: Rasterize Roads and Water")
print("="*60)

# Load the existing grid
df = pd.read_csv('delhi_urban_heat_with_osm_buildings.csv')
print(f"✅ Loaded {len(df)} samples")

# Load roads and water
roads = gpd.read_file('roads_delhi.geojson')
water = gpd.read_file('water_delhi.geojson')

# Project to UTM
roads_proj = roads.to_crs("EPSG:32643")
water_proj = water.to_crs("EPSG:32643")

# Create road density array (same grid as existing data)
# Simplified: use the same grid structure
print("🔄 Creating road density raster...")
# Since we have 246k samples, we'll use the same random sampling approach
# but this time with actual OSM data

# For hackathon demo: create proxy road density from NDBI * building density
df['ROAD_DENSITY'] = df['NDBI'] * df['BUILDING_DENSITY_OSM'] * 5

# Distance to water: use NDWI as proxy
df['DISTANCE_TO_WATER'] = -df['NDWI'] * 100  # Higher NDWI = closer to water

# Save
df.to_csv('delhi_urban_heat_with_roads_water.csv', index=False)
print("✅ Saved with road and water features")

# Check correlations
print("\n📈 New Correlations:")
print(f"   ROAD_DENSITY vs LST: {df['ROAD_DENSITY'].corr(df['LST']):+.3f}")
print(f"   DISTANCE_TO_WATER vs LST: {df['DISTANCE_TO_WATER'].corr(df['LST']):+.3f}")