import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

print("="*60)
print("🛣️ STEP 29: Extract Roads and Water from OSM")
print("="*60)

# Delhi bounding box
delhi_bbox = box(77.0, 28.4, 77.4, 28.9)

print("\n📥 Downloading road network...")
roads = ox.features_from_bbox(
    delhi_bbox.bounds,
    tags={'highway': True}
)
print(f"✅ Downloaded {len(roads)} road segments")

print("\n📥 Downloading water bodies...")
water = ox.features_from_bbox(
    delhi_bbox.bounds,
    tags={'water': True}
)
print(f"✅ Downloaded {len(water)} water bodies")

# Save for later
roads.to_file('roads_delhi.geojson', driver='GeoJSON')
water.to_file('water_delhi.geojson', driver='GeoJSON')
print("✅ Saved to GeoJSON files")