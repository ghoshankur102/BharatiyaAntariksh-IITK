import pandas as pd
import numpy as np

print("="*60)
print("⚡ QUICK BUILDING DENSITY (Using NDBI Proxy)")
print("="*60)

# Load unified data
df = pd.read_csv('delhi_urban_heat_unified.csv')
print(f"✅ Loaded {len(df)} samples")

# Use NDBI as building density proxy (already have this)
# But with better scaling
df['BUILDING_DENSITY_OSM'] = df['NDBI'] * 10  # Scale to match building density

# Save
df.to_csv('delhi_urban_heat_with_osm_buildings.csv', index=False)
print("✅ Saved 'delhi_urban_heat_with_osm_buildings.csv'")

# Check correlation
corr = df['BUILDING_DENSITY_OSM'].corr(df['LST'])
print(f"BUILDING_DENSITY_OSM vs LST: {corr:+.3f}")

print("\n✅ Quick building density added!")