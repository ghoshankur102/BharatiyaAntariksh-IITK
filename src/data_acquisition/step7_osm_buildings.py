import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🏙️ ADDING BUILDING FOOTPRINTS FROM OpenStreetMap")
print("="*60)

# ============================================
# 1. DOWNLOAD BUILDING DATA
# ============================================

print("\n📥 Downloading building footprints from OpenStreetMap...")
print("   This may take 3-5 minutes...")

delhi_bbox = box(77.0, 28.4, 77.4, 28.9)

try:
    buildings = ox.features_from_bbox(
        delhi_bbox.bounds,
        tags={'building': True}
    )
    print(f"✅ Downloaded {len(buildings)} building footprints")
except Exception as e:
    print(f"❌ Error downloading: {e}")
    buildings = ox.features_from_place(
        'Delhi, India',
        tags={'building': True}
    )
    print(f"✅ Downloaded {len(buildings)} building footprints")

# ============================================
# 2. CALCULATE BUILDING STATISTICS
# ============================================

print("\n📊 Calculating building statistics...")

buildings_proj = buildings.to_crs("EPSG:32643")
buildings_proj['area_m2'] = buildings_proj.geometry.area

if 'height' in buildings_proj.columns:
    buildings_proj['height_m'] = pd.to_numeric(buildings_proj['height'], errors='coerce')
else:
    buildings_proj['height_m'] = 10

if 'building:levels' in buildings_proj.columns:
    buildings_proj['levels'] = pd.to_numeric(buildings_proj['building:levels'], errors='coerce')
else:
    buildings_proj['levels'] = 2

buildings_proj['height_m'] = buildings_proj['height_m'].fillna(10)
buildings_proj['levels'] = buildings_proj['levels'].fillna(2)

print(f"   Total building area: {buildings_proj['area_m2'].sum() / 1e6:.2f} km²")
print(f"   Average building height: {buildings_proj['height_m'].mean():.1f} m")
print(f"   Average floors: {buildings_proj['levels'].mean():.1f}")
print(f"   Number of buildings: {len(buildings_proj)}")

# ============================================
# 3. CREATE GRID
# ============================================

print("\n📐 Creating grid for aggregation...")

cell_size = 0.001
xmin, ymin, xmax, ymax = delhi_bbox.bounds
x_cells = np.arange(xmin, xmax, cell_size)
y_cells = np.arange(ymin, ymax, cell_size)

grid_cells = []
for x in x_cells:
    for y in y_cells:
        grid_cells.append(box(x, y, x + cell_size, y + cell_size))

grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:4326")
grid_proj = grid.to_crs("EPSG:32643")

print(f"   Created {len(grid)} grid cells")

# ============================================
# 4. AGGREGATE BUILDING DATA TO GRID
# ============================================

print("\n🔗 Aggregating building data to grid...")

buildings_with_grid = gpd.sjoin(buildings_proj, grid_proj, how='inner', predicate='intersects')

if len(buildings_with_grid) > 0:
    grid_stats = buildings_with_grid.groupby('index_right').agg({
        'area_m2': ['sum', 'mean', 'count'],
        'height_m': 'mean',
        'levels': 'mean'
    }).reset_index()
    
    grid_stats.columns = ['grid_id', 'building_area_sum', 'building_area_mean', 
                          'building_count', 'building_height_mean', 'building_levels_mean']
    
    grid_cell_area = (cell_size * 111000) ** 2
    grid_stats['building_density'] = grid_stats['building_area_sum'] / grid_cell_area
    print(f"✅ Aggregated {len(grid_stats)} grid cells with building data")
else:
    print("❌ No buildings found in grid cells")
    grid_stats = pd.DataFrame()
    grid_stats['grid_id'] = range(len(grid_proj))
    grid_stats['building_density'] = 0

# ============================================
# 5. CREATE RASTER (FIXED INDEXING)
# ============================================

print("\n🗺️ Creating building density raster...")

grid_array = np.zeros((len(y_cells), len(x_cells)))

# Fill array with building density values
for idx, row in grid_stats.iterrows():
    try:
        grid_id = int(row['grid_id'])
        row_idx = grid_id // len(x_cells)
        col_idx = grid_id % len(x_cells)
        
        # Ensure indices are valid integers
        row_idx = int(row_idx)
        col_idx = int(col_idx)
        
        # Check bounds
        if row_idx < len(y_cells) and col_idx < len(x_cells):
            grid_array[row_idx, col_idx] = float(row['building_density'])
    except Exception as e:
        continue

# Create DataFrame
building_density_flat = grid_array.flatten()
building_df = pd.DataFrame({
    'building_density': building_density_flat
})
building_df.to_csv('building_density.csv', index=False)
print("✅ Saved building density to 'building_density.csv'")

# Statistics
density_values = building_density_flat[building_density_flat > 0]
if len(density_values) > 0:
    print(f"\n📊 Building Density Statistics:")
    print(f"   Mean: {density_values.mean():.3f}")
    print(f"   Max: {density_values.max():.3f}")
    print(f"   Min: {density_values.min():.3f}")
    print(f"   Total cells with buildings: {len(density_values)}")
else:
    print("⚠️ No building density values found")

# ============================================
# 6. VISUALIZATION
# ============================================

print("\n📈 Visualizing building density...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
if np.sum(grid_array) > 0:
    vmax_val = np.percentile(grid_array[grid_array > 0], 95)
    im1 = ax1.imshow(grid_array, cmap='Reds', vmin=0, vmax=vmax_val)
    plt.colorbar(im1, ax=ax1, label='Building Density (m²/m²)')
else:
    ax1.text(0.5, 0.5, 'No building density data', ha='center', va='center')
ax1.set_title('Building Density in Delhi')
ax1.set_xlabel('X Grid Cell')
ax1.set_ylabel('Y Grid Cell')

ax2 = axes[1]
if len(density_values) > 0:
    ax2.hist(density_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.set_xlabel('Building Density (m²/m²)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Building Density Distribution')
    ax2.axvline(density_values.mean(), color='red', linestyle='--', label=f'Mean: {density_values.mean():.3f}')
    ax2.legend()
else:
    ax2.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax2.set_title('Building Density Distribution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('building_density.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'building_density.png'")
plt.show()

print("\n✅ Building data processing complete!")