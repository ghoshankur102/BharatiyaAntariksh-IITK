import ee
import geemap

# Initialize Earth Engine
ee.Initialize(project='urbanheatmitigation-500914')

# Delhi coordinates
delhi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.9])

print("🌿 Calculating vegetation and urban indices for Delhi...")

# Use the NEW harmonized Sentinel-2 dataset
sentinel2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(delhi)
             .filterDate('2023-04-01', '2023-06-30')
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
             .select(['B2', 'B3', 'B4', 'B8', 'B11'])  # Select only needed bands
             .median()
             .clip(delhi))

print("✅ Sentinel-2 data loaded!")

# Calculate indices using the selected bands
ndvi = sentinel2.normalizedDifference(['B8', 'B4']).rename('NDVI')  # Vegetation
ndbi = sentinel2.normalizedDifference(['B11', 'B8']).rename('NDBI') # Built-up
ndwi = sentinel2.normalizedDifference(['B3', 'B8']).rename('NDWI')  # Water

print("✅ Indices calculated!")

# Create map
print("\n🗺️ Visualizing indices...")
Map = geemap.Map(center=[28.6, 77.2], zoom=11)

# NDVI (Vegetation - green)
Map.addLayer(
    ndvi,
    {'min': -0.2, 'max': 0.8, 'palette': ['brown', 'yellow', 'green', 'darkgreen']},
    'NDVI - Vegetation'
)

# NDBI (Built-up - red)
Map.addLayer(
    ndbi,
    {'min': -0.5, 'max': 0.5, 'palette': ['blue', 'yellow', 'red']},
    'NDBI - Built-up'
)

# NDWI (Water - blue)
Map.addLayer(
    ndwi,
    {'min': -0.5, 'max': 0.5, 'palette': ['brown', 'blue', 'darkblue']},
    'NDWI - Water'
)

Map.addLayerControl()
Map

# Print statistics
print("\n📊 Index Statistics:")

# NDVI statistics
stats_ndvi = ndvi.reduceRegion(
    reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True)
                 .combine(ee.Reducer.max(), '', True),
    geometry=delhi,
    scale=10,
    maxPixels=1e9
)

# NDBI statistics
stats_ndbi = ndbi.reduceRegion(
    reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True)
                 .combine(ee.Reducer.max(), '', True),
    geometry=delhi,
    scale=10,
    maxPixels=1e9
)

# NDWI statistics
stats_ndwi = ndwi.reduceRegion(
    reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True)
                 .combine(ee.Reducer.max(), '', True),
    geometry=delhi,
    scale=10,
    maxPixels=1e9
)

# Extract values safely
ndvi_mean = stats_ndvi.get('NDVI_mean').getInfo()
ndvi_min = stats_ndvi.get('NDVI_min').getInfo()
ndvi_max = stats_ndvi.get('NDVI_max').getInfo()

ndbi_mean = stats_ndbi.get('NDBI_mean').getInfo()
ndbi_min = stats_ndbi.get('NDBI_min').getInfo()
ndbi_max = stats_ndbi.get('NDBI_max').getInfo()

ndwi_mean = stats_ndwi.get('NDWI_mean').getInfo()
ndwi_min = stats_ndwi.get('NDWI_min').getInfo()
ndwi_max = stats_ndwi.get('NDWI_max').getInfo()

print(f"NDVI (Vegetation - higher = more green):")
print(f"  - Mean: {ndvi_mean:.3f}")
print(f"  - Min: {ndvi_min:.3f}")
print(f"  - Max: {ndvi_max:.3f}")

print(f"\nNDBI (Built-up - higher = more urban):")
print(f"  - Mean: {ndbi_mean:.3f}")
print(f"  - Min: {ndbi_min:.3f}")
print(f"  - Max: {ndbi_max:.3f}")

print(f"\nNDWI (Water - higher = more water):")
print(f"  - Mean: {ndwi_mean:.3f}")
print(f"  - Min: {ndwi_min:.3f}")
print(f"  - Max: {ndwi_max:.3f}")

print("\n✅ Analysis complete! Check the map for visual interpretation.")