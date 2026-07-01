import ee
import geemap

# Initialize Earth Engine
ee.Initialize(project='urbanheatmitigation-500914')

# Delhi coordinates
delhi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.9])

print("🌡️ Calculating Land Surface Temperature for Delhi...")

# Load Landsat collection for summer 2023
lst_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filterBounds(delhi)
                  .filterDate('2023-04-01', '2023-06-30')
                  .filter(ee.Filter.lt('CLOUD_COVER', 10))
                  .select('ST_B10'))

# Convert Kelvin to Celsius
def convert_to_celsius(image):
    return image.multiply(0.00341802).add(149.0).subtract(273.15).copyProperties(image)

lst_celsius = lst_collection.map(convert_to_celsius)

# Get median LST
lst_median = lst_celsius.median().clip(delhi)

print("✅ LST calculated!")

# Visualize on map
print("\n🗺️ Creating temperature map...")
Map = geemap.Map(center=[28.6, 77.2], zoom=11)

# Add LST layer with temperature colormap
Map.addLayer(
    lst_median,
    {
        'min': 25,
        'max': 45,
        'palette': ['#2c7bb6', '#abd9e9', '#fdae61', '#f46d43', '#d7191c']
    },
    'Delhi Land Surface Temperature (°C)'
)

Map.addLayerControl()
Map

# Print temperature stats
print("\n📊 Temperature Statistics (in °C):")
stats = lst_median.reduceRegion(
    reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True)
                 .combine(ee.Reducer.max(), '', True)
                 .combine(ee.Reducer.stdDev(), '', True),
    geometry=delhi,
    scale=100,
    maxPixels=1e9
)

print(f"  - Mean: {stats.get('ST_B10_mean').getInfo():.1f}°C")
print(f"  - Min: {stats.get('ST_B10_min').getInfo():.1f}°C")
print(f"  - Max: {stats.get('ST_B10_max').getInfo():.1f}°C")
print(f"  - Std Dev: {stats.get('ST_B10_stdDev').getInfo():.1f}°C")

# Identify hotspots (areas above 40°C)
print("\n🔥 Identifying Urban Heat Hotspots (areas > 40°C):")
hotspot_threshold = 40

# Create a mask for hotspots
hotspot_mask = lst_median.gt(hotspot_threshold)

# Count pixels above threshold
pixel_count = hotspot_mask.reduceRegion(
    reducer=ee.Reducer.count(),
    geometry=delhi,
    scale=100,
    maxPixels=1e9
)

# Calculate area
try:
    count_value = pixel_count.getInfo()
    # Get the first value from the dictionary
    if isinstance(count_value, dict):
        pixel_count_val = list(count_value.values())[0] if count_value.values() else 0
    else:
        pixel_count_val = count_value
    
    pixel_area_km2 = (100 * 100) / 1_000_000  # 100m pixel in km²
    hotspot_area_km2 = pixel_count_val * pixel_area_km2
    
    print(f"  - Number of hotspot pixels: {pixel_count_val}")
    print(f"  - Area with LST > 40°C: {hotspot_area_km2:.1f} km²")
    print(f"  - Percentage of Delhi: {(hotspot_area_km2 / 1484) * 100:.1f}%")
    
except Exception as e:
    print(f"  - Could not calculate area: {e}")

print("\n✅ Heat stress analysis complete!")