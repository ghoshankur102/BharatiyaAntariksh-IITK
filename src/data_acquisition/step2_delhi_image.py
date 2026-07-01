import ee
import geemap

# Initialize Earth Engine
ee.Initialize(project='urbanheatmitigation-500914')  # Use your actual project ID

# Delhi coordinates
delhi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.9])

print("📍 Analyzing: Delhi, India")
print(f"   Bounding Box: {delhi.getInfo()['coordinates']}")

# Load a single Landsat 8 image (less cloudy)
image = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
         .filterBounds(delhi)
         .filterDate('2023-01-01', '2023-06-30')
         .filter(ee.Filter.lt('CLOUD_COVER', 10))
         .first())

# Get image info
print("\n📸 Image Information:")
print(f"  - Date: {image.get('DATE_ACQUIRED').getInfo()}")
print(f"  - Cloud Cover: {image.get('CLOUD_COVER').getInfo()}%")
print(f"  - Bands: {image.bandNames().getInfo()}")

# Create interactive map
print("\n🗺️ Opening map...")
Map = geemap.Map(center=[28.6, 77.2], zoom=11)

# Add true color image (using correct band names: SR_B4=Red, SR_B3=Green, SR_B2=Blue)
Map.addLayer(
    image,
    {'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min': 0, 'max': 30000},
    'Delhi - True Color'
)

# Add false color (vegetation appears red: SR_B5=NIR, SR_B4=Red, SR_B3=Green)
Map.addLayer(
    image,
    {'bands': ['SR_B5', 'SR_B4', 'SR_B3'], 'min': 0, 'max': 30000},
    'Delhi - False Color'
)

# Display the map
Map