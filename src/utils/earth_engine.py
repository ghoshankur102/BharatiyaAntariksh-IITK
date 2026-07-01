import ee

# Initialize with your project ID
ee.Initialize(project='urbanheatmitigation-500914')  # Replace with your actual project ID

# Simple test
try:
    # Try to load a simple image collection
    test = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').limit(1)
    print("✅ Earth Engine is connected and working!")
    print(f"✅ Successfully loaded test image collection")
except Exception as e:
    print(f"❌ Error: {e}")