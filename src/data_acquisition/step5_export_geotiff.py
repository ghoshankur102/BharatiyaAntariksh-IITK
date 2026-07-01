import ee
import time

ee.Initialize(project='urbanheatmitigation-500914')

# Delhi coordinates
delhi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.9])

print("📤 Exporting data as GeoTIFF (fixed data types)...")

# ============================================
# 1. LOAD DATA
# ============================================

print("Loading data...")

# LST
lst_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filterBounds(delhi)
                  .filterDate('2023-04-01', '2023-06-30')
                  .filter(ee.Filter.lt('CLOUD_COVER', 20))
                  .select('ST_B10'))

def convert_to_celsius(image):
    return image.multiply(0.00341802).add(149.0).subtract(273.15).copyProperties(image)

lst = lst_collection.map(convert_to_celsius).median().clip(delhi).rename('LST')

# Indices from Landsat
landsat = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
           .filterBounds(delhi)
           .filterDate('2023-04-01', '2023-06-30')
           .filter(ee.Filter.lt('CLOUD_COVER', 20))
           .median()
           .clip(delhi))

ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
ndbi = landsat.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
ndwi = landsat.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')

albedo = landsat.select('SR_B4').multiply(0.3).add(landsat.select('SR_B3').multiply(0.4)) \
                .add(landsat.select('SR_B2').multiply(0.2)).add(landsat.select('SR_B5').multiply(0.1)) \
                .rename('ALBEDO')

# ============================================
# 2. CONVERT ALL BANDS TO FLOAT32
# ============================================

print("Converting all bands to Float32...")

lst = lst.toFloat()
ndvi = ndvi.toFloat()
ndbi = ndbi.toFloat()
ndwi = ndwi.toFloat()
albedo = albedo.toFloat()

# ============================================
# 3. COMBINE ALL BANDS
# ============================================

print("Combining bands...")
combined = ee.Image.cat([lst, ndvi, ndbi, ndwi, albedo])

# Verify data types
print("✅ All bands converted to Float32")

# ============================================
# 4. EXPORT TO GOOGLE DRIVE
# ============================================

print("\nStarting export to Google Drive...")
print("This will take 2-5 minutes...")

task = ee.batch.Export.image.toDrive(
    image=combined,
    description='Delhi_Urban_Heat_Data',
    folder='EarthEngine_Exports',
    fileNamePrefix='delhi_urban_heat',
    region=delhi,
    scale=100,
    maxPixels=1e9
)

task.start()

print("✅ Export task started!")
print(f"📁 Check your Google Drive in 'EarthEngine_Exports' folder")
print(f"🔄 Task ID: {task.id}")
print("⏳ Waiting for completion...")

# Monitor progress
while True:
    status = task.status()
    state = status['state']
    print(f"   Status: {state}")
    
    if state == 'COMPLETED':
        print("✅ Export completed successfully!")
        print("📁 Download the file from your Google Drive")
        break
    elif state == 'FAILED':
        print(f"❌ Export failed: {status.get('error_message', 'Unknown error')}")
        break
    else:
        time.sleep(30)  # Check every 30 seconds