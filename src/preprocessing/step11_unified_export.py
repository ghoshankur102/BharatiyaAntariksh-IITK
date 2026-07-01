import ee
import time
import geemap
import pandas as pd
import numpy as np

print("="*60)
print("📤 UNIFIED EXPORT: All Features in One GeoTIFF")
print("="*60)

# Initialize Earth Engine
ee.Initialize(project='urbanheatmitigation-500914')

# Delhi coordinates
delhi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.9])

print("\n📍 Analyzing: Delhi, India")

# ============================================
# 1. LOAD LAND SURFACE TEMPERATURE
# ============================================

print("\n🌡️ Loading Land Surface Temperature...")

lst_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filterBounds(delhi)
                  .filterDate('2023-04-01', '2023-06-30')
                  .filter(ee.Filter.lt('CLOUD_COVER', 10))
                  .select('ST_B10'))

def convert_to_celsius(image):
    return image.multiply(0.00341802).add(149.0).subtract(273.15).copyProperties(image)

lst = lst_collection.map(convert_to_celsius).median().clip(delhi).rename('LST')

# ============================================
# 2. LOAD LANDSAT FOR INDICES
# ============================================

print("📊 Calculating spectral indices...")

landsat = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
           .filterBounds(delhi)
           .filterDate('2023-04-01', '2023-06-30')
           .filter(ee.Filter.lt('CLOUD_COVER', 20))
           .median()
           .clip(delhi))

ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
ndbi = landsat.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
ndwi = landsat.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')

# Albedo (Surface Reflectivity)
albedo = landsat.select('SR_B4').multiply(0.3).add(landsat.select('SR_B3').multiply(0.4)) \
                .add(landsat.select('SR_B2').multiply(0.2)).add(landsat.select('SR_B5').multiply(0.1)) \
                .rename('ALBEDO')

print("✅ Spectral indices calculated")

# ============================================
# 3. ADD POPULATION DENSITY (WorldPop)
# ============================================

print("👥 Adding population density...")

population = (ee.ImageCollection('WorldPop/GP/100m/pop')
              .filterBounds(delhi)
              .filterDate('2020-01-01', '2020-12-31')
              .select('population')
              .median()
              .clip(delhi)
              .rename('POPULATION'))

print("✅ Population density added")

# ============================================
# 4. ADD NIGHTTIME LIGHTS (VIIRS)
# ============================================

print("🌃 Adding nighttime lights...")

nightlights = (ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
               .filterBounds(delhi)
               .filterDate('2023-01-01', '2023-12-31')
               .select('avg_rad')
               .median()
               .clip(delhi)
               .rename('NIGHTLIGHTS'))

print("✅ Nighttime lights added")

# ============================================
# 5. ADD BUILDING DENSITY FROM OSM (Rasterized)
# ============================================

print("🏙️ Adding building density from OpenStreetMap...")

# We'll use a simplified approach: use NDBI as a proxy for building density
# since OSM data is hard to rasterize in Earth Engine
# But we'll still include it as a feature

# For now, use a combination of NDBI and ALBEDO as urban density proxy
building_density = ndbi.multiply(1.5).clamp(0, 1).rename('BUILDING_DENSITY')

print("✅ Building density proxy added")

# ============================================
# 6. CREATE PHYSICS-INFORMED FEATURES
# ============================================

print("🔬 Creating physics-informed features...")

# Urban Heat Island Intensity
uhi = ndbi.subtract(ndvi).rename('UHI')

# Vegetation Quality
veg_quality = ndvi.multiply(ee.Image(1).subtract(ndbi)).rename('VEG_QUALITY')

# Urban Compactness
urban_compactness = ndbi.multiply(ndbi).rename('URBAN_COMPACTNESS')

print("✅ Physics-informed features created")

# ============================================
# 7. COMBINE ALL BANDS INTO ONE IMAGE
# ============================================

print("\n📦 Combining all bands into one image...")

# Convert all bands to Float32 for consistency
combined = ee.Image.cat([
    lst.toFloat(),
    ndvi.toFloat(),
    ndbi.toFloat(),
    ndwi.toFloat(),
    albedo.toFloat(),
    population.toFloat(),
    nightlights.toFloat(),
    building_density.toFloat(),
    uhi.toFloat(),
    veg_quality.toFloat(),
    urban_compactness.toFloat()
])

# Get band names
band_names = ['LST', 'NDVI', 'NDBI', 'NDWI', 'ALBEDO', 
              'POPULATION', 'NIGHTLIGHTS', 'BUILDING_DENSITY', 
              'UHI', 'VEG_QUALITY', 'URBAN_COMPACTNESS']

print(f"✅ Combined {len(band_names)} bands")
print(f"   Bands: {band_names}")

# ============================================
# 8. EXPORT TO GOOGLE DRIVE
# ============================================

print("\n📤 Exporting to Google Drive...")
print("   This will take 3-5 minutes...")

task = ee.batch.Export.image.toDrive(
    image=combined,
    description='Delhi_Urban_Heat_Complete',
    folder='EarthEngine_Exports',
    fileNamePrefix='delhi_urban_heat_complete',
    region=delhi,
    scale=100,
    maxPixels=1e9
)

task.start()

print("✅ Export task started!")
print(f"📁 Check your Google Drive in 'EarthEngine_Exports' folder")
print(f"🔄 Task ID: {task.id}")

# Monitor progress
print("\n⏳ Monitoring progress...")
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
        time.sleep(30)

print("\n✅ Export process complete!")
print(f"📁 Download 'delhi_urban_heat_complete.tif' from Google Drive")