import ee
import geemap
import pandas as pd
import numpy as np

ee.Initialize(project='urbanheatmitigation-500914')

# Delhi coordinates
delhi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.9])

print("📥 Downloading using geemap...")

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

# Indices
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

combined = ee.Image.cat([lst, ndvi, ndbi, ndwi, albedo])

# ============================================
# 2. DOWNLOAD WITH GEEMAP
# ============================================

print("Downloading GeoTIFF...")
print("This will take 2-5 minutes...")

# Create map
Map = geemap.Map()

# Download as GeoTIFF
geemap.download_ee_image(
    combined,
    filename='delhi_urban_heat.tif',
    region=delhi,
    scale=100
)

print("✅ Download complete! File saved as 'delhi_urban_heat.tif'")