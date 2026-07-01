import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("📂 Reading exported GeoTIFFs...")

# ============================================
# 1. READ EACH BAND
# ============================================

band_names = ['LST', 'NDVI', 'NDBI', 'NDWI', 'ALBEDO']
band_files = [
    'delhi_lst.tif',
    'delhi_ndvi.tif', 
    'delhi_ndbi.tif',
    'delhi_ndwi.tif',
    'delhi_albedo.tif'
]

data_dict = {}

for name, file_path in zip(band_names, band_files):
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            data_dict[name] = data.flatten()
            print(f"✅ Loaded {name}: {len(data_dict[name])} pixels")
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print(f"   Make sure you downloaded the file from Google Drive")
        print(f"   Update the file path to where you saved it")
        data_dict[name] = np.array([])

# ============================================
# 2. CREATE DATAFRAME
# ============================================

print("\n📊 Creating DataFrame...")

df = pd.DataFrame(data_dict)

# Remove invalid values
df = df.dropna()
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Filter for realistic values
df = df[(df['LST'] > 20) & (df['LST'] < 50)]
df = df[(df['NDVI'] > -0.5) & (df['NDVI'] < 1.0)]
df = df[(df['NDBI'] > -0.5) & (df['NDBI'] < 0.5)]

print(f"✅ Data extracted! {len(df)} samples collected")

# ============================================
# 3. SAVE CSV
# ============================================

df.to_csv('delhi_urban_heat_final.csv', index=False)
print("✅ Data saved to 'delhi_urban_heat_final.csv'")

# ============================================
# 4. STATISTICS
# ============================================

print("\n📊 Feature Statistics:")
print(df.describe())

print("\n📈 Correlation with LST:")
corr = df.corr()['LST'].sort_values(ascending=False)
print(corr)

# Save correlation
corr.to_csv('correlation_results.csv')
print("✅ Correlation saved to 'correlation_results.csv'")