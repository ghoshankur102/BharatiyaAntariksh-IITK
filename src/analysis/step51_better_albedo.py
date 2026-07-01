"""
STEP 56: Improved Albedo from Landsat SR Bands
"""

import ee
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("☀️ STEP 56: Improved Albedo from Landsat SR")
print("="*60)

# Initialize Earth Engine
ee.Initialize(project='urbanheatmitigation-500914')

# Delhi coordinates
delhi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.9])

print("\n📍 Delhi, India")

# ============================================
# 1. LOAD LANDSAT SR BANDS
# ============================================

print("\n📥 Loading Landsat 8 Surface Reflectance...")

landsat_sr = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
              .filterBounds(delhi)
              .filterDate('2023-04-01', '2023-06-30')
              .filter(ee.Filter.lt('CLOUD_COVER', 20))
              .median()
              .clip(delhi))

# Calculate albedo using Landsat 8 SR bands with standard coefficients
# albedo = 0.356*B2 + 0.130*B4 + 0.373*B5 + 0.085*B6 + 0.072*B7 + 0.058*B8
# Note: B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1, B7=SWIR2, B8=Pan? Actually for Landsat 8, bands are:
# B2 = Blue (0.45-0.51), B3 = Green (0.53-0.59), B4 = Red (0.64-0.67), B5 = NIR (0.85-0.88), B6 = SWIR1 (1.57-1.65), B7 = SWIR2 (2.11-2.29)
# We'll use a common formula: albedo = 0.356*B2 + 0.130*B4 + 0.373*B5 + 0.085*B6 + 0.072*B7 - 0.0018 (adjustment) but we'll simplify.

albedo_sr = (landsat_sr.select('SR_B2').multiply(0.356)
             .add(landsat_sr.select('SR_B4').multiply(0.130))
             .add(landsat_sr.select('SR_B5').multiply(0.373))
             .add(landsat_sr.select('SR_B6').multiply(0.085))
             .add(landsat_sr.select('SR_B7').multiply(0.072))
             .rename('ALBEDO_SR'))

print("✅ Albedo computed from SR bands")

# ============================================
# 2. SAMPLE ALBEDO VALUES
# ============================================

print("\n📊 Sampling albedo values...")

scale = 100
num_pixels = 5000

sample = albedo_sr.sample(
    region=delhi,
    scale=scale,
    numPixels=num_pixels,
    seed=42,
    geometries=False
)

sample_data = sample.getInfo()

if 'features' in sample_data and len(sample_data['features']) > 0:
    data_list = []
    for feature in sample_data['features']:
        data_list.append(feature['properties'])
    
    albedo_df = pd.DataFrame(data_list)
    albedo_df.columns = ['ALBEDO_SR']
    albedo_df = albedo_df.dropna()
    albedo_df.to_csv('albedo_sr.csv', index=False)
    print(f"✅ Saved {len(albedo_df)} albedo samples to 'albedo_sr.csv'")
    print(f"   Mean Albedo (SR): {albedo_df['ALBEDO_SR'].mean():.3f}")
    print(f"   Min: {albedo_df['ALBEDO_SR'].min():.3f}, Max: {albedo_df['ALBEDO_SR'].max():.3f}")
else:
    print("❌ No albedo data extracted")
    exit()

# ============================================
# 3. LOAD EXISTING DATA AND REPLACE ALBEDO
# ============================================

print("\n📊 Loading existing data...")
df = pd.read_csv('delhi_urban_heat_with_osm_buildings.csv')
print(f"✅ Loaded {len(df)} samples")

# Load albedo SR and align
df_albedo = pd.read_csv('albedo_sr.csv')
n_samples = min(len(df), len(df_albedo))
df = df.iloc[:n_samples].reset_index(drop=True)
df_albedo = df_albedo.iloc[:n_samples].reset_index(drop=True)

# Replace ALBEDO with improved version
df['ALBEDO_SR'] = df_albedo['ALBEDO_SR'].values

# Check correlation
print(f"\n📈 Correlation of ALBEDO_SR with LST: {df['ALBEDO_SR'].corr(df['LST']):.3f}")
print(f"   Old ALBEDO correlation: {df['ALBEDO'].corr(df['LST']):.3f}")

# ============================================
# 4. PREPARE FEATURES
# ============================================

print("\n🎯 Preparing features...")

features = ['NDBI', 'BUILDING_DENSITY_OSM', 'NIGHTLIGHTS', 'ALBEDO_SR', 
            'NDWI', 'URBAN_COMPACTNESS', 'VEG_QUALITY', 'POPULATION', 'UHI']

X = df[features]
y = df['LST']

print(f"📋 Features: {X.columns.tolist()}")
print(f"   Total features: {len(X.columns)}")

# ============================================
# 5. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📊 Data Split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# ============================================
# 6. TRAIN XGBOOST
# ============================================

print("\n⚡ Training XGBoost with improved albedo...")

model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

print("✅ Model trained!")

# ============================================
# 7. EVALUATE
# ============================================

print("\n📊 Model Performance:")

y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"   R² Score: {r2:.4f}")
print(f"   RMSE: {rmse:.2f}°C")
print(f"   MAE: {mae:.2f}°C")

# ============================================
# 8. FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({
    'Driver': X.columns,
    'Importance': model.feature_importances_,
    'Percentage': model.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

print("\n🎯 Feature Importance:")
print("-"*55)
for idx, row in importance.iterrows():
    if row['Driver'] in ['NDBI', 'BUILDING_DENSITY_OSM', 'NIGHTLIGHTS', 'POPULATION']:
        emoji = '🔥'
    elif row['Driver'] in ['VEG_QUALITY']:
        emoji = '🌿'
    elif row['Driver'] in ['ALBEDO_SR']:
        emoji = '☀️'
    else:
        emoji = '📊'
    print(f"   {row['Driver']:20s}: {row['Percentage']:5.1f}% {emoji}")
print("-"*55)

# ============================================
# 9. RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 RESULTS SUMMARY - With Improved Albedo")
print("="*60)

print(f"\n1. Model Performance:")
print(f"   - R² Score: {r2:.4f} {'✅' if r2 > 0.7 else '⚠️'}")
print(f"   - RMSE: {rmse:.2f}°C")
print(f"   - MAE: {mae:.2f}°C")

print(f"\n2. Improvement:")
print(f"   - Previous R² (original): 0.643")
print(f"   - Previous R² (with old albedo): 0.643")
print(f"   - New R²: {r2:.4f}")
print(f"   - Change: {(r2 - 0.643):.4f}")

if r2 > 0.643:
    print(f"   ✅ Improvement! Albedo SR helped.")
else:
    print(f"   ⚠️ No improvement. Keep original model (0.643).")

print(f"\n3. Albedo SR Importance:")
alb_pct = importance[importance['Driver'] == 'ALBEDO_SR']['Percentage'].values[0] if 'ALBEDO_SR' in importance['Driver'].values else 0
print(f"   ALBEDO_SR: {alb_pct:.1f}%")

# ============================================
# 10. SAVE MODEL
# ============================================

if r2 > 0.643:
    with open('model_with_albedo_sr.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\n✅ Model saved to 'model_with_albedo_sr.pkl'")
else:
    print("\n⚠️ No improvement. Keeping original model (urban_heat_model_final.pkl)")

print("\n🏆 Process complete!")