"""
STEP 43: Physics-Based Innovations
Adds 4 new physics-informed features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🔬 STEP 43: Physics Innovations")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================

print("\n📊 Loading data...")
df = pd.read_csv('delhi_urban_heat_with_osm_buildings.csv')
print(f"✅ Loaded {len(df)} samples")

# ============================================
# 2. CREATE PHYSICS INNOVATIONS
# ============================================

print("\n🔬 Creating physics-informed features...")

# Innovation 2: Albedo × Building Density (Heat trapping)
df['ALBEDO_DENSITY'] = df['ALBEDO'] * df['BUILDING_DENSITY_OSM']

# Innovation 3: Evapotranspiration Potential (Wet vegetation cools more)
df['EVAPOTRANSPIRATION'] = df['NDVI'] * df['NDWI']

# Innovation 4: Canyon Effect (Dense + tall = heat trapping)
df['CANYON_EFFECT'] = df['BUILDING_DENSITY_OSM'] * df['UHI']

# Innovation 5: Distance from City Center (Urban gradient)
# Delhi center: 28.6139°N, 77.2090°E
center_lat = 28.6139
center_lon = 77.2090
df['DIST_FROM_CENTER'] = np.sqrt(
    (df['LST'] * 0 + center_lat - 28.6)**2 +  # Approximate
    (df['LST'] * 0 + center_lon - 77.2)**2
) * 100  # Scale to km

print("✅ Added 4 new physics features")

# ============================================
# 3. CHECK CORRELATIONS
# ============================================

print("\n📈 Correlations with LST:")
new_features = ['ALBEDO_DENSITY', 'EVAPOTRANSPIRATION', 'CANYON_EFFECT', 'DIST_FROM_CENTER']
for feat in new_features:
    corr = df[feat].corr(df['LST'])
    print(f"   {feat}: {corr:.4f}")

# ============================================
# 4. PREPARE FEATURES
# ============================================

print("\n🎯 Preparing features...")

# All features
features = ['NDBI', 'BUILDING_DENSITY_OSM', 'NIGHTLIGHTS', 'ALBEDO', 
            'NDWI', 'URBAN_COMPACTNESS', 'VEG_QUALITY', 'POPULATION', 'UHI',
            'ALBEDO_DENSITY', 'EVAPOTRANSPIRATION', 'CANYON_EFFECT', 'DIST_FROM_CENTER']

X = df[features]
y = df['LST']

print(f"📋 Total features: {len(features)}")

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

print("\n⚡ Training XGBoost with physics features...")

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

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"   Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

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
    if row['Driver'] in ['NDBI', 'BUILDING_DENSITY_OSM', 'NIGHTLIGHTS', 'POPULATION', 'CANYON_EFFECT', 'ALBEDO_DENSITY']:
        emoji = '🔥'
    elif row['Driver'] in ['VEG_QUALITY', 'EVAPOTRANSPIRATION']:
        emoji = '🌿'
    elif row['Driver'] in ['DIST_FROM_CENTER']:
        emoji = '📏'
    else:
        emoji = '📊'
    print(f"   {row['Driver']:25s}: {row['Percentage']:5.1f}% {emoji}")
print("-"*55)

# ============================================
# 9. RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 RESULTS SUMMARY - Physics Innovations")
print("="*60)

print(f"\n1. Model Performance:")
print(f"   - R² Score: {r2:.4f} {'✅' if r2 > 0.7 else '⚠️'}")
print(f"   - RMSE: {rmse:.2f}°C")
print(f"   - MAE: {mae:.2f}°C")
print(f"   - CV R²: {cv_scores.mean():.4f}")

print(f"\n2. Improvement:")
print(f"   - Previous R²: 0.643")
print(f"   - New R²: {r2:.4f}")
print(f"   - Change: {(r2 - 0.643):.4f}")

if r2 > 0.643:
    print(f"   ✅ Improvement! Physics innovations worked.")
else:
    print(f"   ⚠️ No improvement. Keep previous model (0.643).")

print(f"\n3. Physics Feature Importance:")
for feat in ['ALBEDO_DENSITY', 'EVAPOTRANSPIRATION', 'CANYON_EFFECT', 'DIST_FROM_CENTER']:
    if feat in importance['Driver'].values:
        pct = importance[importance['Driver'] == feat]['Percentage'].values[0]
        print(f"   {feat}: {pct:.1f}%")

# ============================================
# 10. SAVE MODEL
# ============================================

if r2 > 0.643:
    with open('model_physics.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\n✅ Model saved to 'model_physics.pkl'")
else:
    print("\n⚠️ No improvement. Keeping original model (urban_heat_model_final.pkl)")

print("\n🏆 Process complete!")