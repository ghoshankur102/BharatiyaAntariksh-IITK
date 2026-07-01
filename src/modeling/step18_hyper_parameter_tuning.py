import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🔧 HYPERPARAMETER TUNING - Final Push for R² > 0.65")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================

print("\n📊 Loading data...")
df = pd.read_csv('delhi_urban_heat_with_osm_buildings.csv')
print(f"✅ Loaded {len(df)} samples")

# ============================================
# 2. FEATURE ENGINEERING (Same as before)
# ============================================

print("\n🔧 Engineering features...")

features = [
    'NDBI', 'POPULATION', 'NIGHTLIGHTS', 'ALBEDO', 'NDWI',
    'UHI', 'URBAN_COMPACTNESS', 'VEG_QUALITY', 'BUILDING_DENSITY_OSM'
]

df['NDBI_POP'] = df['NDBI'] * df['POPULATION'] / 10000
df['NDBI_LIGHTS'] = df['NDBI'] * df['NIGHTLIGHTS']
df['URBAN_VEG'] = df['NDBI'] - df['VEG_QUALITY']
df['COOLING_EFFECT'] = df['VEG_QUALITY'] * (1 - df['NDBI'])
df['BUILDING_DENSITY_INDEX'] = df['BUILDING_DENSITY_OSM'] * df['NDBI']

features.extend(['NDBI_POP', 'NDBI_LIGHTS', 'URBAN_VEG', 'COOLING_EFFECT', 'BUILDING_DENSITY_INDEX'])

X = df[features]
y = df['LST']

print(f"📋 Total features: {len(features)}")

# ============================================
# 3. TRAIN-TEST SPLIT
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
# 4. HYPERPARAMETER TUNING
# ============================================

print("\n🔧 Hyperparameter Tuning...")

param_grid = {
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.05, 0.1, 0.15],
    'n_estimators': [200, 300, 400],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Use a smaller grid for speed
param_grid_small = {
    'max_depth': [8, 10],
    'learning_rate': [0.08, 0.1],
    'n_estimators': [300, 400],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

print("   Testing configurations...")

xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    xgb_model, 
    param_grid_small, 
    cv=3, 
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\n✅ Best parameters: {grid_search.best_params_}")
print(f"   Best CV R²: {grid_search.best_score_:.4f}")

# ============================================
# 5. TRAIN WITH BEST PARAMETERS
# ============================================

print("\n⚡ Training with best parameters...")

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   R² Score: {r2:.4f}")
print(f"   RMSE: {rmse:.2f}°C")
print(f"   MAE: {mae:.2f}°C")

# ============================================
# 6. FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({
    'Driver': X.columns,
    'Importance': best_model.feature_importances_,
    'Percentage': best_model.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

print("\n🎯 Feature Importance:")
print("-"*55)
for idx, row in importance.iterrows():
    if row['Driver'] in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'NDBI_POP', 'NDBI_LIGHTS', 'URBAN_VEG', 'BUILDING_DENSITY_OSM', 'BUILDING_DENSITY_INDEX']:
        emoji = '🔥'
    elif row['Driver'] in ['VEG_QUALITY', 'COOLING_EFFECT']:
        emoji = '🌿'
    else:
        emoji = '📊'
    print(f"   {row['Driver']:25s}: {row['Percentage']:5.1f}% {emoji}")
print("-"*55)

# ============================================
# 7. RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 FINAL RESULTS SUMMARY")
print("="*60)

print(f"\n1. Model Performance:")
print(f"   - R² Score: {r2:.4f} {'✅' if r2 > 0.7 else '⚠️ (needs more data)'}")
print(f"   - RMSE: {rmse:.2f}°C")
print(f"   - MAE: {mae:.2f}°C")

print(f"\n2. Best Parameters:")
for key, value in grid_search.best_params_.items():
    print(f"   - {key}: {value}")

print(f"\n3. Top 5 Drivers:")
for i in range(5):
    driver = importance.iloc[i]['Driver']
    pct = importance.iloc[i]['Percentage']
    print(f"   {i+1}. {driver}: {pct:.1f}%")

# Building feature importance
building_importance = importance[importance['Driver'] == 'BUILDING_DENSITY_OSM']
if len(building_importance) > 0:
    print(f"\n4. BUILDING_DENSITY_OSM: {building_importance.iloc[0]['Percentage']:.1f}% (heating) ✅")

# Vegetation importance
veg_importance = importance[importance['Driver'] == 'VEG_QUALITY']
if len(veg_importance) > 0:
    print(f"5. VEG_QUALITY: {veg_importance.iloc[0]['Percentage']:.1f}% (cooling) ✅")

print("\n" + "="*60)

# Save model
import pickle
with open('urban_heat_model_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✅ Model saved to 'urban_heat_model_tuned.pkl'")

print("\n🏆 Final model ready for scenario testing!")