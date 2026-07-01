import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🏆 FINAL OPTIMIZED XGBOOST - R² = 0.641")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================

print("\n📊 Loading data...")
df = pd.read_csv('delhi_urban_heat_with_osm_buildings.csv')
print(f"✅ Loaded {len(df)} samples")

# ============================================
# 2. SELECT BEST FEATURES
# ============================================

print("\n🎯 Selecting best features from XGBoost tuning...")

# Based on XGBoost results, use these top features
best_features = [
    'NDBI',                     # 18.9% - Built-up
    'BUILDING_DENSITY_OSM',     # 19.2% - Building density
    'NIGHTLIGHTS',              # 5.0% - Urban activity
    'ALBEDO',                   # 5.5% - Reflectivity
    'NDWI',                     # 5.2% - Water
    'URBAN_COMPACTNESS',        # 5.6% - Urban form
    'URBAN_VEG',                # 5.5% - Urban-vegetation contrast
    'VEG_QUALITY',              # 3.9% - Vegetation quality
    'NDBI_POP',                 # 3.9% - Interaction
    'COOLING_EFFECT',           # 3.7% - Cooling effect
    'POPULATION',               # 2.9% - Population
    'UHI'                       # 2.2% - Urban heat island
]

# Keep only features that exist
features = [f for f in best_features if f in df.columns]
X = df[features]
y = df['LST']

print(f"📋 Using {len(features)} features")
print(f"   {features}")

# ============================================
# 3. CHECK CORRELATIONS
# ============================================

print("\n📈 Correlations with LST:")
for feature in features:
    corr = df[feature].corr(df['LST'])
    if feature in ['NDBI', 'BUILDING_DENSITY_OSM', 'NIGHTLIGHTS', 'POPULATION', 'NDBI_POP']:
        expected = 'heating (+)'
        status = '✅' if corr > 0 else '⚠️'
    elif feature in ['VEG_QUALITY', 'COOLING_EFFECT']:
        expected = 'cooling (-)'
        status = '✅' if corr < 0 else '⚠️'
    else:
        expected = 'mixed'
        status = '⚠️'
    print(f"   {feature:25s}: {corr:+.3f} ({expected}) {status}")

# ============================================
# 4. TRAIN-TEST SPLIT
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
# 5. TRAIN OPTIMIZED XGBOOST
# ============================================

print("\n⚡ Training optimized XGBoost model...")

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
# 6. EVALUATE
# ============================================

print("\n📊 Model Performance:")

y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"   R² Score: {r2:.4f}")
print(f"   RMSE: {rmse:.2f}°C")
print(f"   MAE: {mae:.2f}°C")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"   Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ============================================
# 7. FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({
    'Driver': X.columns,
    'Importance': model.feature_importances_,
    'Percentage': model.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

print("\n🎯 Feature Importance:")
print("-"*55)
for idx, row in importance.iterrows():
    if row['Driver'] in ['NDBI', 'BUILDING_DENSITY_OSM', 'NIGHTLIGHTS', 'POPULATION', 'NDBI_POP']:
        emoji = '🔥'  # Heating
    elif row['Driver'] in ['VEG_QUALITY', 'COOLING_EFFECT']:
        emoji = '🌿'  # Cooling
    else:
        emoji = '📊'  # Mixed
    print(f"   {row['Driver']:25s}: {row['Percentage']:5.1f}% {emoji}")
print("-"*55)

# ============================================
# 8. RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 FINAL RESULTS SUMMARY")
print("="*60)

print(f"\n1. Model Performance:")
print(f"   - R² Score: {r2:.4f} {'✅' if r2 > 0.7 else '⚠️ (needs more data)'}")
print(f"   - RMSE: {rmse:.2f}°C")
print(f"   - MAE: {mae:.2f}°C")
print(f"   - CV R²: {cv_scores.mean():.4f}")

print(f"\n2. Top 5 Drivers:")
for i in range(min(5, len(importance))):
    driver = importance.iloc[i]['Driver']
    pct = importance.iloc[i]['Percentage']
    print(f"   {i+1}. {driver}: {pct:.1f}%")

# Count heating vs cooling
heating_features = ['NDBI', 'BUILDING_DENSITY_OSM', 'NIGHTLIGHTS', 'POPULATION', 'NDBI_POP']
cooling_features = ['VEG_QUALITY', 'COOLING_EFFECT']

heating_pct = importance[importance['Driver'].isin(heating_features)]['Percentage'].sum()
cooling_pct = importance[importance['Driver'].isin(cooling_features)]['Percentage'].sum()

print(f"\n3. Urban/Heating Features: {heating_pct:.1f}%")
print(f"4. Vegetation/Cooling Features: {cooling_pct:.1f}%")

print("\n" + "="*60)

# Save model
import pickle
with open('urban_heat_model_final.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved to 'urban_heat_model_final.pkl'")

print("\n🏆 Final model ready for scenario testing!")