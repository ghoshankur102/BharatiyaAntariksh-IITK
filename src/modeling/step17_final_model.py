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
print("🏆 FINAL MODEL - With Building Data")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================

print("\n📊 Loading data...")

# Try to load with OSM buildings first
try:
    df = pd.read_csv('delhi_urban_heat_with_osm_buildings.csv')
    print(f"✅ Loaded with OSM buildings: {len(df)} samples")
    has_buildings = True
except:
    # Fall back to unified data
    df = pd.read_csv('delhi_urban_heat_unified.csv')
    print(f"✅ Loaded unified data: {len(df)} samples")
    # Create building density from NDBI
    df['BUILDING_DENSITY_OSM'] = df['NDBI'] * 10
    has_buildings = False

print(f"   Features: {df.columns.tolist()}")

# ============================================
# 2. FEATURE ENGINEERING
# ============================================

print("\n🔧 Engineering features...")

# Check what features we have
if 'BUILDING_DENSITY_OSM' in df.columns:
    building_feature = 'BUILDING_DENSITY_OSM'
elif 'BUILDING_DENSITY' in df.columns:
    building_feature = 'BUILDING_DENSITY'
else:
    # Create from NDBI
    df['BUILDING_DENSITY_OSM'] = df['NDBI'] * 10
    building_feature = 'BUILDING_DENSITY_OSM'

# Ensure we have all needed features
required_features = ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'ALBEDO', 'NDWI', 'UHI', 'URBAN_COMPACTNESS', 'VEG_QUALITY']
for feature in required_features:
    if feature not in df.columns:
        print(f"   ⚠️ Missing {feature}, creating from available data...")
        if feature == 'POPULATION':
            df['POPULATION'] = 1000  # Default value
        elif feature == 'NIGHTLIGHTS':
            df['NIGHTLIGHTS'] = 0.1
        elif feature == 'ALBEDO':
            df['ALBEDO'] = df['NDBI'] * 0.5
        elif feature == 'NDWI':
            df['NDWI'] = -df['NDVI'] * 0.5
        elif feature == 'UHI':
            df['UHI'] = df['NDBI'] - df['NDVI']
        elif feature == 'URBAN_COMPACTNESS':
            df['URBAN_COMPACTNESS'] = df['NDBI'] ** 2
        elif feature == 'VEG_QUALITY':
            df['VEG_QUALITY'] = df['NDVI'] * (1 - df['NDBI'])

# Now build features
features = [
    'NDBI',              # Built-up (heating) ✅
    'POPULATION',        # Human activity (heating) ✅
    'NIGHTLIGHTS',       # Urban activity (heating) ✅
    'ALBEDO',            # Keep but watch direction
    'NDWI',              # Keep but watch direction
    'UHI',               # Urban Heat Island
    'URBAN_COMPACTNESS', # Urban density
    'VEG_QUALITY',       # Vegetation quality (cooling) ✅
    building_feature     # Building data ✅
]

# Create interaction features
df['NDBI_POP'] = df['NDBI'] * df['POPULATION'] / 10000
df['NDBI_LIGHTS'] = df['NDBI'] * df['NIGHTLIGHTS']
df['URBAN_VEG'] = df['NDBI'] - df['VEG_QUALITY']
df['COOLING_EFFECT'] = df['VEG_QUALITY'] * (1 - df['NDBI'])
df['BUILDING_DENSITY_INDEX'] = df[building_feature] * df['NDBI']

# Add to features
features.extend(['NDBI_POP', 'NDBI_LIGHTS', 'URBAN_VEG', 'COOLING_EFFECT', 'BUILDING_DENSITY_INDEX'])

X = df[features]
y = df['LST']

print(f"📋 Total features: {len(features)}")

# ============================================
# 3. CHECK CORRELATIONS
# ============================================

print("\n📈 Correlations with LST:")
for feature in features:
    if feature in df.columns:
        corr = df[feature].corr(df['LST'])
        if feature in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'NDBI_POP', 'NDBI_LIGHTS', 'URBAN_VEG', building_feature, 'BUILDING_DENSITY_INDEX']:
            expected = 'positive (+)'
            status = '✅' if corr > 0 else '⚠️'
        elif feature in ['VEG_QUALITY', 'COOLING_EFFECT']:
            expected = 'negative (-)'
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
# 5. TRAIN XGBOOST
# ============================================

print("\n⚡ Training XGBoost model...")

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
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
    if row['Driver'] in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'NDBI_POP', 'NDBI_LIGHTS', 'URBAN_VEG', building_feature, 'BUILDING_DENSITY_INDEX']:
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
print(f"   - R² Score: {r2:.4f} {'✅' if r2 > 0.7 else '⚠️ (needs improvement)'}")
print(f"   - RMSE: {rmse:.2f}°C")
print(f"   - MAE: {mae:.2f}°C")
print(f"   - CV R²: {cv_scores.mean():.4f}")

print(f"\n2. Top 5 Drivers:")
for i in range(min(5, len(importance))):
    driver = importance.iloc[i]['Driver']
    pct = importance.iloc[i]['Percentage']
    print(f"   {i+1}. {driver}: {pct:.1f}%")

# Check building importance
building_importance = importance[importance['Driver'] == building_feature]
if len(building_importance) > 0:
    building_pct = building_importance.iloc[0]['Percentage']
    print(f"\n3. {building_feature} Importance: {building_pct:.1f}%")

print("\n" + "="*60)
print("✅ Analysis Complete!")

# Save model
import pickle
with open('final_urban_heat_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved to 'final_urban_heat_model.pkl'")

# ============================================
# 9. VISUALIZATION
# ============================================

print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Feature Importance
ax1 = axes[0, 0]
top_features = importance.head(8)
colors = ['#d7191c' if any(x in f for x in ['NDBI', 'POP', 'LIGHTS', 'URBAN', building_feature]) 
          else '#2c7bb6' for f in top_features['Driver']]
bars = ax1.barh(top_features['Driver'], top_features['Percentage'], color=colors)
ax1.set_title(f'Feature Importance (R² = {r2:.3f})', fontsize=14, fontweight='bold')
ax1.set_xlabel('Contribution to Heat (%)')
ax1.set_ylabel('Driver')
ax1.grid(True, alpha=0.3)

for bar, pct in zip(bars, top_features['Percentage']):
    width = bar.get_width()
    ax1.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
             f'{pct:.1f}%', ha='left', va='center', fontweight='bold')

# Predicted vs Actual
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred, alpha=0.3, s=10, c='steelblue')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Temperature (°C)')
ax2.set_ylabel('Predicted Temperature (°C)')
ax2.set_title('XGBoost Performance', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

text = f'R² = {r2:.3f}\nRMSE = {rmse:.2f}°C\nMAE = {mae:.2f}°C'
ax2.text(0.05, 0.95, text, transform=ax2.transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Residuals
ax3 = axes[1, 0]
residuals = y_pred - y_test
ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Residual (°C)')
ax3.set_ylabel('Frequency')
ax3.set_title('Residual Distribution', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Feature-Temperature Relationship (Top 3)
ax4 = axes[1, 1]
top3 = importance.head(3)['Driver'].tolist()
for feature in top3:
    if feature in df.columns:
        ax4.scatter(df[feature], df['LST'], alpha=0.15, s=5, label=feature)
ax4.set_xlabel('Feature Value')
ax4.set_ylabel('LST (°C)')
ax4.set_title('Top 3 Drivers vs Temperature', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_model_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'final_model_results.png'")
plt.show()

print("\n🏆 Final model ready for scenario testing!")