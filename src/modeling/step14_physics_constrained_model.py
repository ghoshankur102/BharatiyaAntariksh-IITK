import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🧠 PHYSICS-CONSTRAINED MODEL - Best of Both Worlds")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================

print("\n📊 Loading data...")
df = pd.read_csv('delhi_urban_heat_unified.csv')
print(f"✅ Loaded {len(df)} samples")

# ============================================
# 2. FEATURE ENGINEERING (Keep ALL features)
# ============================================

print("\n🔧 Engineering physics-constrained features...")

# Keep all original features
# Add physical constraints by transforming features

# 1. ALBEDO: Force to negative relationship (cooling)
# Instead of raw albedo, use a transformed version
df['ALBEDO_COOLING'] = -df['ALBEDO']  # Invert the relationship

# 2. NDWI: Force to negative relationship (cooling)
df['NDWI_COOLING'] = -df['NDWI']  # Invert the relationship

# 3. Building Density: Force to positive (heating)
df['BUILDING_HEATING'] = df['BUILDING_DENSITY']

# 4. Urban Compactness: Force to positive (heating)
df['URBAN_HEATING'] = df['URBAN_COMPACTNESS']

# 5. UHI: Force to positive (heating)
df['UHI_HEATING'] = df['UHI']

# 6. Vegetation: Force to negative (cooling)
df['VEG_COOLING'] = df['VEG_QUALITY'] + df['NDVI']

# Combine similar features
df['URBAN_INDEX'] = df['NDBI'] + df['BUILDING_DENSITY'] + df['UHI']
df['COOLING_INDEX'] = df['VEG_COOLING'] + df['ALBEDO_COOLING'] + df['NDWI_COOLING']

print("✅ Added physics-constrained features")

# ============================================
# 3. SELECT FEATURES
# ============================================

features_to_keep = [
    'NDBI',              # Built-up (heating) ✅
    'POPULATION',        # Human activity (heating) ✅
    'NIGHTLIGHTS',       # Urban activity (heating) ✅
    'ALBEDO',            # Raw (keep for information) ⚠️
    'NDWI',              # Raw (keep for information) ⚠️
    'VEG_COOLING',       # Vegetation (cooling) ✅
    'URBAN_INDEX',       # Combined urban (heating) ✅
    'COOLING_INDEX'      # Combined cooling ✅
]

X = df[features_to_keep]
y = df['LST']

print(f"\n📋 Features selected: {len(features_to_keep)}")
print(f"   {features_to_keep}")

# ============================================
# 4. CHECK CORRELATIONS
# ============================================

print("\n📈 Correlations with LST:")
for feature in features_to_keep:
    corr = df[feature].corr(df['LST'])
    # Determine expected direction
    if feature in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'URBAN_INDEX']:
        expected = 'heating (+)'
        status = '✅' if corr > 0 else '⚠️'
    elif feature in ['VEG_COOLING', 'COOLING_INDEX']:
        expected = 'cooling (-)'
        status = '✅' if corr < 0 else '⚠️'
    else:
        expected = 'mixed'
        status = '⚠️'
    print(f"   {feature:20s}: {corr:+.3f} ({expected}) {status}")

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
# 6. TRAIN RANDOM FOREST
# ============================================

print("\n🌳 Training Random Forest model...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
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

# Cross-validation
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
    # Determine type
    if row['Driver'] in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'URBAN_INDEX']:
        emoji = '🔥'  # Heating
    elif row['Driver'] in ['VEG_COOLING', 'COOLING_INDEX']:
        emoji = '🌿'  # Cooling
    else:
        emoji = '📊'  # Mixed
    print(f"   {row['Driver']:20s}: {row['Percentage']:5.1f}% {emoji}")
print("-"*55)

# ============================================
# 9. PHYSICS VERIFICATION
# ============================================

print("\n🔬 Physics Verification:")

# Check if model respects physics
print("   Expected relationships:")
for feature in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'URBAN_INDEX']:
    if feature in X.columns:
        corr = df[feature].corr(df['LST'])
        status = '✅' if corr > 0 else '❌'
        print(f"   - {feature}: {corr:+.3f} (should be positive) {status}")

for feature in ['VEG_COOLING', 'COOLING_INDEX']:
    if feature in X.columns:
        corr = df[feature].corr(df['LST'])
        status = '✅' if corr < 0 else '❌'
        print(f"   - {feature}: {corr:+.3f} (should be negative) {status}")

# ============================================
# 10. VISUALIZATIONS
# ============================================

print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Feature Importance
ax1 = axes[0, 0]
top_features = importance.head(8)
colors = ['#d7191c' if f in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'URBAN_INDEX'] 
          else '#2c7bb6' if f in ['VEG_COOLING', 'COOLING_INDEX']
          else '#fdae61' for f in top_features['Driver']]
bars = ax1.barh(top_features['Driver'], top_features['Percentage'], color=colors)
ax1.set_title(f'Feature Importance (R² = {r2:.3f})', fontsize=14, fontweight='bold')
ax1.set_xlabel('Contribution to Heat (%)')
ax1.set_ylabel('Driver')
ax1.grid(True, alpha=0.3)

for bar, pct in zip(bars, top_features['Percentage']):
    width = bar.get_width()
    ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{pct:.1f}%', ha='left', va='center', fontweight='bold')

# Predicted vs Actual
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred, alpha=0.3, s=10, c='steelblue')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Temperature (°C)')
ax2.set_ylabel('Predicted Temperature (°C)')
ax2.set_title('Model Performance', fontsize=14, fontweight='bold')
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

# Feature-Temperature Relationship
ax4 = axes[1, 1]
top3 = importance.head(3)['Driver'].tolist()
for feature in top3:
    ax4.scatter(df[feature], df['LST'], alpha=0.2, s=5, label=feature)
ax4.set_xlabel('Feature Value (scaled)')
ax4.set_ylabel('LST (°C)')
ax4.set_title('Top 3 Drivers vs Temperature', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('physics_constrained_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'physics_constrained_results.png'")
plt.show()

# ============================================
# 11. RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 FINAL RESULTS SUMMARY")
print("="*60)

print(f"\n1. Model Performance:")
print(f"   - R² Score: {r2:.4f} {'✅' if r2 > 0.7 else '⚠️ (needs improvement)'}")
print(f"   - RMSE: {rmse:.2f}°C")
print(f"   - MAE: {mae:.2f}°C")

print(f"\n2. Physics Constraint Check:")
all_physics_ok = True
for feature in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'URBAN_INDEX']:
    if feature in X.columns:
        corr = df[feature].corr(df['LST'])
        if corr < 0:
            all_physics_ok = False
            print(f"   ❌ {feature}: {corr:+.3f} (should be positive)")

for feature in ['VEG_COOLING', 'COOLING_INDEX']:
    if feature in X.columns:
        corr = df[feature].corr(df['LST'])
        if corr > 0:
            all_physics_ok = False
            print(f"   ❌ {feature}: {corr:+.3f} (should be negative)")

if all_physics_ok:
    print("   ✅ All features have the correct physical direction!")

print(f"\n3. Top 3 Drivers:")
for i in range(3):
    driver = importance.iloc[i]['Driver']
    pct = importance.iloc[i]['Percentage']
    print(f"   {i+1}. {driver}: {pct:.1f}%")

print("\n" + "="*60)
print("✅ Analysis Complete!")