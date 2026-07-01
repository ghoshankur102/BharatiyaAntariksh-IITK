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
print("🤖 IMPROVED MODEL - Removing Problematic Features")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================

print("\n📊 Loading data...")
df = pd.read_csv('delhi_urban_heat_unified.csv')
print(f"✅ Loaded {len(df)} samples")

# ============================================
# 2. FEATURE ENGINEERING (Improved)
# ============================================

print("\n🔧 Engineering better features...")

# Drop ALBEDO (wrong direction) and NDWI (wrong direction)
# Keep only features that make physical sense

# 1. Combine NDVI and VEG_QUALITY into single vegetation feature
df['VEGETATION'] = df['NDVI'] + df['VEG_QUALITY']

# 2. Combine UHI and NDBI into urban heat feature
df['URBAN_HEAT'] = df['NDBI'] + df['UHI']

# 3. Urban intensity (combines NDBI and building density)
df['URBAN_INTENSITY'] = df['NDBI'] * df['BUILDING_DENSITY']

# 4. Cooling potential (vegetation vs urban)
df['COOLING_POTENTIAL'] = df['VEGETATION'] - df['URBAN_INTENSITY']

print("✅ Added 4 new features")

# ============================================
# 3. SELECT FEATURES
# ============================================

# Keep only the best features
features_to_keep = [
    'NDBI',              # Built-up (heating) ✅
    'POPULATION',        # Human activity (heating) ✅
    'NIGHTLIGHTS',       # Urban activity (heating) ✅
    'VEGETATION',        # Combined vegetation (cooling) ✅
    'URBAN_INTENSITY',   # Combined urban (heating) ✅
    'COOLING_POTENTIAL'  # Net cooling effect ✅
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
    direction = '✅' if (feature in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'URBAN_INTENSITY'] and corr > 0) or \
                         (feature in ['VEGETATION', 'COOLING_POTENTIAL'] and corr < 0) else '⚠️'
    print(f"   {feature:20s}: {corr:+.3f} {direction}")

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
    n_estimators=200,
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
print("-"*50)
for idx, row in importance.iterrows():
    direction = '✅' if (row['Driver'] in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'URBAN_INTENSITY']) else '🌿'
    print(f"   {row['Driver']:20s}: {row['Percentage']:.1f}% {direction}")
print("-"*50)

# ============================================
# 9. VISUALIZATION
# ============================================

print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Feature Importance
ax1 = axes[0]
colors = ['#d7191c' if d in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'URBAN_INTENSITY'] 
          else '#2c7bb6' for d in importance['Driver']]
bars = ax1.barh(importance['Driver'], importance['Percentage'], color=colors)
ax1.set_title(f'Feature Importance (R² = {r2:.3f})', fontsize=14, fontweight='bold')
ax1.set_xlabel('Contribution to Heat (%)')
ax1.set_ylabel('Driver')
ax1.grid(True, alpha=0.3)

for bar, pct in zip(bars, importance['Percentage']):
    width = bar.get_width()
    ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{pct:.1f}%', ha='left', va='center', fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d7191c', label='Heating (Urban)'),
    Patch(facecolor='#2c7bb6', label='Cooling (Vegetation)')
]
ax1.legend(handles=legend_elements, loc='lower right')

# Predicted vs Actual
ax2 = axes[1]
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

plt.tight_layout()
plt.savefig('improved_model_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'improved_model_results.png'")
plt.show()

# ============================================
# 10. RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 FINAL RESULTS SUMMARY")
print("="*60)
print(f"\n1. Model Performance:")
print(f"   - R² Score: {r2:.4f} {'✅' if r2 > 0.7 else '⚠️ (needs improvement)'}")
print(f"   - RMSE: {rmse:.2f}°C")
print(f"   - MAE: {mae:.2f}°C")

print(f"\n2. Top Drivers:")
for i in range(min(3, len(importance))):
    driver = importance.iloc[i]['Driver']
    pct = importance.iloc[i]['Percentage']
    print(f"   {i+1}. {driver}: {pct:.1f}%")

print(f"\n3. Key Insights:")
print(f"   - Urban features (NDBI, Nightlights, Population): {importance[importance['Driver'].isin(['NDBI', 'NIGHTLIGHTS', 'POPULATION'])]['Percentage'].sum():.1f}%")
print(f"   - Vegetation cooling: {importance[importance['Driver'] == 'COOLING_POTENTIAL']['Percentage'].values[0]:.1f}%")

print("\n" + "="*60)

# ============================================
# 11. SAVE MODEL
# ============================================

import pickle
with open('urban_heat_model_improved.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved to 'urban_heat_model_improved.pkl'")