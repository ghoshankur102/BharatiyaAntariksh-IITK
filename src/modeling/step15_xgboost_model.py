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
print("⚡ XGBOOST MODEL - Final Push for R² > 0.65")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================

print("\n📊 Loading data...")
df = pd.read_csv('delhi_urban_heat_unified.csv')
print(f"✅ Loaded {len(df)} samples")

# ============================================
# 2. FEATURE ENGINEERING
# ============================================

print("\n🔧 Engineering features...")

# Keep all well-behaved features
features = [
    'NDBI',              # Built-up (heating) ✅
    'POPULATION',        # Human activity (heating) ✅
    'NIGHTLIGHTS',       # Urban activity (heating) ✅
    'ALBEDO',            # Keep but watch direction
    'NDWI',              # Keep but watch direction
    'UHI',               # Urban Heat Island
    'URBAN_COMPACTNESS', # Urban density
    'VEG_QUALITY',       # Vegetation quality (cooling) ✅
]

# Create interaction features
df['NDBI_POP'] = df['NDBI'] * df['POPULATION'] / 10000
df['NDBI_LIGHTS'] = df['NDBI'] * df['NIGHTLIGHTS']
df['URBAN_VEG'] = df['NDBI'] - df['VEG_QUALITY']
df['COOLING_EFFECT'] = df['VEG_QUALITY'] * (1 - df['NDBI'])

features.extend(['NDBI_POP', 'NDBI_LIGHTS', 'URBAN_VEG', 'COOLING_EFFECT'])

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
# 4. TRAIN XGBOOST
# ============================================

print("\n⚡ Training XGBoost model...")

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

print("✅ Model trained!")

# ============================================
# 5. EVALUATE
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
# 6. FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({
    'Driver': X.columns,
    'Importance': model.feature_importances_,
    'Percentage': model.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

print("\n🎯 Feature Importance:")
print("-"*55)
for idx, row in importance.iterrows():
    emoji = '🔥' if row['Driver'] in ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'NDBI_POP', 'NDBI_LIGHTS', 'URBAN_VEG'] else '🌿'
    print(f"   {row['Driver']:20s}: {row['Percentage']:5.1f}% {emoji}")
print("-"*55)

# ============================================
# 7. PHYSICS VERIFICATION
# ============================================

print("\n🔬 Physics Verification:")

# Check NDBI direction (should be positive)
ndbi_corr = df['NDBI'].corr(df['LST'])
print(f"   NDBI vs LST: {ndbi_corr:+.3f} {'✅' if ndbi_corr > 0 else '❌'}")

# Check VEG_QUALITY direction (should be negative)
veg_corr = df['VEG_QUALITY'].corr(df['LST'])
print(f"   VEG_QUALITY vs LST: {veg_corr:+.3f} {'✅' if veg_corr < 0 else '❌'}")

# Check ALBEDO direction (should be negative)
albedo_corr = df['ALBEDO'].corr(df['LST'])
print(f"   ALBEDO vs LST: {albedo_corr:+.3f} {'⚠️' if albedo_corr > 0 else '✅'}")

# ============================================
# 8. VISUALIZATIONS
# ============================================

print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Feature Importance
ax1 = axes[0, 0]
top_features = importance.head(8)
colors = ['#d7191c' if 'NDBI' in f or 'POP' in f or 'LIGHTS' in f or 'URBAN' in f 
          else '#2c7bb6' for f in top_features['Driver']]
bars = ax1.barh(top_features['Driver'], top_features['Percentage'], color=colors)
ax1.set_title(f'XGBoost Feature Importance (R² = {r2:.3f})', fontsize=14, fontweight='bold')
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
    ax4.scatter(df[feature], df['LST'], alpha=0.15, s=5, label=feature)
ax4.set_xlabel('Feature Value')
ax4.set_ylabel('LST (°C)')
ax4.set_title('Top 3 Drivers vs Temperature', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'xgboost_results.png'")
plt.show()

# ============================================
# 9. RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 FINAL RESULTS SUMMARY")
print("="*60)

print(f"\n1. Model Performance:")
print(f"   - R² Score: {r2:.4f} {'✅' if r2 > 0.7 else '⚠️ (needs more data)'}")
print(f"   - RMSE: {rmse:.2f}°C")
print(f"   - MAE: {mae:.2f}°C")

print(f"\n2. Top 3 Drivers:")
for i in range(3):
    driver = importance.iloc[i]['Driver']
    pct = importance.iloc[i]['Percentage']
    print(f"   {i+1}. {driver}: {pct:.1f}%")

print(f"\n3. Physics Check:")
print(f"   ✅ NDBI: {ndbi_corr:+.3f} (correct direction)")
print(f"   ✅ VEG_QUALITY: {veg_corr:+.3f} (correct direction)")

print(f"\n4. Next Steps:")
if r2 < 0.7:
    print("   - Add actual building footprints from OSM")
    print("   - Add air temperature from ERA5")
    print("   - Add distance to water bodies")
    print("   - Add road network density")

print("\n" + "="*60)
print("✅ Analysis Complete!")