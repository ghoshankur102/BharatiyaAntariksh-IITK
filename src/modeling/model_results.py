"""
plot_final_correlation.py
Replicates the exact feature engineering and scaling from step17_final_model.py
to generate the correct Actual vs Predicted scatter plot.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pickle

# ================================================================
# 1. LOAD DATA (exact same as step17)
# ================================================================

print("📊 Loading data...")
try:
    df = pd.read_csv('delhi_urban_heat_with_osm_buildings.csv')
    print(f"✅ Loaded with OSM buildings: {len(df)} samples")
    has_buildings = True
except:
    df = pd.read_csv('delhi_urban_heat_unified.csv')
    print(f"✅ Loaded unified data: {len(df)} samples")
    df['BUILDING_DENSITY_OSM'] = df['NDBI'] * 10
    has_buildings = False

# Determine building feature name
if 'BUILDING_DENSITY_OSM' in df.columns:
    building_feature = 'BUILDING_DENSITY_OSM'
elif 'BUILDING_DENSITY' in df.columns:
    building_feature = 'BUILDING_DENSITY'
else:
    df['BUILDING_DENSITY_OSM'] = df['NDBI'] * 10
    building_feature = 'BUILDING_DENSITY_OSM'

# ================================================================
# 2. FEATURE ENGINEERING (exact copy from step17)
# ================================================================

print("🔧 Engineering features (exact step17 formulas)...")

# Ensure base features exist
required = ['NDBI', 'POPULATION', 'NIGHTLIGHTS', 'ALBEDO', 'NDWI', 'UHI', 'URBAN_COMPACTNESS', 'VEG_QUALITY']
for f in required:
    if f not in df.columns:
        # fallback (as in step17)
        if f == 'POPULATION': df[f] = 1000
        elif f == 'NIGHTLIGHTS': df[f] = 0.1
        elif f == 'ALBEDO': df[f] = df['NDBI'] * 0.5
        elif f == 'NDWI': df[f] = -df['NDVI'] * 0.5 if 'NDVI' in df.columns else 0
        elif f == 'UHI': df[f] = df['NDBI'] - df['NDVI'] if 'NDVI' in df.columns else 0
        elif f == 'URBAN_COMPACTNESS': df[f] = df['NDBI'] ** 2
        elif f == 'VEG_QUALITY': df[f] = df['NDVI'] * (1 - df['NDBI']) if 'NDVI' in df.columns else 0

# Create features (EXACT formulas from step17)
df['NDBI_POP'] = df['NDBI'] * df['POPULATION'] / 10000
df['NDBI_LIGHTS'] = df['NDBI'] * df['NIGHTLIGHTS']
df['URBAN_VEG'] = df['NDBI'] - df['VEG_QUALITY']
df['COOLING_EFFECT'] = df['VEG_QUALITY'] * (1 - df['NDBI'])
df['BUILDING_DENSITY_INDEX'] = df[building_feature] * df['NDBI']

# Feature list (exactly as in step17)
features = [
    'NDBI', 'POPULATION', 'NIGHTLIGHTS', 'ALBEDO', 'NDWI', 'UHI',
    'URBAN_COMPACTNESS', 'VEG_QUALITY', building_feature,
    'NDBI_POP', 'NDBI_LIGHTS', 'URBAN_VEG', 'COOLING_EFFECT',
    'BUILDING_DENSITY_INDEX'
]

X = df[features]
y = df['LST']

print(f"✅ Engineered {len(features)} features")

# ================================================================
# 3. TRAIN/TEST SPLIT (same as step17)
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================================================
# 4. SCALE (re‑fit scaler on training set, like step17)
# ================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Test set size: {len(X_test)} samples")

# ================================================================
# 5. LOAD THE SAVED MODEL
# ================================================================

print("📂 Loading saved model...")
with open('final_urban_heat_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ================================================================
# 6. PREDICT AND COMPUTE METRICS (dynamic)
# ================================================================

y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
pearson_r, _ = pearsonr(y_test, y_pred)

print("\n" + "="*50)
print("📈 FINAL MODEL PERFORMANCE (on test set)")
print("="*50)
print(f"   Pearson's r = {pearson_r:.4f}")
print(f"   R²         = {r2:.4f}")
print(f"   RMSE       = {rmse:.3f} °C")
print(f"   MAE        = {mae:.3f} °C")
print("="*50 + "\n")

# ================================================================
# 7. CREATE THE SCATTER PLOT
# ================================================================

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(y_test, y_pred, alpha=0.5, s=12, c='#1f77b4', edgecolors='none', label='Predictions')

# 1:1 perfect line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, alpha=0.6, label='1:1 Perfect Line')

# Regression fit (to show bias)
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
ax.plot(y_test, p(y_test), "r-", lw=2, alpha=0.8, label=f'Regression (y={z[0]:.2f}x + {z[1]:.2f})')

# Metrics box (fully dynamic)
textstr = (
    f'$r$ (Pearson) = {pearson_r:.4f}\n'
    f'$R^2$ = {r2:.4f}\n'
    f'RMSE = {rmse:.2f} °C\n'
    f'MAE  = {mae:.2f} °C'
)
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#333333')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
        verticalalignment='top', bbox=props, family='monospace')

ax.set_xlabel('Actual LST (°C)', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted LST (°C)', fontsize=14, fontweight='bold')
ax.set_title('Delhi Urban Heat: Final Model Validation\n(Observed vs Predicted)', fontsize=16, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='lower right', fontsize=10)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('final_correlation.png', dpi=300, bbox_inches='tight')
print("✅ Figure saved as 'final_correlation.png'")

plt.show()