import rasterio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("📂 READING UNIFIED GEOTIFF & TRAINING MODEL")
print("="*60)

# ============================================
# 1. READ THE GEOTIFF
# ============================================

file_path = r"D:\urban_heat_mitigation\delhi_urban_heat_complete.tif"

print(f"\n📂 Reading: {file_path}")

try:
    src = rasterio.open(file_path)
    print(f"✅ File opened successfully")
    print(f"   Shape: {src.shape}")
    print(f"   Bands: {src.count}")
    
    # Read all bands
    data = src.read()
    src.close()
    
    # Band names
    band_names = ['LST', 'NDVI', 'NDBI', 'NDWI', 'ALBEDO', 
                  'POPULATION', 'NIGHTLIGHTS', 'BUILDING_DENSITY', 
                  'UHI', 'VEG_QUALITY', 'URBAN_COMPACTNESS']
    
    print(f"   Bands: {band_names[:src.count]}")
    
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
    print("   Please download the file from Google Drive first")
    print("   Update the file path if needed")
    exit()

# ============================================
# 2. CONVERT TO DATAFRAME
# ============================================

print("\n📊 Converting to DataFrame...")

n_bands, height, width = data.shape
df = pd.DataFrame()

for i, band_name in enumerate(band_names[:n_bands]):
    flattened = data[i].flatten()
    df[band_name] = flattened

# Remove invalid values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Filter for realistic values
df = df[(df['LST'] > 20) & (df['LST'] < 50)]
df = df[(df['NDVI'] > -0.5) & (df['NDVI'] < 1.0)]
df = df[(df['NDBI'] > -0.5) & (df['NDBI'] < 0.5)]
df = df[df['POPULATION'] > 0]  # Remove no-population pixels

print(f"✅ Data extracted! {len(df)} samples collected")

# ============================================
# 3. SAVE CSV
# ============================================

df.to_csv('delhi_urban_heat_unified.csv', index=False)
print("✅ Data saved to 'delhi_urban_heat_unified.csv'")

# ============================================
# 4. CORRELATIONS
# ============================================

print("\n📈 Correlations with LST (sorted):")
correlations = df.corr()['LST'].sort_values(ascending=False)
print(correlations)

# ============================================
# 5. PREPARE FEATURES
# ============================================

print("\n📋 Preparing features...")

# Features (all except LST)
X = df.drop('LST', axis=1)
y = df['LST']

print(f"   Features: {X.columns.tolist()}")
print(f"   Total features: {len(X.columns)}")

# ============================================
# 6. TRAIN-TEST SPLIT
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
# 7. TRAIN RANDOM FOREST
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
# 8. EVALUATE
# ============================================

print("\n📊 Model Performance:")

y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"   R² Score: {r2:.4f}")
print(f"   RMSE: {rmse:.2f}°C")
print(f"   MAE: {mae:.2f}°C")

# Feature importance
importance = pd.DataFrame({
    'Driver': X.columns,
    'Importance': model.feature_importances_,
    'Percentage': model.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

print("\n🎯 Feature Importance:")
print("-"*50)
for idx, row in importance.iterrows():
    print(f"   {row['Driver']:20s}: {row['Percentage']:.1f}%")
print("-"*50)

# ============================================
# 9. PHYSICS VERIFICATION
# ============================================

print("\n🔬 Physics Verification:")

# Check NDVI direction (should be negative/cooling)
ndvi_corr = df['NDVI'].corr(df['LST'])
print(f"   NDVI vs LST: {ndvi_corr:.3f} {'✅ (cooling)' if ndvi_corr < 0 else '⚠️ (wrong direction)'}")

# Check NDBI direction (should be positive/heating)
ndbi_corr = df['NDBI'].corr(df['LST'])
print(f"   NDBI vs LST: {ndbi_corr:.3f} {'✅ (heating)' if ndbi_corr > 0 else '⚠️ (wrong direction)'}")

# Check ALBEDO direction (should be negative/cooling)
albedo_corr = df['ALBEDO'].corr(df['LST'])
print(f"   ALBEDO vs LST: {albedo_corr:.3f} {'✅ (cooling)' if albedo_corr < 0 else '⚠️ (wrong direction)'}")

# Check BUILDING_DENSITY direction (should be positive/heating)
bd_corr = df['BUILDING_DENSITY'].corr(df['LST'])
print(f"   BUILDING_DENSITY vs LST: {bd_corr:.3f} {'✅ (heating)' if bd_corr > 0 else '⚠️ (wrong direction)'}")

# ============================================
# 10. VISUALIZATIONS
# ============================================

print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Feature Importance
ax1 = axes[0, 0]
top_features = importance.head(8)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
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

# Correlation Heatmap
ax4 = axes[1, 1]
# Select top features for heatmap
top_features_names = importance.head(8)['Driver'].tolist() + ['LST']
corr_matrix = df[top_features_names].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, 
            ax=ax4, cbar_kws={'label': 'Correlation'})
ax4.set_title('Top 8 Features Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('unified_model_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'unified_model_results.png'")
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

print(f"\n2. Physics Check:")
print(f"   - NDVI (cooling): {'✅' if ndvi_corr < 0 else '❌'}")
print(f"   - NDBI (heating): {'✅' if ndbi_corr > 0 else '❌'}")
print(f"   - ALBEDO (cooling): {'✅' if albedo_corr < 0 else '❌'}")
print(f"   - BUILDING_DENSITY (heating): {'✅' if bd_corr > 0 else '❌'}")

print(f"\n3. Top 5 Drivers:")
for i in range(min(5, len(importance))):
    driver = importance.iloc[i]['Driver']
    pct = importance.iloc[i]['Percentage']
    print(f"   {i+1}. {driver}: {pct:.1f}%")

print("\n" + "="*60)
print("✅ Analysis Complete!")

# Save model for later use
import pickle
with open('urban_heat_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved to 'urban_heat_model.pkl'")