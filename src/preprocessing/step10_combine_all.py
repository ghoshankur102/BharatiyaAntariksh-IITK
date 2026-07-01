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
print("🔄 COMBINING ALL DATA FOR MODEL TRAINING")
print("="*60)

# ============================================
# 1. LOAD ALL DATA
# ============================================

print("\n📊 Loading all datasets...")

# Load main dataset (LST and indices)
df_main = pd.read_csv('delhi_urban_heat_final.csv')
print(f"✅ Main dataset: {len(df_main)} samples")

# Load building density
try:
    df_buildings = pd.read_csv('building_density.csv')
    print(f"✅ Building density: {len(df_buildings)} samples")
except:
    print("⚠️ Building density not found, using zeros")
    df_buildings = pd.DataFrame({'building_density': np.zeros(len(df_main))})

# Load population density
try:
    df_population = pd.read_csv('population_density.csv')
    print(f"✅ Population density: {len(df_population)} samples")
except:
    print("⚠️ Population density not found, using zeros")
    df_population = pd.DataFrame({'population_density': np.zeros(len(df_main))})

# Load nighttime lights
try:
    df_lights = pd.read_csv('nighttime_lights.csv')
    print(f"✅ Nighttime lights: {len(df_lights)} samples")
except:
    print("⚠️ Nighttime lights not found, using zeros")
    df_lights = pd.DataFrame({'nightlights': np.zeros(len(df_main))})

# ============================================
# 2. ALIGN DATA
# ============================================

print("\n🔗 Aligning data...")

# Get the number of samples
n_samples = min(len(df_main), len(df_buildings), len(df_population), len(df_lights))
print(f"   Using {n_samples} samples")

# Truncate all datasets to same length
df_main = df_main.iloc[:n_samples]
df_buildings = df_buildings.iloc[:n_samples]
df_population = df_population.iloc[:n_samples]
df_lights = df_lights.iloc[:n_samples]

# Combine all features
df_combined = pd.DataFrame({
    'LST': df_main['LST'].values,
    'NDVI': df_main['NDVI'].values,
    'NDBI': df_main['NDBI'].values,
    'NDWI': df_main['NDWI'].values,
    'ALBEDO': df_main['ALBEDO'].values,
    'building_density': df_buildings['building_density'].values,
    'population_density': df_population['population_density'].values,
    'nightlights': df_lights['nightlights'].values
})

# Remove NaN values
df_combined = df_combined.dropna()
print(f"✅ Combined data: {len(df_combined)} samples")

# ============================================
# 3. PREPARE FEATURES
# ============================================

print("\n📋 Preparing features...")

X = df_combined.drop('LST', axis=1)
y = df_combined['LST']

print(f"   Features: {X.columns.tolist()}")
print(f"   Total features: {len(X.columns)}")

# Check correlations
print("\n📈 Correlations with LST (sorted):")
correlations = df_combined.corr()['LST'].sort_values(ascending=False)
print(correlations)

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
# 5. TRAIN MODEL
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
# 7. VISUALIZATIONS
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
corr_matrix = df_combined.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, 
            ax=ax4, cbar_kws={'label': 'Correlation'})
ax4.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('combined_model_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'combined_model_results.png'")
plt.show()

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

print(f"\n2. Top 5 Drivers of Urban Heat:")
for i in range(min(5, len(importance))):
    driver = importance.iloc[i]['Driver']
    pct = importance.iloc[i]['Percentage']
    print(f"   {i+1}. {driver}: {pct:.1f}%")

print(f"\n3. Key Insights:")
print(f"   - {importance.iloc[0]['Driver']} is the primary driver")
if 'NDVI' in importance['Driver'].values:
    ndvi_pct = importance[importance['Driver'] == 'NDVI']['Percentage'].values[0]
    print(f"   - NDVI (vegetation) shows {ndvi_pct:.1f}% importance (cooling)")
if 'building_density' in importance['Driver'].values:
    bd_pct = importance[importance['Driver'] == 'building_density']['Percentage'].values[0]
    print(f"   - Building density shows {bd_pct:.1f}% importance (heating)")

print("\n" + "="*60)
print("✅ Analysis Complete!")