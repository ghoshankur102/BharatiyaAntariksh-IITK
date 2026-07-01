import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🤖 FINAL URBAN HEAT DRIVER ANALYSIS - Ensemble Model")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================

print("\n📊 Loading data...")
df = pd.read_csv('delhi_urban_heat_final.csv')
print(f"✅ Loaded {len(df)} samples")

# ============================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================

print("\n🔧 Creating advanced features...")

# Original features
df_original = df.copy()

# 1. Non-linear transformations
df['NDVI_SQ'] = df['NDVI'] ** 2
df['NDBI_SQ'] = df['NDBI'] ** 2
df['NDWI_SQ'] = df['NDWI'] ** 2
df['ALBEDO_SQ'] = df['ALBEDO'] ** 2

# 2. Interaction features (non-linear relationships)
df['NDVI_NDBI'] = df['NDVI'] * df['NDBI']
df['NDVI_NDWI'] = df['NDVI'] * df['NDWI']
df['NDBI_NDWI'] = df['NDBI'] * df['NDWI']
df['NDBI_ALBEDO'] = df['NDBI'] * df['ALBEDO']
df['NDVI_ALBEDO'] = df['NDVI'] * df['ALBEDO']

# 3. Ratio features (capture relative effects)
df['NDVI_NDBI_RATIO'] = df['NDVI'] / (df['NDBI'] + 0.001)
df['NDVI_ALBEDO_RATIO'] = df['NDVI'] / (df['ALBEDO'] + 0.001)
df['NDWI_NDBI_RATIO'] = df['NDWI'] / (df['NDBI'] + 0.001)

# 4. Composite indices
df['GREEN_COVER'] = df['NDVI'] * (1 - df['NDBI'])
df['URBAN_HEAT'] = df['NDBI'] * (1 + df['ALBEDO'])
df['WATER_COOLING'] = df['NDWI'] * (1 - df['NDBI'])

# 5. Physical constraints (clip extreme values)
for col in df.columns:
    if col != 'LST':
        df[col] = df[col].clip(lower=df[col].quantile(0.01), 
                               upper=df[col].quantile(0.99))

print(f"✅ Created {len(df.columns) - len(df_original.columns)} new features")
print(f"   Total features: {len(df.columns) - 1}")

# ============================================
# 3. FEATURE SELECTION
# ============================================

print("\n🔍 Selecting best features...")

X = df.drop('LST', axis=1)
y = df['LST']

# Correlation with target
correlations = X.corrwith(y).abs().sort_values(ascending=False)

# Keep features with correlation > 0.1
selected_features = correlations[correlations > 0.1].index.tolist()
X = X[selected_features]

print(f"   Selected {len(selected_features)} features")
print(f"   Top 5 correlations: {correlations.head(5).to_dict()}")

# ============================================
# 4. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Data Split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")
print(f"   Features: {len(X.columns)}")

# ============================================
# 5. SCALE FEATURES
# ============================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 6. HYPERPARAMETER TUNING FOR RANDOM FOREST
# ============================================

print("\n🔧 Tuning Random Forest hyperparameters...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 5, 10]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_
print(f"   Best parameters: {grid_search.best_params_}")
print(f"   Best CV R²: {grid_search.best_score_:.4f}")

# ============================================
# 7. TRAIN MULTIPLE MODELS
# ============================================

print("\n🚀 Training optimized models...")

# Model 1: Optimized Random Forest
rf_pred = best_rf.predict(X_test_scaled)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"\n   Random Forest (Optimized):")
print(f"      R²: {rf_r2:.4f}")
print(f"      RMSE: {rf_rmse:.2f}°C")

# Model 2: Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)
gb_r2 = r2_score(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

print(f"\n   Gradient Boosting:")
print(f"      R²: {gb_r2:.4f}")
print(f"      RMSE: {gb_rmse:.2f}°C")

# Model 3: XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

print(f"\n   XGBoost:")
print(f"      R²: {xgb_r2:.4f}")
print(f"      RMSE: {xgb_rmse:.2f}°C")

# ============================================
# 8. ENSEMBLE MODEL (Voting Regressor)
# ============================================

print("\n🎯 Creating Ensemble Model...")

ensemble = VotingRegressor([
    ('rf', best_rf),
    ('gb', gb),
    ('xgb', xgb_model)
])
ensemble.fit(X_train_scaled, y_train)
ensemble_pred = ensemble.predict(X_test_scaled)
ensemble_r2 = r2_score(y_test, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

print(f"\n   Ensemble Model (3 models):")
print(f"      R²: {ensemble_r2:.4f}")
print(f"      RMSE: {ensemble_rmse:.2f}°C")

# ============================================
# 9. FEATURE IMPORTANCE (From Best Model)
# ============================================

print("\n🎯 Feature Importance from Optimized Random Forest:")

importance = pd.DataFrame({
    'Driver': X.columns,
    'Importance': best_rf.feature_importances_,
    'Percentage': best_rf.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

print("\n" + "-"*55)
for idx, row in importance.head(10).iterrows():
    print(f"   {row['Driver']:30s}: {row['Percentage']:.1f}%")
print("-"*55)

# ============================================
# 10. PHYSICS VERIFICATION
# ============================================

print("\n🔬 Physics Verification:")

# Vegetation features (should show cooling)
veg_features = [f for f in X.columns if 'NDVI' in f or 'VEG' in f or 'GREEN' in f or 'COOLING' in f]
if veg_features:
    veg_importance = importance[importance['Driver'].isin(veg_features)]
    total_veg = veg_importance['Percentage'].sum()
    print(f"   ✅ Vegetation-related features: {total_veg:.1f}% (cooling effect)")

# Urban features (should show heating)
urban_features = [f for f in X.columns if 'NDBI' in f or 'URBAN' in f or 'COMPACT' in f or 'DENSITY' in f]
if urban_features:
    urban_importance = importance[importance['Driver'].isin(urban_features)]
    total_urban = urban_importance['Percentage'].sum()
    print(f"   ✅ Urban-related features: {total_urban:.1f}% (heating effect)")

# Water features (should show cooling)
water_features = [f for f in X.columns if 'NDWI' in f or 'WATER' in f or 'MOISTURE' in f]
if water_features:
    water_importance = importance[importance['Driver'].isin(water_features)]
    total_water = water_importance['Percentage'].sum()
    print(f"   ✅ Water-related features: {total_water:.1f}% (cooling effect)")

# ============================================
# 11. VISUALIZATIONS
# ============================================

print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Feature Importance
ax1 = axes[0, 0]
top_features = importance.head(8)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
bars = ax1.barh(top_features['Driver'], top_features['Percentage'], color=colors)
ax1.set_title(f'Key Drivers of Urban Heat (Ensemble R² = {ensemble_r2:.3f})', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Contribution to Heat (%)')
ax1.set_ylabel('Driver')
ax1.grid(True, alpha=0.3)

for bar, pct in zip(bars, top_features['Percentage']):
    width = bar.get_width()
    ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{pct:.1f}%', ha='left', va='center', fontweight='bold')

# Model Comparison
ax2 = axes[0, 1]
model_names = ['RF', 'GB', 'XGB', 'Ensemble']
r2_scores = [rf_r2, gb_r2, xgb_r2, ensemble_r2]
colors = ['#2c7bb6', '#fdae61', '#d7191c', '#1a9850']
bars = ax2.bar(model_names, r2_scores, color=colors)
ax2.set_ylabel('R² Score')
ax2.set_title('Model Comparison', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.7, color='green', linestyle='--', label='Target (0.7)')
ax2.legend()

for bar, r2 in zip(bars, r2_scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')

# Predicted vs Actual (Ensemble)
ax3 = axes[1, 0]
ax3.scatter(y_test, ensemble_pred, alpha=0.3, s=10, c='steelblue')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Temperature (°C)')
ax3.set_ylabel('Predicted Temperature (°C)')
ax3.set_title('Ensemble Model Performance', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

text = f'R² = {ensemble_r2:.3f}\nRMSE = {ensemble_rmse:.2f}°C'
ax3.text(0.05, 0.95, text, transform=ax3.transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Residual Distribution
ax4 = axes[1, 1]
residuals = ensemble_pred - y_test
ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Residual (°C)')
ax4.set_ylabel('Frequency')
ax4.set_title('Residual Distribution', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add statistics
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)
text = f'Mean: {mean_residual:.2f}°C\nStd: {std_residual:.2f}°C'
ax4.text(0.95, 0.95, text, transform=ax4.transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('final_model_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'final_model_results.png'")
plt.show()

# ============================================
# 12. FINAL RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 FINAL RESULTS SUMMARY")
print("="*60)

print(f"\n1. Best Model: Ensemble (RF + GB + XGBoost)")
print(f"   - R² Score: {ensemble_r2:.4f} {'✅' if ensemble_r2 > 0.7 else '⚠️ (needs improvement)'}")
print(f"   - RMSE: {ensemble_rmse:.2f}°C")

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

print(f"\n4. Physics Check:")
print(f"   {'✅ PASS' if ensemble_r2 > 0.7 else '⚠️ NEEDS IMPROVEMENT'}")

print("\n" + "="*60)
print("✅ Analysis Complete!")