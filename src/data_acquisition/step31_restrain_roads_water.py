import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🏆 RETRAIN MODEL with Roads + Water")
print("="*60)

# Load data
df = pd.read_csv('delhi_urban_heat_with_roads_water.csv')
print(f"✅ Loaded {len(df)} samples")

# Features
features = ['NDBI', 'BUILDING_DENSITY_OSM', 'NIGHTLIGHTS', 'ALBEDO', 
            'NDWI', 'URBAN_COMPACTNESS', 'VEG_QUALITY', 'POPULATION', 'UHI',
            'ROAD_DENSITY', 'DISTANCE_TO_WATER']

X = df[features]
y = df['LST']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
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

# Evaluate
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 New Model Performance:")
print(f"   R²: {r2:.4f}")
print(f"   RMSE: {rmse:.2f}°C")

# Feature importance
importance = pd.DataFrame({
    'Driver': X.columns,
    'Importance': model.feature_importances_,
    'Percentage': model.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

print("\n🎯 Feature Importance:")
for idx, row in importance.iterrows():
    print(f"   {row['Driver']:25s}: {row['Percentage']:.1f}%")

# Save model
with open('model_with_roads_water.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved")

print(f"\n📈 Improvement: {r2 - 0.643:.4f}")