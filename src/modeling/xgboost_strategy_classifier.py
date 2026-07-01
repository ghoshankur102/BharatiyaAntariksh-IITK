#!/usr/bin/env python3
"""
xgboost_strategy_clean.py
XGBoost classifier for cooling strategies – uses HVI as a feature.
No extra arguments to fit() – pure training, then evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import os

# ================================================================
# 1. CONFIGURATION
# ================================================================

PIXEL_CSV = "delhi_heat_with_districts_pop_poverty.csv"
HVI_CSV = "heat_vulnerability_index_final.csv"
MODEL_OUTPUT = "xgboost_strategy_model.pkl"
OUTPUT_METRICS = "strategy_classification_report.txt"
OUTPUT_CM = "confusion_matrix_strategy.png"
OUTPUT_FI = "feature_importance_strategy.png"

# ================================================================
# 2. STRATEGY MAPPING
# ================================================================

strategy_map = {
    'West': 'Cool Roofs',
    'Central': 'Cool Roofs',
    'South West': 'Green Corridors',
    'North West': 'Blue-Green Infrastructure',
    'North East': 'Tree Corridors',
    'New Delhi': 'Cooling Centres',
    'North': 'Cool Roofs (equity)',
    'South': 'Maintain Green',
    'East': 'Perimeter Shading'
}

# ================================================================
# 3. LOAD DATA
# ================================================================

print("📊 Loading pixel data...")
df_pixels = pd.read_csv(PIXEL_CSV)
print(f"   Shape: {df_pixels.shape}")

print("📊 Loading HVI data...")
df_hvi = pd.read_csv(HVI_CSV)
print(f"   Shape: {df_hvi.shape}")

# Merge HVI
df_pixels = df_pixels.merge(df_hvi[['district', 'HVI']], on='district', how='left')
print(f"   Merged shape: {df_pixels.shape}")

# ================================================================
# 4. ASSIGN STRATEGY LABELS
# ================================================================

df_pixels['strategy'] = df_pixels['district'].map(strategy_map)
df_pixels = df_pixels.dropna(subset=['strategy'])
print(f"   After assigning strategies: {df_pixels.shape}")

# ================================================================
# 5. FEATURES & TARGET
# ================================================================

feature_cols = [
    'LST', 'NDVI', 'NDBI', 'NDWI', 'ALBEDO',
    'POPULATION', 'NIGHTLIGHTS', 'BUILDING_DENSITY',
    'UHI', 'VEG_QUALITY', 'URBAN_COMPACTNESS',
    'BUILDING_DENSITY_OSM',
    'HVI'
]
feature_cols = [c for c in feature_cols if c in df_pixels.columns]

X = df_pixels[feature_cols].copy()
y = df_pixels['strategy'].copy()

print(f"\n🔢 Features: {len(X.columns)}")
print(f"   {X.columns.tolist()}")
print(f"\n📊 Strategy distribution:\n{y.value_counts()}")

# ================================================================
# 6. ENCODE TARGET
# ================================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, 'strategy_label_encoder.pkl')

# ================================================================
# 7. TRAIN/TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n📊 Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# ================================================================
# 8. CLASS WEIGHTS (sample_weight)
# ================================================================

class_counts = np.bincount(y_encoded)
total = len(y_encoded)
n_classes = len(class_counts)
class_weights = {i: total / (n_classes * c) for i, c in enumerate(class_counts)}
sample_weights = np.array([class_weights[c] for c in y_train])

# ================================================================
# 9. TRAIN XGBOOST (NO extra arguments)
# ================================================================

print("\n⚡ Training XGBoost classifier...")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Only X, y, and sample_weight
model.fit(X_train, y_train, sample_weight=sample_weights)

print("✅ Model trained.")

# ================================================================
# 10. EVALUATE
# ================================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 Accuracy on test set: {accuracy:.4f}")

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("\n📋 Classification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix – Cooling Strategies')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(OUTPUT_CM, dpi=300)
print(f"✅ Confusion matrix saved to: {OUTPUT_CM}")

with open(OUTPUT_METRICS, 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)
print(f"✅ Report saved to: {OUTPUT_METRICS}")

# ================================================================
# 11. FEATURE IMPORTANCE
# ================================================================

importance = model.feature_importances_
df_imp = pd.DataFrame({'feature': X.columns, 'importance': importance})
df_imp = df_imp.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(df_imp['feature'], df_imp['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance – XGBoost Strategy Classifier')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_FI, dpi=300)
print(f"✅ Feature importance plot saved to: {OUTPUT_FI}")

print("\n🏆 Top 5 Features:")
print(df_imp.head(5).to_string(index=False))

# ================================================================
# 12. SAVE MODEL
# ================================================================

joblib.dump(model, MODEL_OUTPUT)
print(f"✅ Model saved to: {MODEL_OUTPUT}")

print("\n✅ Analysis complete!")