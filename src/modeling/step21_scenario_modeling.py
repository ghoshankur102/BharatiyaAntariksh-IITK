"""
STEP 21: Scenario Modeling - Cooling Interventions
Using Best Model (R² = 0.643)
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🌿 SCENARIO MODELING - Cooling Interventions")
print("="*60)

# ============================================
# 1. LOAD MODEL AND DATA
# ============================================

print("\n📊 Loading model and data...")

# Load the trained model
with open('urban_heat_model_final.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the data
df = pd.read_csv('delhi_urban_heat_with_osm_buildings.csv')
print(f"✅ Loaded {len(df)} samples")

# Features used in the model (from Step 20)
features = ['NDBI', 'BUILDING_DENSITY_OSM', 'NIGHTLIGHTS', 'ALBEDO', 
            'NDWI', 'URBAN_COMPACTNESS', 'VEG_QUALITY', 'POPULATION', 'UHI']

print(f"📋 Model features: {features}")

# ============================================
# 2. DEFINE SCENARIOS
# ============================================

print("\n🔧 Defining cooling scenarios...")

# Create copies of the data
baseline = df.copy()
scenario1 = df.copy()  # +30% Tree Cover
scenario2 = df.copy()  # Cool Roofs (+20% Albedo)
scenario3 = df.copy()  # Combined

# Scenario 1: +30% Tree Cover (Increase vegetation)
print("\n🌳 Scenario 1: +30% Tree Cover")
scenario1['VEG_QUALITY'] = scenario1['VEG_QUALITY'] * 1.3
scenario1['NDVI'] = scenario1['NDVI'] * 1.3
# Physical constraints
scenario1['VEG_QUALITY'] = scenario1['VEG_QUALITY'].clip(0, 1)
scenario1['NDVI'] = scenario1['NDVI'].clip(0, 1)

# Scenario 2: Cool Roofs (Increase Albedo)
print("\n🏠 Scenario 2: Cool Roofs (+20% Albedo)")
scenario2['ALBEDO'] = scenario2['ALBEDO'] * 1.2
scenario2['ALBEDO'] = scenario2['ALBEDO'].clip(0, 0.8)

# Scenario 3: Combined (Both interventions)
print("\n🌳🏠 Scenario 3: Combined Intervention")
scenario3['VEG_QUALITY'] = scenario3['VEG_QUALITY'] * 1.3
scenario3['NDVI'] = scenario3['NDVI'] * 1.3
scenario3['ALBEDO'] = scenario3['ALBEDO'] * 1.2
scenario3['VEG_QUALITY'] = scenario3['VEG_QUALITY'].clip(0, 1)
scenario3['NDVI'] = scenario3['NDVI'].clip(0, 1)
scenario3['ALBEDO'] = scenario3['ALBEDO'].clip(0, 0.8)

print("✅ Scenarios defined!")

# ============================================
# 3. PREPARE FEATURES FOR PREDICTION
# ============================================

print("\n📊 Preparing features for prediction...")

# Extract features for each scenario
X_baseline = baseline[features]
X_scenario1 = scenario1[features]
X_scenario2 = scenario2[features]
X_scenario3 = scenario3[features]

# ============================================
# 4. PREDICT TEMPERATURES
# ============================================

print("\n🌡️ Predicting temperatures for each scenario...")

# Predict baseline
baseline_temp = model.predict(X_baseline)
print(f"   Baseline Mean Temp: {baseline_temp.mean():.2f}°C")
print(f"   Baseline Max Temp: {baseline_temp.max():.2f}°C")
print(f"   Baseline Min Temp: {baseline_temp.min():.2f}°C")

# Predict Scenario 1
scenario1_temp = model.predict(X_scenario1)
delta1 = scenario1_temp - baseline_temp
print(f"\n   Scenario 1 (+30% Tree Cover):")
print(f"   Mean Temp: {scenario1_temp.mean():.2f}°C")
print(f"   Temperature Reduction: {delta1.mean():.2f}°C")
print(f"   Max Cooling: {delta1.min():.2f}°C")

# Predict Scenario 2
scenario2_temp = model.predict(X_scenario2)
delta2 = scenario2_temp - baseline_temp
print(f"\n   Scenario 2 (Cool Roofs):")
print(f"   Mean Temp: {scenario2_temp.mean():.2f}°C")
print(f"   Temperature Reduction: {delta2.mean():.2f}°C")
print(f"   Max Cooling: {delta2.min():.2f}°C")

# Predict Scenario 3
scenario3_temp = model.predict(X_scenario3)
delta3 = scenario3_temp - baseline_temp
print(f"\n   Scenario 3 (Combined):")
print(f"   Mean Temp: {scenario3_temp.mean():.2f}°C")
print(f"   Temperature Reduction: {delta3.mean():.2f}°C")
print(f"   Max Cooling: {delta3.min():.2f}°C")

# ============================================
# 5. CALCULATE IMPACT METRICS
# ============================================

print("\n📊 Impact Metrics:")

# Area with significant cooling (>1°C reduction)
significant_cooling1 = np.sum(delta1 < -1) / len(delta1) * 100
significant_cooling2 = np.sum(delta2 < -1) / len(delta2) * 100
significant_cooling3 = np.sum(delta3 < -1) / len(delta3) * 100

print(f"\n   Area with >1°C cooling:")
print(f"   Scenario 1 (Tree Cover): {significant_cooling1:.1f}%")
print(f"   Scenario 2 (Cool Roofs): {significant_cooling2:.1f}%")
print(f"   Scenario 3 (Combined): {significant_cooling3:.1f}%")

# Areas with cooling (any reduction)
any_cooling1 = np.sum(delta1 < 0) / len(delta1) * 100
any_cooling2 = np.sum(delta2 < 0) / len(delta2) * 100
any_cooling3 = np.sum(delta3 < 0) / len(delta3) * 100

print(f"\n   Area with any cooling:")
print(f"   Scenario 1 (Tree Cover): {any_cooling1:.1f}%")
print(f"   Scenario 2 (Cool Roofs): {any_cooling2:.1f}%")
print(f"   Scenario 3 (Combined): {any_cooling3:.1f}%")

# ============================================
# 6. COST-BENEFIT ANALYSIS
# ============================================

print("\n💰 Cost-Benefit Analysis:")

# Simplified cost estimates
tree_cost_per_km2 = 50000  # USD per km² for tree planting
coolroof_cost_per_km2 = 100000  # USD per km² for cool roofs

# Estimated area with significant cooling
area_km2 = 1484  # Delhi area in km²

cooling_area1 = significant_cooling1 / 100 * area_km2
cooling_area2 = significant_cooling2 / 100 * area_km2
cooling_area3 = significant_cooling3 / 100 * area_km2

cost1 = cooling_area1 * tree_cost_per_km2
cost2 = cooling_area2 * coolroof_cost_per_km2
cost3 = cooling_area3 * (tree_cost_per_km2 + coolroof_cost_per_km2)

print(f"\n   Scenario 1 (Tree Cover):")
print(f"   Area with >1°C cooling: {cooling_area1:.0f} km²")
print(f"   Estimated Cost: ${cost1:,.0f}")

print(f"\n   Scenario 2 (Cool Roofs):")
print(f"   Area with >1°C cooling: {cooling_area2:.0f} km²")
print(f"   Estimated Cost: ${cost2:,.0f}")

print(f"\n   Scenario 3 (Combined):")
print(f"   Area with >1°C cooling: {cooling_area3:.0f} km²")
print(f"   Estimated Cost: ${cost3:,.0f}")

# ============================================
# 7. VISUALIZATIONS
# ============================================

print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Temperature Reduction Box Plot
ax1 = axes[0, 0]
data_to_plot = [delta1, delta2, delta3]
bp = ax1.boxplot(data_to_plot, 
                 patch_artist=True)
ax1.set_xticklabels(['Tree Cover', 'Cool Roofs', 'Combined'])
colors = ['green', 'orange', 'blue']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_ylabel('Temperature Change (°C)')
ax1.set_title('Temperature Reduction by Scenario')
ax1.grid(True, alpha=0.3)

# Add mean reduction labels
means = [delta1.mean(), delta2.mean(), delta3.mean()]
for i, mean in enumerate(means):
    ax1.text(i+1, mean - 0.1, f'{mean:.2f}°C', ha='center', va='top', fontweight='bold')

# 2. Cumulative Distribution of Cooling
ax2 = axes[0, 1]
for delta, label, color in zip([delta1, delta2, delta3], 
                               ['Tree Cover', 'Cool Roofs', 'Combined'],
                               ['green', 'orange', 'blue']):
    sorted_delta = np.sort(delta)
    cumulative = np.arange(1, len(sorted_delta) + 1) / len(sorted_delta)
    ax2.plot(sorted_delta, cumulative, label=label, color=color, linewidth=2)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax2.axvline(x=-1, color='gray', linestyle=':', linewidth=1, label='-1°C threshold')
ax2.set_xlabel('Temperature Change (°C)')
ax2.set_ylabel('Cumulative Fraction')
ax2.set_title('Cumulative Cooling Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Bar Chart: Mean and Max Cooling
ax3 = axes[1, 0]
scenarios = ['Tree Cover', 'Cool Roofs', 'Combined']
mean_deltas = [delta1.mean(), delta2.mean(), delta3.mean()]
max_deltas = [delta1.min(), delta2.min(), delta3.min()]
colors = ['green', 'orange', 'blue']

x = np.arange(len(scenarios))
width = 0.35

bars1 = ax3.bar(x - width/2, mean_deltas, width, label='Mean Cooling', color=colors)
bars2 = ax3.bar(x + width/2, max_deltas, width, label='Max Cooling', 
                color=['lightgreen', 'peachpuff', 'lightskyblue'])
ax3.set_ylabel('Temperature Reduction (°C)')
ax3.set_title('Mean and Maximum Cooling by Scenario')
ax3.set_xticks(x)
ax3.set_xticklabels(scenarios)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add value labels
for bar, val in zip(bars1, mean_deltas):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05, 
             f'{val:.2f}°C', ha='center', va='top', fontweight='bold', color='white')

# 4. Summary Table
ax4 = axes[1, 1]
ax4.axis('off')

# Create summary table
summary_data = [
    ['Scenario', 'Mean Cooling', 'Max Cooling', '>1°C Cooling', 'Est. Cost'],
    ['Tree Cover', f'{delta1.mean():.2f}°C', f'{delta1.min():.2f}°C', f'{significant_cooling1:.0f}%', f'${cost1:,.0f}'],
    ['Cool Roofs', f'{delta2.mean():.2f}°C', f'{delta2.min():.2f}°C', f'{significant_cooling2:.0f}%', f'${cost2:,.0f}'],
    ['Combined', f'{delta3.mean():.2f}°C', f'{delta3.min():.2f}°C', f'{significant_cooling3:.0f}%', f'${cost3:,.0f}']
]

# Create table
table = ax4.table(cellText=summary_data, loc='center', cellLoc='center', 
                  colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

# Color header
for i in range(5):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

ax4.set_title('Scenario Summary', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('scenario_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'scenario_results.png'")
plt.show()

# ============================================
# 8. RECOMMENDATIONS
# ============================================

print("\n" + "="*60)
print("📋 RECOMMENDATIONS")
print("="*60)

# Find best scenario
best_scenario_idx = np.argmin([delta1.mean(), delta2.mean(), delta3.mean()])
scenario_names = ['Tree Cover (30%)', 'Cool Roofs (20%)', 'Combined']
best_scenario = scenario_names[best_scenario_idx]
best_cooling = [delta1.mean(), delta2.mean(), delta3.mean()][best_scenario_idx]

print(f"\n🏆 Best Intervention: {best_scenario}")
print(f"   Expected Cooling: {best_cooling:.2f}°C")

print(f"\n📊 Recommended Strategy:")
if best_scenario_idx == 2:  # Combined
    print("   1. Implement tree cover increase in priority zones")
    print("   2. Apply cool roofs to high-density urban areas")
    print("   3. Focus on areas with >1°C cooling potential")
elif best_scenario_idx == 0:  # Tree Cover
    print("   1. Prioritize tree planting in urban hotspots")
    print("   2. Target areas with low NDVI and high NDBI")
else:  # Cool Roofs
    print("   1. Apply cool roofs to dense urban areas")
    print("   2. Focus on areas with high building density")

print("\n" + "="*60)
print("✅ Scenario modeling complete!")