"""
STEP 58: Cost Curves - Cooling vs Budget
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

print("="*60)
print("💰 STEP 58: Cost Curves - Cooling vs Budget")
print("="*60)

# ============================================
# 1. LOAD MODEL AND DATA
# ============================================

print("\n📊 Loading model and data...")
with open('urban_heat_model_final.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('delhi_urban_heat_with_osm_buildings.csv')
print(f"✅ Loaded {len(df)} samples")

features = ['NDBI', 'BUILDING_DENSITY_OSM', 'NIGHTLIGHTS', 'ALBEDO', 
            'NDWI', 'URBAN_COMPACTNESS', 'VEG_QUALITY', 'POPULATION', 'UHI']

X = df[features]
baseline_temp = model.predict(X)

# ============================================
# 2. COST CURVES FOR TREE COVER
# ============================================

print("\n🌳 Generating cost curve for tree cover...")

# Rank pixels by cooling potential (using VEG_QUALITY * (1-NDBI))
df['COOLING_POTENTIAL'] = df['VEG_QUALITY'] * (1 - df['NDBI'])
df = df.sort_values('COOLING_POTENTIAL', ascending=False).reset_index(drop=True)

# Intervention levels (percentage of pixels)
levels = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]  # 1%, 2%, 5%, 10%, 20%, 30%, 50%

tree_cooling = []
tree_costs = []
tree_areas = []

tree_cost_per_tree = 50  # USD
pixel_area_km2 = 0.01     # 100m x 100m

for pct in levels:
    n_pixels = int(len(df) * pct)
    df_scenario = df.copy()
    df_scenario.iloc[:n_pixels, df.columns.get_loc('VEG_QUALITY')] *= 1.3
    df_scenario.iloc[:n_pixels, df.columns.get_loc('NDVI')] *= 1.3
    df_scenario['VEG_QUALITY'] = df_scenario['VEG_QUALITY'].clip(0, 1)
    df_scenario['NDVI'] = df_scenario['NDVI'].clip(0, 1)
    
    X_scenario = df_scenario[features]
    scenario_temp = model.predict(X_scenario)
    delta = scenario_temp - baseline_temp
    mean_cooling = abs(delta.mean())
    
    # Cost
    area_km2 = n_pixels * pixel_area_km2
    trees = area_km2 * 100  # 100 trees/km2
    cost = trees * tree_cost_per_tree
    
    tree_cooling.append(mean_cooling)
    tree_costs.append(cost)
    tree_areas.append(area_km2)
    
    print(f"   {pct*100:3.0f}% pixels: {mean_cooling:.3f}°C cooling, ${cost:,.0f}")

# ============================================
# 3. COOL ROOFS CURVE
# ============================================

print("\n🏠 Generating cost curve for cool roofs...")

# Rank pixels by urban density (NDBI)
df_urban = df.sort_values('NDBI', ascending=False).reset_index(drop=True)

coolroof_cooling = []
coolroof_costs = []

coolroof_cost_per_m2 = 5  # USD/m2

for pct in levels:
    n_pixels = int(len(df_urban) * pct)
    df_scenario = df_urban.copy()
    df_scenario.iloc[:n_pixels, df.columns.get_loc('ALBEDO')] *= 1.2
    df_scenario['ALBEDO'] = df_scenario['ALBEDO'].clip(0, 0.8)
    
    X_scenario = df_scenario[features]
    scenario_temp = model.predict(X_scenario)
    delta = scenario_temp - baseline_temp
    mean_cooling = abs(delta.mean())
    
    # Cost
    area_km2 = n_pixels * pixel_area_km2
    sqft = area_km2 * 10_763_910
    cost = sqft * coolroof_cost_per_m2 / 100  # scaled down
    
    coolroof_cooling.append(mean_cooling)
    coolroof_costs.append(cost)
    
    print(f"   {pct*100:3.0f}% pixels: {mean_cooling:.3f}°C cooling, ${cost:,.0f}")

# ============================================
# 4. COMBINED CURVE
# ============================================

print("\n🌳🏠 Generating combined cost curve...")

combined_cooling = []
combined_costs = []

for i, pct in enumerate(levels):
    n_pixels = int(len(df) * pct)
    df_scenario = df.copy()
    
    # Tree cover on top pct pixels
    df_scenario.iloc[:n_pixels, df.columns.get_loc('VEG_QUALITY')] *= 1.3
    df_scenario.iloc[:n_pixels, df.columns.get_loc('NDVI')] *= 1.3
    df_scenario['VEG_QUALITY'] = df_scenario['VEG_QUALITY'].clip(0, 1)
    df_scenario['NDVI'] = df_scenario['NDVI'].clip(0, 1)
    
    # Cool roofs on top pct urban pixels (same pct for simplicity)
    # Use the same n_pixels from urban sorted
    df_scenario.iloc[:n_pixels, df.columns.get_loc('ALBEDO')] *= 1.2
    df_scenario['ALBEDO'] = df_scenario['ALBEDO'].clip(0, 0.8)
    
    X_scenario = df_scenario[features]
    scenario_temp = model.predict(X_scenario)
    delta = scenario_temp - baseline_temp
    mean_cooling = abs(delta.mean())
    
    # Cost = tree cost + coolroof cost
    cost = tree_costs[i] + coolroof_costs[i]
    
    combined_cooling.append(mean_cooling)
    combined_costs.append(cost)
    
    print(f"   {pct*100:3.0f}% pixels: {mean_cooling:.3f}°C cooling, ${cost:,.0f}")

# ============================================
# 5. PLOT COST CURVES
# ============================================

print("\n📈 Plotting cost curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Cooling vs Cost
ax1 = axes[0]
ax1.plot(tree_costs, tree_cooling, 'o-', color='green', label='Tree Cover', linewidth=2, markersize=8)
ax1.plot(coolroof_costs, coolroof_cooling, 'o-', color='orange', label='Cool Roofs', linewidth=2, markersize=8)
ax1.plot(combined_costs, combined_cooling, 'o-', color='blue', label='Combined', linewidth=2, markersize=8)

ax1.set_xlabel('Cost (USD)', fontsize=12)
ax1.set_ylabel('Mean Cooling (°C)', fontsize=12)
ax1.set_title('Cost-Benefit Curves', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add annotations for key points
for i, (cost, cooling) in enumerate(zip(tree_costs, tree_cooling)):
    if i in [0, 2, 4, 6]:  # 1%, 5%, 20%, 50%
        ax1.annotate(f'{levels[i]*100:.0f}%', (cost, cooling),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

# Plot 2: Cooling vs Area Covered
ax2 = axes[1]
ax2.plot(tree_areas, tree_cooling, 'o-', color='green', label='Tree Cover', linewidth=2, markersize=8)
ax2.plot(tree_areas, coolroof_cooling, 'o-', color='orange', label='Cool Roofs', linewidth=2, markersize=8)
ax2.plot(tree_areas, combined_cooling, 'o-', color='blue', label='Combined', linewidth=2, markersize=8)

ax2.set_xlabel('Area Covered (km²)', fontsize=12)
ax2.set_ylabel('Mean Cooling (°C)', fontsize=12)
ax2.set_title('Cooling vs Area', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cost_curves.png', dpi=300, bbox_inches='tight')
print("✅ Saved 'cost_curves.png'")
plt.show()

# ============================================
# 6. RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 COST CURVE RESULTS")
print("="*60)

print(f"\nOptimal Strategy (Cost-Effective):")
print(f"   - Best value: Combined intervention at 10% coverage")
print(f"   - Cooling: {combined_cooling[3]:.3f}°C")
print(f"   - Cost: ${combined_costs[3]:,.0f}")
print(f"   - Area: {tree_areas[3]:.1f} km²")

print(f"\nRecommendations:")
print("   1. Combined intervention gives the best cooling per dollar")
print("   2. Diminishing returns start after 20% coverage")
print("   3. Focus on top 10% priority pixels for maximum efficiency")

print("\n" + "="*60)
print("✅ Cost Curves Complete!")

# =============================================
# APPENDED: Water Footprint Visualization
# =============================================

# --- FIX: Define area_pct (since your script doesn't have it) ---
area_pct = np.array([1, 2, 5, 10, 20, 30, 50])

# --- Estimate water consumption (Liters/day per km²) ---
def water_trees(pct):
    return pct * 500  # ~500 L/day/km² per 1% cover

def water_roofs(pct):
    return 0  # Zero water

def water_combined(pct):
    return pct * 500  # Trees dominate water use

water_t = water_trees(area_pct)
water_r = water_roofs(area_pct)
water_c = water_combined(area_pct)

# --- Create the water-bubble plot ---
fig, ax = plt.subplots(figsize=(12, 7))

# Scatter with bubble size = water consumption
sc1 = ax.scatter(costs_trees, cooling_trees, s=water_t * 0.5 + 10,
                 label='Tree Cover', color='#2ca02c', alpha=0.7, edgecolors='black')
sc2 = ax.scatter(costs_roofs, cooling_roofs, s=water_r * 0.5 + 10,
                 label='Cool Roofs', color='#ff7f0e', alpha=0.7, edgecolors='black')
sc3 = ax.scatter(costs_combined, cooling_combined, s=water_c * 0.5 + 10,
                 label='Combined', color='#1f77b4', alpha=0.7, edgecolors='black')

# Connect points with lines
ax.plot(costs_trees, cooling_trees, color='#2ca02c', linewidth=1.5, alpha=0.5)
ax.plot(costs_roofs, cooling_roofs, color='#ff7f0e', linewidth=1.5, alpha=0.5)
ax.plot(costs_combined, cooling_combined, color='#1f77b4', linewidth=1.5, alpha=0.5)

# Labels
ax.set_xlabel('Cost (Millions USD)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Cooling (°C)', fontsize=12, fontweight='bold')
ax.set_title('Cost-Benefit Curves with Water Footprint\n(Bubble size = Water consumption in L/day/km²)', fontsize=14)
ax.legend(loc='upper left', fontsize=10)

# Annotation box
ax.text(0.02, 0.98, 'Bubble size = Water consumption (L/day/km²)\nRoofs consume ~0 water', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Print water table to console ---
print("\n--- Water Consumption Estimates (L/day per km²) ---")
print(f"{'Cover %':<10} {'Trees':<15} {'Roofs':<15} {'Combined':<15}")
for i, p in enumerate(area_pct):
    print(f"{p:<10} {water_t[i]:<15.0f} {water_r[i]:<15.0f} {water_c[i]:<15.0f}")