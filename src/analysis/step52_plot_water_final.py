import matplotlib.pyplot as plt
import numpy as np

# =============================================
# HARDCODED DATA FROM YOUR TERMINAL OUTPUT
# =============================================

# Costs (in Millions USD)
costs_trees = np.array([0.123, 0.246, 0.615, 1.230, 2.460, 3.691, 6.152])
costs_roofs = np.array([13.239, 26.484, 66.219, 132.439, 264.883, 397.328, 662.217])
costs_combined = np.array([13.362, 26.730, 66.834, 133.669, 267.344, 401.019, 668.369])

# Cooling (°C)
cooling_trees = np.array([0.001, 0.003, 0.012, 0.019, 0.018, 0.002, 0.013])
cooling_roofs = np.array([0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033])
cooling_combined = np.array([0.032, 0.031, 0.022, 0.017, 0.023, 0.037, 0.066])

# Area percentages (from your print output)
area_pct = np.array([1, 2, 5, 10, 20, 30, 50])

# =============================================
# WATER FOOTPRINT ESTIMATES
# =============================================

def water_trees(pct):
    return pct * 500  # L/day/km²

def water_roofs(pct):
    return 0  # Zero water

def water_combined(pct):
    return pct * 500  # Trees dominate

water_t = water_trees(area_pct)
water_r = water_roofs(area_pct)
water_c = water_combined(area_pct)

# =============================================
# PLOT: COST vs COOLING (BUBBLE SIZE = WATER)
# =============================================

fig, ax = plt.subplots(figsize=(12, 7))

# Scatter plots with bubbles
ax.scatter(costs_trees, cooling_trees, s=water_t * 0.5 + 10,
           label='Tree Cover', color='#2ca02c', alpha=0.7, edgecolors='black')
ax.scatter(costs_roofs, cooling_roofs, s=water_r * 0.5 + 10,
           label='Cool Roofs', color='#ff7f0e', alpha=0.7, edgecolors='black')
ax.scatter(costs_combined, cooling_combined, s=water_c * 0.5 + 10,
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

# Print water table
print("\n--- Water Consumption Estimates (L/day per km²) ---")
print(f"{'Cover %':<10} {'Trees':<15} {'Roofs':<15} {'Combined':<15}")
for i, p in enumerate(area_pct):
    print(f"{p:<10} {water_t[i]:<15.0f} {water_r[i]:<15.0f} {water_c[i]:<15.0f}")