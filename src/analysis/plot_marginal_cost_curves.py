import matplotlib.pyplot as plt
import numpy as np

# =============================================
# YOUR DATA (copied from terminal output)
# =============================================
area_pct = np.array([1, 2, 5, 10, 20, 30, 50])

# Costs in Millions USD
cost_trees = np.array([0.123, 0.246, 0.615, 1.230, 2.460, 3.691, 6.152])
cost_roofs = np.array([13.239, 26.484, 66.219, 132.439, 264.883, 397.328, 662.217])
cost_combined = np.array([13.362, 26.730, 66.834, 133.669, 267.344, 401.019, 668.369])

# Cooling (°C)
cooling_trees = np.array([0.001, 0.003, 0.012, 0.019, 0.018, 0.002, 0.013])
cooling_roofs = np.array([0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033])
cooling_combined = np.array([0.032, 0.031, 0.022, 0.017, 0.023, 0.037, 0.066])

# =============================================
# CALCULATE MARGINAL COSTS
# =============================================

def compute_marginal_cost(cost_arr, cooling_arr, area_pct):
    """
    Returns:
    - marginal_cost: Million USD per °C for each step (i -> i+1)
    - area_mid: midpoint area for plotting (x-axis)
    """
    marginal = []
    area_mid = []
    
    for i in range(len(cost_arr) - 1):
        delta_cost = cost_arr[i+1] - cost_arr[i]          # $M
        delta_cool = cooling_arr[i+1] - cooling_arr[i]    # °C
        
        # Avoid division by zero
        if delta_cool == 0:
            marginal.append(np.inf)   # Infinite cost per °C
        else:
            marginal.append(delta_cost / delta_cool)
        
        # Midpoint of the interval for x-axis
        area_mid.append((area_pct[i] + area_pct[i+1]) / 2)
    
    return np.array(marginal), np.array(area_mid)

# Compute for all three strategies
marg_trees, mid_trees = compute_marginal_cost(cost_trees, cooling_trees, area_pct)
marg_roofs, mid_roofs = compute_marginal_cost(cost_roofs, cooling_roofs, area_pct)
marg_combined, mid_combined = compute_marginal_cost(cost_combined, cooling_combined, area_pct)

# =============================================
# PLOT 1: Marginal Cost (Log Scale)
# =============================================
fig, ax = plt.subplots(figsize=(12, 7))

# Plot only finite values (skip inf for roofs)
# For roofs, the constant cooling gives infinite marginal cost – we plot a horizontal line at the top
ax.plot(mid_trees, marg_trees, marker='o', label='Tree Cover', color='#2ca02c', linewidth=2)
ax.plot(mid_combined, marg_combined, marker='^', label='Combined', color='#1f77b4', linewidth=2)

# Roofs: handle infinite values
finite_mask = np.isfinite(marg_roofs)
if np.any(finite_mask):
    ax.plot(mid_roofs[finite_mask], marg_roofs[finite_mask], marker='s', label='Cool Roofs', color='#ff7f0e', linewidth=2, linestyle='--')
else:
    # If all are inf, plot a horizontal line at a very high value (e.g., 10^6)
    ax.axhline(y=1e5, color='#ff7f0e', linestyle='--', label='Cool Roofs (Infinite Marginal Cost)')
    ax.text(15, 1e5, 'Roofs: ΔCooling = 0 → ∞ Cost', color='#ff7f0e', fontsize=9, ha='center')

# Set log scale for y-axis because costs span orders of magnitude
ax.set_yscale('log')
ax.set_ylim(1, 1e8)  # Adjust as needed for your data

# Labels
ax.set_xlabel('Area Covered (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Marginal Cost (Million USD per °C)', fontsize=12, fontweight='bold')
ax.set_title('Dynamic Marginal Cost Curves\n(Lower is better – diminishing returns kick in when curve rises)', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6, which='both')

# Annotate the "sweet spot" where Combined becomes cheaper than Trees/Roofs
# Find the index where Combined marginal cost drops below Trees
min_combined_idx = np.argmin(marg_combined)
ax.annotate(f'Min Marginal Cost\n({mid_combined[min_combined_idx]:.0f}% cover)',
            xy=(mid_combined[min_combined_idx], marg_combined[min_combined_idx]),
            xytext=(mid_combined[min_combined_idx]+5, marg_combined[min_combined_idx]*2),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# =============================================
# PLOT 2: Marginal Cooling per Million USD
# (Alternative view – shows efficiency drop)
# =============================================
fig2, ax2 = plt.subplots(figsize=(12, 7))

# Compute cooling per $M for each interval
def compute_marginal_efficiency(cost_arr, cooling_arr):
    efficiency = []
    for i in range(len(cost_arr) - 1):
        delta_cost = cost_arr[i+1] - cost_arr[i]
        delta_cool = cooling_arr[i+1] - cooling_arr[i]
        if delta_cost == 0:
            efficiency.append(0)
        else:
            efficiency.append(delta_cool / delta_cost)   # °C per Million USD
    return np.array(efficiency)

eff_trees = compute_marginal_efficiency(cost_trees, cooling_trees)
eff_roofs = compute_marginal_efficiency(cost_roofs, cooling_roofs)
eff_combined = compute_marginal_efficiency(cost_combined, cooling_combined)

ax2.plot(mid_trees, eff_trees, marker='o', label='Tree Cover', color='#2ca02c', linewidth=2)
ax2.plot(mid_roofs, eff_roofs, marker='s', label='Cool Roofs', color='#ff7f0e', linewidth=2)
ax2.plot(mid_combined, eff_combined, marker='^', label='Combined', color='#1f77b4', linewidth=2)

ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Area Covered (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Marginal Cooling per Million USD (°C / $M)', fontsize=12, fontweight='bold')
ax2.set_title('Marginal Cooling Efficiency\n(Declining line = diminishing returns)', fontsize=14)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)

# Highlight that Roofs drop to zero (flat cooling = zero marginal efficiency)
ax2.annotate('Roofs: ΔCooling = 0 → Zero Efficiency',
             xy=(10, 0), xytext=(15, 0.0002),
             arrowprops=dict(arrowstyle='->', color='#ff7f0e'),
             fontsize=9, color='#ff7f0e')

plt.tight_layout()
plt.show()

# =============================================
# PRINT SUMMARY TABLE
# =============================================
print("\n" + "="*70)
print("MARGINAL COST SUMMARY (Million USD per °C)")
print("="*70)
print(f"{'Interval':<15} {'Trees':<20} {'Roofs':<20} {'Combined':<20}")
for i, mid in enumerate(mid_trees):
    tree_str = f"{marg_trees[i]:.2f}" if np.isfinite(marg_trees[i]) else "∞"
    roof_str = f"{marg_roofs[i]:.2f}" if np.isfinite(marg_roofs[i]) else "∞"
    comb_str = f"{marg_combined[i]:.2f}" if np.isfinite(marg_combined[i]) else "∞"
    
    # Format interval
    interval = f"{area_pct[i]}%–{area_pct[i+1]}%"
    print(f"{interval:<15} {tree_str:<20} {roof_str:<20} {comb_str:<20}")

print("\n" + "="*70)
print("MARGINAL EFFICIENCY SUMMARY (°C per Million USD)")
print("="*70)
print(f"{'Interval':<15} {'Trees':<20} {'Roofs':<20} {'Combined':<20}")
for i, mid in enumerate(mid_trees):
    print(f"{area_pct[i]}%–{area_pct[i+1]}% {eff_trees[i]:<20.6f} {eff_roofs[i]:<20.6f} {eff_combined[i]:<20.6f}")