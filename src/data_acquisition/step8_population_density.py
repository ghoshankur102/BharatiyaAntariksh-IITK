import ee
import pandas as pd
import numpy as np
import geemap
import matplotlib.pyplot as plt

print("="*60)
print("👥 ADDING POPULATION DENSITY DATA")
print("="*60)

# Initialize Earth Engine
ee.Initialize(project='urbanheatmitigation-500914')

# Delhi coordinates
delhi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.9])

print("\n📥 Loading population density from WorldPop...")

# ============================================
# 1. LOAD POPULATION DATA
# ============================================

# WorldPop population density (2020)
population = (ee.ImageCollection('WorldPop/GP/100m/pop')
              .filterBounds(delhi)
              .filterDate('2020-01-01', '2020-12-31')
              .select('population')
              .median()
              .clip(delhi))

print("✅ Population data loaded!")

# ============================================
# 2. EXTRACT POPULATION DATA
# ============================================

print("\n📊 Extracting population data...")

# Sample population data
scale = 100
num_pixels = 5000

sample = population.sample(
    region=delhi,
    scale=scale,
    numPixels=num_pixels,
    seed=42,
    geometries=False
)

sample_data = sample.getInfo()

if 'features' in sample_data and len(sample_data['features']) > 0:
    data_list = []
    for feature in sample_data['features']:
        data_list.append(feature['properties'])
    
    pop_df = pd.DataFrame(data_list)
    pop_df.columns = ['population_density']
    
    # Clean data
    pop_df = pop_df.dropna()
    pop_df = pop_df[pop_df['population_density'] > 0]
    
    # Save
    pop_df.to_csv('population_density.csv', index=False)
    print(f"✅ Saved {len(pop_df)} population samples to 'population_density.csv'")
    
    print(f"\n📊 Population Density Statistics:")
    print(f"   Mean: {pop_df['population_density'].mean():.0f} people/km²")
    print(f"   Max: {pop_df['population_density'].max():.0f} people/km²")
    print(f"   Min: {pop_df['population_density'].min():.0f} people/km²")
    
    # ============================================
    # 3. VISUALIZATION
    # ============================================
    
    print("\n📈 Visualizing population density...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Population density histogram
    ax1 = axes[0]
    pop_values = pop_df['population_density']
    ax1.hist(pop_values[pop_values < pop_values.quantile(0.95)], 
             bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Population Density (people/km²)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Population Density Distribution')
    ax1.axvline(pop_values.mean(), color='red', linestyle='--', label=f'Mean: {pop_values.mean():.0f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Population density stats
    ax2 = axes[1]
    ax2.text(0.5, 0.5, 
             f'Population Density Summary:\n\n'
             f'Mean: {pop_values.mean():.0f} people/km²\n'
             f'Median: {pop_values.median():.0f} people/km²\n'
             f'Max: {pop_values.max():.0f} people/km²\n'
             f'Min: {pop_values.min():.0f} people/km²\n'
             f'Samples: {len(pop_values)}',
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.set_title('Population Density Summary')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('population_density.png', dpi=300, bbox_inches='tight')
    print("✅ Saved 'population_density.png'")
    plt.show()
    
else:
    print("❌ No population data extracted")

print("\n✅ Population data processing complete!")