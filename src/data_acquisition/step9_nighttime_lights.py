import ee
import pandas as pd
import numpy as np
import geemap
import matplotlib.pyplot as plt

print("="*60)
print("🌃 ADDING NIGHTTIME LIGHTS DATA")
print("="*60)

# Initialize Earth Engine
ee.Initialize(project='urbanheatmitigation-500914')

# Delhi coordinates
delhi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.9])

print("\n📥 Loading nighttime lights from VIIRS...")

# ============================================
# 1. LOAD NIGHTTIME LIGHTS DATA
# ============================================

# VIIRS Nighttime Lights (2023)
nightlights = (ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
               .filterBounds(delhi)
               .filterDate('2023-01-01', '2023-12-31')
               .select('avg_rad')
               .median()
               .clip(delhi))

print("✅ Nighttime lights data loaded!")

# ============================================
# 2. EXTRACT LIGHTS DATA
# ============================================

print("\n📊 Extracting nighttime lights data...")

# Sample lights data
scale = 100
num_pixels = 5000

sample = nightlights.sample(
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
    
    lights_df = pd.DataFrame(data_list)
    lights_df.columns = ['nightlights']
    
    # Clean data
    lights_df = lights_df.dropna()
    lights_df = lights_df[lights_df['nightlights'] > 0]
    
    # Save
    lights_df.to_csv('nighttime_lights.csv', index=False)
    print(f"✅ Saved {len(lights_df)} samples to 'nighttime_lights.csv'")
    
    print(f"\n📊 Nighttime Lights Statistics:")
    print(f"   Mean: {lights_df['nightlights'].mean():.3f}")
    print(f"   Max: {lights_df['nightlights'].max():.3f}")
    print(f"   Min: {lights_df['nightlights'].min():.3f}")
    
    # ============================================
    # 3. VISUALIZATION
    # ============================================
    
    print("\n📈 Visualizing nighttime lights...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Lights histogram
    ax1 = axes[0]
    lights_values = lights_df['nightlights']
    ax1.hist(lights_values[lights_values < lights_values.quantile(0.95)], 
             bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax1.set_xlabel('Nighttime Radiance (nW/cm²/sr)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Nighttime Lights Distribution')
    ax1.axvline(lights_values.mean(), color='red', linestyle='--', label=f'Mean: {lights_values.mean():.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Nighttime lights summary
    ax2 = axes[1]
    ax2.text(0.5, 0.5, 
             f'Nighttime Lights Summary:\n\n'
             f'Mean: {lights_values.mean():.3f}\n'
             f'Median: {lights_values.median():.3f}\n'
             f'Max: {lights_values.max():.3f}\n'
             f'Min: {lights_values.min():.3f}\n'
             f'Samples: {len(lights_values)}',
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.set_title('Nighttime Lights Summary')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('nighttime_lights.png', dpi=300, bbox_inches='tight')
    print("✅ Saved 'nighttime_lights.png'")
    plt.show()
    
else:
    print("❌ No nighttime lights data extracted")

print("\n✅ Nighttime lights processing complete!")