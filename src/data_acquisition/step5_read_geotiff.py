import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("📂 Reading downloaded GeoTIFF...")

# ============================================
# 1. READ THE GEOTIFF
# ============================================

file_path = r"D:\urban_heat_mitigation\delhi_urban_heat.tif"

try:
    # Open the GeoTIFF
    src = rasterio.open(file_path)
    print(f"✅ File opened successfully")
    print(f"   Shape: {src.shape}")
    print(f"   Bands: {src.count}")
    print(f"   CRS: {src.crs}")
    
    # Read all bands
    data = src.read()
    print(f"   Data shape: {data.shape}")
    
    # Get band names (if available)
    try:
        band_names = src.descriptions
        if not band_names or all(b == '' for b in band_names):
            band_names = ['LST', 'NDVI', 'NDBI', 'NDWI', 'ALBEDO']
    except:
        band_names = ['LST', 'NDVI', 'NDBI', 'NDWI', 'ALBEDO']
    
    print(f"   Band names: {band_names}")
    
    # Close the file
    src.close()
    
    # ============================================
    # 2. CONVERT TO DATAFRAME
    # ============================================
    
    print("\n📊 Converting to DataFrame...")
    
    # Reshape data (flatten each band)
    n_bands, height, width = data.shape
    
    # Create DataFrame
    df = pd.DataFrame()
    
    for i, band_name in enumerate(band_names[:n_bands]):  # Only use available bands
        # Flatten the band
        flattened = data[i].flatten()
        df[band_name] = flattened
    
    # Remove invalid values (NaN, inf, etc.)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Filter for realistic values
    df = df[(df['LST'] > 20) & (df['LST'] < 50)]
    df = df[(df['NDVI'] > -0.5) & (df['NDVI'] < 1.0)]
    df = df[(df['NDBI'] > -0.5) & (df['NDBI'] < 0.5)]
    
    print(f"✅ Data extracted! {len(df)} samples collected")
    
    # ============================================
    # 3. SAVE CSV
    # ============================================
    
    df.to_csv('delhi_urban_heat_final.csv', index=False)
    print("✅ Data saved to 'delhi_urban_heat_final.csv'")
    
    # ============================================
    # 4. STATISTICS
    # ============================================
    
    print("\n📊 Feature Statistics:")
    print(df.describe())
    
    print("\n📈 Correlation with LST:")
    corr = df.corr()['LST'].sort_values(ascending=False)
    print(corr)
    
    # ============================================
    # 5. VISUALIZE DATA
    # ============================================
    
    print("\n📈 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot each band
    for i, band_name in enumerate(band_names[:n_bands]):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Get the band data
        band_data = data[i]
        
        # Plot with percentile stretching
        vmin = np.percentile(band_data[~np.isnan(band_data)], 5)
        vmax = np.percentile(band_data[~np.isnan(band_data)], 95)
        
        im = ax.imshow(band_data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(band_name)
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    # Remove empty subplot if any
    if n_bands < 6:
        for i in range(n_bands, 6):
            row = i // 3
            col = i % 3
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('delhi_bands.png', dpi=300)
    print("✅ Saved 'delhi_bands.png'")
    plt.show()
    
    # ============================================
    # 6. CORRELATION HEATMAP
    # ============================================
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='RdYlBu_r', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)
    print("✅ Saved 'correlation_heatmap.png'")
    plt.show()
    
    print("\n✅ Analysis complete! Ready for ML model.")

except FileNotFoundError:
    print(f"❌ File not found at: {file_path}")
    print("Please check the file path and make sure the file exists")
    
except Exception as e:
    print(f"❌ Error: {e}")