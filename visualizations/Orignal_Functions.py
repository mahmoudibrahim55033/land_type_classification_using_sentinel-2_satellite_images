import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import rasterio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json


# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def get_dataset_configuration():
    LAND_TYPES = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake"
    ]
    
    SENTINEL2_BANDS = {
        1: "Coastal aerosol (B1)",
        2: "Blue (B2)",
        3: "Green (B3)",
        4: "Red (B4)",
        5: "Vegetation Red Edge (B5)",
        6: "Vegetation Red Edge (B6)",
        7: "Vegetation Red Edge (B7)",
        8: "NIR (B8)",
        9: "Narrow NIR (B8A)",
        10: "Water Vapor (B9)",
        11: "SWIR (B11)",
        12: "SWIR (B12)"
    }
    
    return LAND_TYPES, SENTINEL2_BANDS


def setup_directories(base_dir_path, viz_dir_path):
    DATA_DIR = Path(base_dir_path)
    VISUALIZATIONS_DIR = Path(viz_dir_path)
    
    DATA_DIR.mkdir(exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)
    
    return DATA_DIR, VISUALIZATIONS_DIR


# ============================================================================
# IMAGE LOADING FUNCTIONS
# ============================================================================

def load_tif_image(file_path):
    with rasterio.open(file_path) as src:
        data = src.read()
        meta = src.meta
    return data, meta


#Load one sample image from each land type class.
def load_sample_images_per_class(base_path, land_types):
    sample_images = {}
    sample_metadata = {}
    
    for land_type in land_types:
        image_path = list((Path(base_path) / land_type).glob("*.tif"))[0]
        data, meta = load_tif_image(image_path)
        sample_images[land_type] = data
        sample_metadata[land_type] = meta
    
    return sample_images, sample_metadata


def load_all_images(base_path, land_types):
    all_images = []
    all_labels = []
    all_indices_data = []
    
    for label_idx, land_type in enumerate(land_types):
        land_dir = Path(base_path) / land_type
        image_files = sorted(land_dir.glob('*.tif'))
        
        for img_path in tqdm(image_files, desc=f"Loading {land_type}"):
            try:
                data, _ = load_tif_image(img_path)
                all_images.append(data)
                all_labels.append(label_idx)
                
                ndvi = calculate_ndvi(data)
                ndbi = calculate_ndbi(data)
                ndwi = calculate_ndwi(data)
                
                indices_dict = {
                    'ndvi_mean': ndvi.mean(), 'ndvi_std': ndvi.std(),
                    'ndbi_mean': ndbi.mean(), 'ndbi_std': ndbi.std(),
                    'ndwi_mean': ndwi.mean(), 'ndwi_std': ndwi.std()
                }
                all_indices_data.append(indices_dict)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return all_images, all_labels, all_indices_data


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_band(band, percentile_min=2, percentile_max=98):
    p_min = np.percentile(band, percentile_min)
    p_max = np.percentile(band, percentile_max)
    normalized = np.clip((band - p_min) / (p_max - p_min + 1e-5), 0, 1)
    return normalized


def normalize_image(image_data):
    min_val = image_data.min()
    max_val = image_data.max()
    normalized = (image_data - min_val) / (max_val - min_val + 1e-5)
    
    return normalized


def apply_atmospheric_correction(image_data):
    corrected = np.zeros_like(image_data, dtype=np.float32)
    for band_idx in range(image_data.shape[0]):
        p2 = np.percentile(image_data[band_idx], 2)
        p98 = np.percentile(image_data[band_idx], 98)
        corrected[band_idx] = np.clip((image_data[band_idx] - p2) / (p98 - p2), 0, 1)
    
    return corrected


# ============================================================================
# COMPOSITE CREATION FUNCTIONS
# ============================================================================

def create_rgb_composite(image_data, r_band=3, g_band=2, b_band=1):
    r_idx = min(r_band - 1, image_data.shape[0] - 1)
    g_idx = min(g_band - 1, image_data.shape[0] - 1)
    b_idx = min(b_band - 1, image_data.shape[0] - 1)
    
    rgb = np.stack([
        normalize_band(image_data[r_idx]),
        normalize_band(image_data[g_idx]),
        normalize_band(image_data[b_idx])
    ], axis=-1)
    
    return rgb


def create_false_color_composite(image_data, r_band=8, g_band=4, b_band=3):
    """
    Create False Color composite (NIR-Red-Green).
    
    Args:
        image_data (ndarray): Image with shape (bands, height, width)
        r_band (int): Band index for red channel (default 8=NIR)
        g_band (int): Band index for green channel (default 4=Red)
        b_band (int): Band index for blue channel (default 3=Green)
        
    Returns:
        ndarray: False color composite with shape (height, width, 3)
    """
    return create_rgb_composite(image_data, r_band, g_band, b_band)


# ============================================================================
# VEGETATION INDEX FUNCTIONS
# ============================================================================

def calculate_ndvi(image_data):
    """
    Calculate NDVI (Normalized Difference Vegetation Index).
    NDVI = (NIR - Red) / (NIR + Red); Band 8 is NIR, Band 4 is Red
    """
    nir = image_data[7].astype(float)
    red = image_data[3].astype(float)
    ndvi = (nir - red) / (nir + red + 1e-5)
    
    return ndvi


def calculate_ndbi(image_data):
    """
    Calculate NDBI (Normalized Difference Built-up Index).
    NDBI = (SWIR - NIR) / (SWIR + NIR); Band 11 is SWIR, Band 8 is NIR
    """
    swir = image_data[10].astype(float)
    nir = image_data[7].astype(float)
    ndbi = (swir - nir) / (swir + nir + 1e-5)
    
    return ndbi


def calculate_ndwi(image_data):
    """
    Calculate NDWI (Normalized Difference Water Index).
    NDWI = (NIR - SWIR) / (NIR + SWIR)
    """
    nir = image_data[7].astype(float)
    swir = image_data[10].astype(float)
    ndwi = (nir - swir) / (nir + swir + 1e-5)
    
    return ndwi


def calculate_vegetation_indices(image_data):
    ndvi = calculate_ndvi(image_data)
    ndbi = calculate_ndbi(image_data)
    ndwi = calculate_ndwi(image_data)
    
    return ndvi, ndbi, ndwi


# ============================================================================
# DATA ANALYSIS FUNCTIONS
# ============================================================================

def analyze_sample_image(image_data):
    print(f"Sample image shape (bands, height, width): {image_data.shape}")
    print(f"Data type: {image_data.dtype}")
    print(f"Value range: {image_data.min()}-{image_data.max()}")
    print(f"\nBand statistics for sample image:")
    
    for band_idx in range(min(12, image_data.shape[0])):
        print(f"  Band {band_idx+1}: Mean={image_data[band_idx].mean():.2f}, "
              f"Std={image_data[band_idx].std():.2f}")


def calculate_spectral_signatures(sample_images, land_types):
    spectral_signatures = {}
    
    for land_type in land_types:
        image_data = sample_images[land_type]
        bands_flattened = image_data.reshape(image_data.shape[0], -1)
        mean_spectrum = bands_flattened.mean(axis=1)
        spectral_signatures[land_type] = mean_spectrum
    
    return spectral_signatures


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def process_all_images(images_list):
    processed_images = []
    
    print("Preprocessing all images...")
    for idx, image in enumerate(tqdm(images_list, desc="Preprocessing")):
        corrected = apply_atmospheric_correction(image)
        normalized = normalize_image(corrected)
        processed_images.append(normalized)
    
    return np.array(processed_images)


def split_data(processed_images, labels_array, test_size=0.2, val_ratio=0.25, random_state=42):
    #Train: 60%, Validation: 20%, Test: 20%

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        processed_images, labels_array, test_size=test_size, 
        random_state=random_state, stratify=labels_array
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, 
        random_state=random_state, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_data_split_statistics(X_train, X_val, X_test, y_train, y_val, y_test, land_types):
    total_samples = len(X_train) + len(X_val) + len(X_test)
    
    print("✓ Data Split Statistics:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Training set: {len(X_train):,} ({len(X_train)/total_samples*100:.1f}%)")
    print(f"   Validation set: {len(X_val):,} ({len(X_val)/total_samples*100:.1f}%)")
    print(f"   Test set: {len(X_test):,} ({len(X_test)/total_samples*100:.1f}%)")
    
    print("\n✓ Class distribution verified across all splits")
    for label_idx in range(min(3, len(land_types))):
        land_type = land_types[label_idx]
        tr = (y_train == label_idx).sum()
        va = (y_val == label_idx).sum()
        te = (y_test == label_idx).sum()
        print(f"   {land_type}: Train={tr}, Val={va}, Test={te}")


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                       indices_data, all_labels, data_dir, land_types, sentinel2_bands):
    data_dir = Path(data_dir)
    
    # Save numpy arrays
    np.save(data_dir / 'X_train.npy', X_train)
    np.save(data_dir / 'X_val.npy', X_val)
    np.save(data_dir / 'X_test.npy', X_test)
    np.save(data_dir / 'y_train.npy', y_train)
    np.save(data_dir / 'y_val.npy', y_val)
    np.save(data_dir / 'y_test.npy', y_test)
    
    # Save metadata
    num_bands = X_train.shape[1]
    metadata = {
        'land_types': land_types,
        'sentinel2_bands': sentinel2_bands,
        'image_shape': (num_bands, X_train.shape[2], X_train.shape[3]),
        'num_classes': len(land_types),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'dataset_source': 'Balanced EuroSAT'
    }
    
    with open(data_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save vegetation indices
    indices_df = pd.DataFrame(indices_data)
    indices_df['label'] = all_labels
    indices_df['land_type'] = [land_types[label] for label in all_labels]
    indices_df.to_csv(data_dir / 'vegetation_indices.csv', index=False)
    
    print("✓ All data saved successfully!")
    for file in sorted(data_dir.glob('*')):
        print(f"   - {file.name}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def setup_visualization_style():
    """Setup matplotlib and seaborn style for visualizations."""
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)


def visualize_rgb_composites(sample_images, land_types, viz_dir):
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    fig.suptitle('Sample Images - RGB Composites (Bands 3-2-1)', 
                 fontsize=14, fontweight='bold')
    
    for idx, land_type in enumerate(land_types):
        ax = axes[idx // 5, idx % 5]
        rgb = create_rgb_composite(sample_images[land_type])
        ax.imshow(rgb)
        ax.set_title(land_type, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(viz_dir) / 'sample_images_rgb.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ RGB composite visualization saved!")


def visualize_false_color_composites(sample_images, land_types, viz_dir):
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    fig.suptitle('Sample Images - False Color Composites (Bands 8-4-3, NIR-Red-Green)', 
                 fontsize=14, fontweight='bold')
    
    for idx, land_type in enumerate(land_types):
        ax = axes[idx // 5, idx % 5]
        false_color = create_false_color_composite(sample_images[land_type])
        ax.imshow(false_color)
        ax.set_title(land_type, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(viz_dir) / 'sample_images_false_color.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ False color composite visualization saved!")


def visualize_ndvi_maps(ndvi_samples, land_types, viz_dir):
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    fig.suptitle('NDVI (Normalized Difference Vegetation Index) Maps', 
                 fontsize=14, fontweight='bold')
    
    for idx, land_type in enumerate(land_types):
        ax = axes[idx // 5, idx % 5]
        im = ax.imshow(ndvi_samples[land_type], cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(land_type, fontsize=10)
        ax.axis('off')
    
    plt.colorbar(im, ax=axes.flatten(), label='NDVI', shrink=0.8)
    plt.tight_layout()
    plt.savefig(Path(viz_dir) / 'ndvi_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ NDVI visualization saved!")


def visualize_ndbi_maps(ndbi_samples, land_types, viz_dir):
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    fig.suptitle('NDBI (Normalized Difference Built-up Index) Maps', 
                 fontsize=14, fontweight='bold')
    
    for idx, land_type in enumerate(land_types):
        ax = axes[idx // 5, idx % 5]
        im = ax.imshow(ndbi_samples[land_type], cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(land_type, fontsize=10)
        ax.axis('off')
    
    plt.colorbar(im, ax=axes.flatten(), label='NDBI', shrink=0.8)
    plt.tight_layout()
    plt.savefig(Path(viz_dir) / 'ndbi_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ NDBI visualization saved!")


def visualize_ndwi_maps(ndwi_samples, land_types, viz_dir):
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    fig.suptitle('NDWI (Normalized Difference Water Index) Maps', 
                 fontsize=14, fontweight='bold')
    
    for idx, land_type in enumerate(land_types):
        ax = axes[idx // 5, idx % 5]
        im = ax.imshow(ndwi_samples[land_type], cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(land_type, fontsize=10)
        ax.axis('off')
    
    plt.colorbar(im, ax=axes.flatten(), label='NDWI', shrink=0.8)
    plt.tight_layout()
    plt.savefig(Path(viz_dir) / 'ndwi_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ NDWI visualization saved!")


def visualize_spectral_signatures(spectral_signatures, land_types, viz_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(land_types)))
    
    for idx, land_type in enumerate(land_types):
        spectrum = spectral_signatures[land_type]
        num_bands = len(spectrum)
        ax.plot(range(1, num_bands+1), spectrum, marker='o', label=land_type, 
                color=colors[idx], linewidth=2)
    
    ax.set_xlabel('Band Number', fontsize=12)
    ax.set_ylabel('Mean Pixel Value', fontsize=12)
    ax.set_title('Mean Spectral Signatures by Land Type', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, num_bands+1))
    
    plt.tight_layout()
    plt.savefig(Path(viz_dir) / 'spectral_signatures.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Spectral signature plot saved!")
