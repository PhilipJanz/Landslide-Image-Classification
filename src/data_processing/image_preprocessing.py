import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from scipy.ndimage import gaussian_filter

import warnings
warnings.filterwarnings('ignore')

# Add src to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TRAIN_IMAGE_DIR, TEST_IMAGE_DIR, PROCESSED_DATA_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, SEED


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def load_npy_file(file_path):
    """Load a single .npy file."""
    try:
        return np.load(file_path, allow_pickle=False)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_all_npy_files(directory):
    """Get all .npy files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory {directory} does not exist!")
        return []
    
    npy_files = list(directory.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files in {directory}")
    return npy_files

def calculate_band_statistics(image_files, num_bands=10):
    """
    Calculate mean and std for each band across all images, after SAR augmentation.
    Args:
        image_files: List of paths to .npy files
        num_bands: Number of bands in each image (should be 10 after SAR augmentation)
    Returns:
        band_means: Array of means for each band
        band_stds: Array of standard deviations for each band
    """
    print("Calculating band-wise statistics (after SAR augmentation)...")
    band_sums = np.zeros(num_bands, dtype=np.float64)
    band_sq_sums = np.zeros(num_bands, dtype=np.float64)
    total_pixels = 0
    for file_path in tqdm(image_files, desc="Computing statistics"):
        img = load_npy_file(file_path)
        if img is None:
            continue
        if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
            print(f"Warning: {file_path} has shape {img.shape}, expected {(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)}")
            continue
        # --- SAR augmentation (same as in process_dataset) ---
        desc_vv = img[:, :, 4]
        desc_vh = img[:, :, 5]
        desc_diff_vv = img[:, :, 6]
        desc_diff_vh = img[:, :, 7]
        asc_vv = img[:, :, 8]
        asc_vh = img[:, :, 9]
        asc_diff_vv = img[:, :, 10]
        asc_diff_vh = img[:, :, 11]
        desc_vv_sub = desc_vv - desc_diff_vv
        desc_vh_sub = desc_vh - desc_diff_vh
        asc_vv_sub = asc_vv - asc_diff_vv
        asc_vh_sub = asc_vh - asc_diff_vh
        desc_vv_vh = desc_vv_sub - desc_vh_sub
        asc_vv_vh = asc_vv_sub - asc_vh_sub
        img_new = np.zeros((img.shape[0], img.shape[1], 10), dtype=img.dtype)
        img_new[:, :, :4] = img[:, :, :4]
        img_new[:, :, 4] = desc_vv_sub
        img_new[:, :, 5] = desc_vh_sub
        img_new[:, :, 6] = asc_vv_sub
        img_new[:, :, 7] = asc_vh_sub
        img_new[:, :, 8] = desc_vv_vh
        img_new[:, :, 9] = asc_vv_vh
        # --- End SAR augmentation ---
        for band in range(num_bands):
            band_data = img_new[:, :, band].flatten()
            band_sums[band] += np.sum(band_data)
            band_sq_sums[band] += np.sum(band_data ** 2)
        total_pixels += IMAGE_HEIGHT * IMAGE_WIDTH
    band_means = band_sums / total_pixels
    band_vars = (band_sq_sums / total_pixels) - (band_means ** 2)
    band_stds = np.sqrt(band_vars)
    band_stds = np.where(band_stds == 0, 1.0, band_stds)
    print("Band-wise statistics (after SAR augmentation):")
    for i in range(num_bands):
        print(f"Band {i+1}: Mean = {band_means[i]:.6f}, Std = {band_stds[i]:.6f}")
    return band_means, band_stds

def normalize_image(img, band_means, band_stds):
    """
    Normalize an image using global band-wise statistics.
    Args:
        img: Input image array (H, W, C)
        band_means: Array of means for each band
        band_stds: Array of standard deviations for each band
    Returns:
        Normalized image as float32
    """
    normalized = np.zeros_like(img, dtype=np.float32)
    for band in range(img.shape[2]):
        normalized[:, :, band] = (img[:, :, band] - band_means[band]) / band_stds[band]
    return normalized

def process_dataset(input_dir, output_dir, dataset_name, band_means=None, band_stds=None):
    """
    Process a dataset (train or test) by normalizing all images using statistics computed from the new bands, or provided statistics.
    Args:
        input_dir: Directory containing input .npy files
        output_dir: Directory to save processed images
        dataset_name: Name of the dataset for logging
        band_means: Optional, array of means for each band
        band_stds: Optional, array of standard deviations for each band
    Returns:
        If band_means and band_stds are not provided, returns (band_means, band_stds) computed from the data.
    """
    print(f"\nProcessing {dataset_name} dataset...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = get_all_npy_files(input_dir)
    if not image_files:
        print(f"No .npy files found in {input_dir}")
        return (None, None) if band_means is None or band_stds is None else None
    imgs_new = []
    for file_path in tqdm(image_files, desc=f"Preparing {dataset_name} (for stats)"):
        img = load_npy_file(file_path)
        if img is None:
            continue
        if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
            print(f"Skipping {file_path.name}: wrong shape {img.shape}")
            continue
        desc_vv = img[:, :, 4]
        desc_vh = img[:, :, 5]
        desc_diff_vv = img[:, :, 6]
        desc_diff_vh = img[:, :, 7]
        asc_vv = img[:, :, 8]
        asc_vh = img[:, :, 9]
        asc_diff_vv = img[:, :, 10]
        asc_diff_vh = img[:, :, 11]
        desc_vv_sub = desc_vv - desc_diff_vv
        desc_vh_sub = desc_vh - desc_diff_vh
        asc_vv_sub = asc_vv - asc_diff_vv
        asc_vh_sub = asc_vh - asc_diff_vh
        desc_vv_vh = desc_vv_sub - desc_vh_sub
        asc_vv_vh = asc_vv_sub - asc_vh_sub
        img_new = np.zeros((img.shape[0], img.shape[1], 10), dtype=img.dtype)
        img_new[:, :, :4] = img[:, :, :4]
        img_new[:, :, 4] = desc_vv_sub
        img_new[:, :, 5] = desc_vh_sub
        img_new[:, :, 6] = desc_vv_vh
        img_new[:, :, 7] = asc_vv_sub
        img_new[:, :, 8] = asc_vh_sub
        img_new[:, :, 9] = asc_vv_vh
        imgs_new.append(img_new)
    imgs_new = np.stack(imgs_new, axis=0)
    if band_means is None or band_stds is None:
        band_means = imgs_new.mean(axis=(0, 1, 2))
        band_stds = imgs_new.std(axis=(0, 1, 2))
        band_stds = np.where(band_stds == 0, 1.0, band_stds)
        print("Band-wise statistics (from processed images):")
        for i in range(10):
            print(f"Band {i+1}: Mean = {band_means[i]:.6f}, Std = {band_stds[i]:.6f}")
    processed_count = 0
    for i, file_path in enumerate(tqdm(image_files, desc=f"Processing {dataset_name}")):
        img_new = imgs_new[i]
        normalized_img = normalize_image(img_new, band_means, band_stds)
        output_path = output_dir / file_path.name
        np.save(output_path, normalized_img.astype(np.float32))
        processed_count += 1
    print(f"Processed {processed_count} images for {dataset_name} dataset")
    return band_means, band_stds

def main():
    """Main preprocessing pipeline."""
    print("Starting image preprocessing pipeline...")
    print(f"Using seed {SEED} for reproducibility")
    np.random.seed(SEED)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Process training dataset and get statistics
    train_output_dir = PROCESSED_DATA_DIR / "train"
    band_means, band_stds = process_dataset(TRAIN_IMAGE_DIR, train_output_dir, "training")
    # Save statistics for future use
    stats_path = PROCESSED_DATA_DIR / "band_statistics.npz"
    np.savez(stats_path, means=band_means, stds=band_stds)
    print(f"Saved band statistics to {stats_path}")
    # Process test dataset using training statistics
    test_output_dir = PROCESSED_DATA_DIR / "test"
    process_dataset(TEST_IMAGE_DIR, test_output_dir, "test", band_means, band_stds)
    print("\nPreprocessing completed successfully!")
    print(f"Processed images saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
