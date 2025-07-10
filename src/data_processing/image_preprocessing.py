import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TRAIN_IMAGE_DIR, TEST_IMAGE_DIR, PROCESSED_DATA_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, SEED

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

def calculate_band_statistics(image_files, num_bands=12):
    """
    Calculate mean and std for each band across all images.
    
    Args:
        image_files: List of paths to .npy files
        num_bands: Number of bands in each image
    
    Returns:
        band_means: Array of means for each band
        band_stds: Array of standard deviations for each band
    """
    print("Calculating band-wise statistics...")
    
    # Initialize arrays to accumulate sums and squared sums
    band_sums = np.zeros(num_bands, dtype=np.float64)
    band_sq_sums = np.zeros(num_bands, dtype=np.float64)
    total_pixels = 0
    
    # Process all images
    for file_path in tqdm(image_files, desc="Computing statistics"):
        img = load_npy_file(file_path)
        if img is None:
            continue
            
        # Ensure correct shape
        if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, num_bands):
            print(f"Warning: {file_path} has shape {img.shape}, expected {(IMAGE_HEIGHT, IMAGE_WIDTH, num_bands)}")
            continue
        
        # Accumulate sums for each band
        for band in range(num_bands):
            band_data = img[:, :, band].flatten()
            band_sums[band] += np.sum(band_data)
            band_sq_sums[band] += np.sum(band_data ** 2)
        
        total_pixels += IMAGE_HEIGHT * IMAGE_WIDTH
    
    # Calculate means and standard deviations
    band_means = band_sums / total_pixels
    band_vars = (band_sq_sums / total_pixels) - (band_means ** 2)
    band_stds = np.sqrt(band_vars)
    
    # Handle zero standard deviations
    band_stds = np.where(band_stds == 0, 1.0, band_stds)
    
    print("Band-wise statistics:")
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
    # create prior images from diff 
    img[:, :, 6] = img[:, :, 4] - img[:, :, 6]
    img[:, :, 7] = img[:, :, 5] - img[:, :, 7]
    img[:, :, 10] = img[:, :, 8] - img[:, :, 10]
    img[:, :, 11] = img[:, :, 9] - img[:, :, 11]
    for band in range(img.shape[2]):
        if np.isin(band, [6, 7, 10, 11]): # prior images get the same trafo like their post images
            normalized[:, :, band] = (img[:, :, band] - band_means[band - 2]) / band_stds[band - 2]
        else:
            normalized[:, :, band] = (img[:, :, band] - band_means[band]) / band_stds[band]
    return normalized

def process_dataset(input_dir, output_dir, band_means, band_stds, dataset_name):
    """
    Process a dataset (train or test) by normalizing all images using global statistics.
    Args:
        input_dir: Directory containing input .npy files
        output_dir: Directory to save processed images
        band_means: Array of means for each band
        band_stds: Array of standard deviations for each band
        dataset_name: Name of the dataset for logging
    """
    print(f"\nProcessing {dataset_name} dataset...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = get_all_npy_files(input_dir)
    if not image_files:
        print(f"No .npy files found in {input_dir}")
        return
    processed_count = 0
    for file_path in tqdm(image_files, desc=f"Processing {dataset_name}"):
        img = load_npy_file(file_path)
        if img is None:
            continue
        if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
            print(f"Skipping {file_path.name}: wrong shape {img.shape}")
            continue
        normalized_img = normalize_image(img, band_means, band_stds)
        output_path = output_dir / file_path.name
        np.save(output_path, normalized_img.astype(np.float32))
        processed_count += 1
    print(f"Processed {processed_count} images for {dataset_name} dataset")

def main():
    """Main preprocessing pipeline."""
    print("Starting image preprocessing pipeline...")
    print(f"Using seed {SEED} for reproducibility")
    np.random.seed(SEED)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Get all training images for statistics calculation
    train_files = get_all_npy_files(TRAIN_IMAGE_DIR)
    if not train_files:
        print("No training files found! Cannot calculate statistics.")
        return
    # Calculate band-wise statistics from training data only
    band_means, band_stds = calculate_band_statistics(train_files, IMAGE_CHANNELS)
    # Save statistics for future use
    stats_path = PROCESSED_DATA_DIR / "band_statistics.npz"
    np.savez(stats_path, means=band_means, stds=band_stds)
    print(f"Saved band statistics to {stats_path}")
    # Process training dataset
    train_output_dir = PROCESSED_DATA_DIR / "train"
    process_dataset(TRAIN_IMAGE_DIR, train_output_dir, band_means, band_stds, "training")
    # Process test dataset
    test_output_dir = PROCESSED_DATA_DIR / "test"
    process_dataset(TEST_IMAGE_DIR, test_output_dir, band_means, band_stds, "test")
    print("\nPreprocessing completed successfully!")
    print(f"Processed images saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
