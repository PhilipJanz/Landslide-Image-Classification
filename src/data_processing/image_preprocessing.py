import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TRAIN_IMAGE_DIR, TEST_IMAGE_DIR, PROCESSED_DATA_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, SEED, PROCESSED_FEATURE_PATH
import pandas as pd
from feature_preprocessing import extract_features
# Import BENv2_utils for statistics
from utils import BENv2_utils


def get_all_npy_files(directory):
    """Get all .npy files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory {directory} does not exist!")
        return []
    
    npy_files = list(directory.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files in {directory}")
    return npy_files


def normalize_image(img):
    """
    Normalize an image using BENv2_utils statistics and the specified band order.
    Args:
        img: Input image array (H, W, C)
    Returns:
        Normalized image as float32
    """
    # Create prior images from diff
    img[:, :, 6] = img[:, :, 4] - img[:, :, 6]
    img[:, :, 7] = img[:, :, 5] - img[:, :, 7]
    img[:, :, 10] = img[:, :, 8] - img[:, :, 10]
    img[:, :, 11] = img[:, :, 9] - img[:, :, 11]

    # Band mapping as specified
    band_names = ["B04", "B03", "B02", "B08", "VV", "VH", "VV", "VH", "VV", "VH", "VV", "VH"]
    means = BENv2_utils.means["no_interpolation"]
    stds = BENv2_utils.stds["no_interpolation"]

    band_stack = np.stack([((img[:, :, i] - means[band_names[i]]) / stds[band_names[i]]) 
                       for i in range(img.shape[2])], axis=2)
    # add rule based cloud mask
    cloud_mask = (band_stack[:,:,[0]] < 6) * 1
    #band_stack[:, :, :4] = band_stack[:, :, :4] * cloud_mask
    input_tensor = np.concat([cloud_mask, band_stack], axis=2)

    normalized = input_tensor.astype(np.float32)

    return normalized

def process_dataset(input_dir, output_dir, dataset_name):
    """
    Process a dataset (train or test) by normalizing all images using BENv2_utils statistics.
    Args:
        input_dir: Directory containing input .npy files
        output_dir: Directory to save processed images
        dataset_name: Name of the dataset for logging
    """
    print(f"\nProcessing {dataset_name} dataset...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = get_all_npy_files(input_dir)
    if not image_files:
        print(f"No .npy files found in {input_dir}")
        return
    file_names = []
    # list of features
    feature_ls = []
    feature_labels = None
    processed_count = 0
    for file_path in tqdm(image_files, desc=f"Processing {dataset_name}"):
        img = np.load(file_path, allow_pickle=False)
        if img is None:
            continue
        if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
            print(f"Skipping {file_path.name}: wrong shape {img.shape}")
            continue
        normalized_img = normalize_image(img)
        output_path = output_dir / file_path.name
        file_names.append(file_path.name[:-4])
        np.save(output_path, normalized_img.astype(np.float32))
        # calculate features for stage-1 decision-tree model
        features, feature_labels = extract_features(normalized_img)
        feature_ls.append(features)
        processed_count += 1
    # Save features as CSV
    if feature_ls and feature_labels is not None:
        df = pd.DataFrame(feature_ls, columns=feature_labels)
        csv_path = PROCESSED_FEATURE_PATH / f"{dataset_name}.csv"
        df.index = file_names
        df.to_csv(csv_path, index=True)
        print(f"Saved features to {csv_path}")
    print(f"Processed {processed_count} images for {dataset_name} dataset")

def main():
    """Main preprocessing pipeline using BENv2_utils statistics."""
    print("Starting image preprocessing pipeline...")
    print(f"Using seed {SEED} for reproducibility")
    np.random.seed(SEED)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Process training dataset
    train_output_dir = PROCESSED_DATA_DIR / "train"
    process_dataset(TRAIN_IMAGE_DIR, train_output_dir, "train")
    # Process test dataset
    test_output_dir = PROCESSED_DATA_DIR / "test"
    process_dataset(TEST_IMAGE_DIR, test_output_dir, "test")
    print("\nPreprocessing completed successfully!")
    print(f"Processed images saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
