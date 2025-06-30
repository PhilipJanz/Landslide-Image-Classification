import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import glob

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import config
from model.architecture_config import get_vit_model

class TestDataset(Dataset):
    """Dataset for loading processed test images."""
    
    def __init__(self, image_dir, csv_path, device="cpu"):
        """
        Args:
            image_dir: Directory containing processed .npy files
            csv_path: Path to CSV file with image IDs
            device: Device to load data onto ("cpu" or "cuda")
        """
        self.image_dir = Path(image_dir)
        self.device = device
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        self.image_ids = self.df['ID'].values
        
        # Load all images into memory at once
        print("Loading all test images into memory...")
        self.images = []
        self.valid_indices = []
        
        for i, img_id in enumerate(tqdm(self.image_ids, desc="Loading test images")):
            img_path = self.image_dir / f"{img_id}.npy"
            if img_path.exists():
                # Load image and convert to tensor
                image = np.load(img_path).astype(np.float32)
                image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
                
                # Move to device if specified
                if device != "cpu":
                    image = image.to(device)
                
                self.images.append(image)
                self.valid_indices.append(i)
            else:
                print(f"Warning: {img_path} not found")
        
        print(f"Loaded {len(self.valid_indices)} valid test images out of {len(self.image_ids)}")
        print(f"Total memory usage: {sum(img.element_size() * img.nelement() for img in self.images) / 1024**3:.2f} GB")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get valid index
        valid_idx = self.valid_indices[idx]
        img_id = self.image_ids[valid_idx]
        image = self.images[idx]  # Already loaded in memory
        
        return image, img_id

def predict_model():
    """Main prediction function."""
    print(f"Using device: {config.DEVICE}")
    device = torch.device(config.DEVICE)
    
    # Create test dataset with data loaded into memory
    print("Loading test dataset into memory...")
    test_dataset = TestDataset(
        image_dir=config.PROCESSED_TEST_IMAGE_DIR,
        csv_path=config.TEST_CSV_PATH,
        device=device  # Load directly to GPU if available
    )
    
    if len(test_dataset) == 0:
        print("No valid test samples found. Aborting prediction.")
        return
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0  # No need for multiple workers since data is in memory
    )
    
    # Load all model checkpoints from the cross-validation folder
    model_dir = config.MODEL_DIR / config.MODEL_NAME
    model_paths = sorted(glob.glob(str(model_dir / f"{config.MODEL_NAME}_*.pth")))
    if not model_paths:
        print(f"No model checkpoints found in {model_dir}")
        return
    print(f"Found {len(model_paths)} model checkpoints for ensembling.")
    all_probs = []
    for model_path in model_paths:
        print(f"Loading model from {model_path}")
        model = get_vit_model().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        probs = []
        with torch.no_grad():
            for images, image_ids in test_loader:
                outputs = model(images)
                probabilities = torch.sigmoid(outputs).squeeze()
                probs.append(probabilities.cpu().numpy())
        all_probs.append(np.concatenate(probs))
    # Average probabilities across all models
    avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    # Get image IDs in order
    all_image_ids = [img_id for _, img_id in test_dataset]
    # Convert to binary predictions
    predictions = (avg_probs > 0.5).astype(float)
    submission_df = pd.DataFrame({
        'ID': all_image_ids,
        'label': predictions
    })
    config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    model_name_without_ext = config.MODEL_NAME.replace('.pth', '')
    submission_file_path = config.SUBMISSIONS_DIR / f"submission_{model_name_without_ext}.csv"
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to {submission_file_path}")
    print(f"\nPrediction Statistics:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Positive predictions (landslide): {sum(predictions)}")
    print(f"Negative predictions (no landslide): {len(predictions) - sum(predictions)}")
    print(f"Positive rate: {sum(predictions) / len(predictions) * 100:.2f}%")

if __name__ == "__main__":
    predict_model()