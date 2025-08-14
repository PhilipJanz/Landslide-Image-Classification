import numpy as np
import torch
from torch.utils.data import Dataset

from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import config


class LandslideDataset(Dataset):
    """Dataset for loading processed landslide detection images - loads all data into memory."""
    
    def __init__(self, image_dir, 
                 csv_path, 
                 transform=None, 
                 filter_obvious_negatives = True,
                 band_selection=None,
                 device="cpu"
                 ):
        """
        Args:
            image_dir: Directory containing processed .npy files
            csv_path: Path to CSV file with image IDs and labels
            transform: Optional transform to apply to images
            filter_obvious_negatives: Ignore the obvious negative images (based on feature model) eg. to shrink training data  
            band_selection: Optional list of band indices to select from the images
            device: Device to load data onto ("cpu" or "cuda")
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.device = device
        self.band_selection = band_selection
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)

        if filter_obvious_negatives:
            # load feature model predictions
            if "train" in image_dir.name.lower():
                feature_model_preds = pd.read_csv(config.PROCESSED_FEATURE_PATH / "train_prediction.csv")
            else:
                feature_model_preds = pd.read_csv(config.PROCESSED_FEATURE_PATH / "test_prediction.csv")
            obvious_neg_image_ids = feature_model_preds.ID.values[feature_model_preds.label == 0]
            self.df = self.df[[id not in obvious_neg_image_ids for id in self.df.ID]]

        self.image_ids = self.df['ID'].values
        if "label" in self.df.columns:
            self.labels = self.df['label'].values.astype(np.float32)
        else:
            self.labels = np.empty(len(self.df))
        
        # Load all images into memory at once
        self.images = []
        
        for _, img_id in enumerate(tqdm(self.image_ids, desc="Loading images")):
            img_path = self.image_dir / f"{img_id}.npy"
            # Load image and convert to tensor
            image = np.load(img_path).astype(np.float32)
            image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
            
            # Apply band selection if specified
            if self.band_selection is not None:
                image = image[self.band_selection, :, :]
            
            # Move to device if specified
            if device != "cpu":
                image = image.to(device)
            
            self.images.append(image)
        
        print(f"Total memory usage: {sum(img.element_size() * img.nelement() for img in self.images) / 1024**3:.2f} GB")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get valid index
        label = self.labels[idx] 
        image = self.images[idx]  # Already loaded in memory
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class TransformedSubset(Dataset):
    """
    A wrapper for a Subset that applies a transform.
    
    Args:
        subset (Subset): The subset of the dataset.
        transform (callable, optional): A function/transform to be applied to the image.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)
    