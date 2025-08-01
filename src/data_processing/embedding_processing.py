import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import torch
from torch import nn
from torchvision.transforms import v2, transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from claymodel.module import ClayMAEModule

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import config
from config import SEED, EMBEDDING_DATA_DIR, TRAIN_IMAGE_DIR, TEST_IMAGE_DIR, FM_MODEL_DIR
from utils.dataset_loader import LandslideDataset

def run_clay(model, images, waves, device):
    datacube = {
        "pixels": images.to(device),              # shape: [B, C, H, W]
        "time": torch.zeros(len(images), 6).to(device),    # shape: [B, 2] — week, hour
        "latlon": torch.zeros(len(images), 2).to(device),  # shape: [B, 2] — lat, lon
        "gsd": torch.tensor([10.0], device=device),  # shape: [1]
        "waves": waves.to(device),         # shape: [C] or [1, C]
    }

    # Generate embeddings
    with torch.no_grad():
        embeddings, _, _, _ = model.model.encoder(datacube)
    
    return embeddings


class SelectBands:
    def __init__(self, band_ixs):
        self.band_ixs = band_ixs

    def __call__(self, x):
        # x is a tensor of shape [C, H, W]
        return x[self.band_ixs]

def process_dataset(output_dir, dataset_name, model, metadata, device):
    """
    Process a dataset (train or test) by running it through the clay FM for beautiful embeddings
    Args:
        output_dir: Directory to save processed embeddings
        dataset_name: Name of the dataset for logging
        TODO
    """
    print(f"\nProcessing {dataset_name} dataset...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset with data loaded into memory
    print("Loading dataset into memory...")
    if dataset_name == "train":
        dataset = LandslideDataset(
            image_dir=config.TRAIN_IMAGE_DIR,
            csv_path=config.TRAIN_CSV_PATH,
            filter_obvious_negatives=False,
            device=device
        )
    else:
        dataset = LandslideDataset(
            image_dir=config.TEST_IMAGE_DIR,
            csv_path=config.TEST_CSV_PATH,
            filter_obvious_negatives=False,
            device=device
        )

    # Apply normalization
    sensor= "sentinel-2"
    band_ixs = [0, 1, 2, 3]
    bands = metadata[sensor]["band_order"]
    mean = [metadata[sensor]["bands"]["mean"][b] for b in bands]
    std = [metadata[sensor]["bands"]["std"][b] for b in bands]
    dataset.transform = transforms.Compose([
        SelectBands(band_ixs),
        transforms.Normalize(mean, std),
    ])
    waves = torch.tensor([metadata[sensor]["bands"]["wavelength"][b] * 1000 for b in bands], dtype=torch.float32).to(device) 
    
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # list of embeddings
    embedding_ls = []
    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Processing {dataset_name} batches")):
        embeddings = run_clay(model, images, waves, device)
        # only get CLS embedding
        cls_embeddings = embeddings[:, 1:, :].mean(axis=1)
        embedding_ls.append(cls_embeddings)

    embedding_tensor = torch.concat(embedding_ls)
    
    # Save embeddings to file
    sensor = "sentinel-2"
    filename = f"{dataset_name}_{sensor}_embeddings.pt"
    save_path = output_dir / filename
    torch.save(embedding_tensor, save_path)
    print(f"Saved embeddings to: {save_path}")
    print(f"Embedding tensor shape: {embedding_tensor.shape}")
    
    return embedding_tensor 


def main():
    """Pipeline for embedding calculation using the clay FM"""
    print("Starting clay embedding pipeline...")
    print(f"Using seed {SEED} for reproducibility")
    np.random.seed(SEED)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load clay FM
    model = ClayMAEModule.load_from_checkpoint(FM_MODEL_DIR / "clay-v1.5.ckpt", metadata_path=config.PROJECT_ROOT / "configs/metadata.yaml")
    model.eval()
    # prepare encoder
    model.model.encoder.mask_ratio = 0

    # Load sensor metadata
    with open(config.PROJECT_ROOT / "configs/metadata.yaml", "r") as f:
        metadata = yaml.safe_load(f)

    EMBEDDING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Process training dataset
    train_output_dir = EMBEDDING_DATA_DIR / "train"
    process_dataset(train_output_dir, "train", model, metadata, device)

    # Process test dataset
    test_output_dir = EMBEDDING_DATA_DIR / "test"
    process_dataset(test_output_dir, "test", model, metadata, device)

    print("\nEmbedding preprocessing completed successfully!")

if __name__ == "__main__":
    main()
