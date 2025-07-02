import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
import warnings
import math
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import config
from model.architecture_config import get_multimodal_cnn_model

class LandslideDataset(Dataset):
    """Dataset for loading processed landslide detection images - loads all data into memory."""
    
    def __init__(self, image_dir, csv_path, transform=None, device="cpu"):
        """
        Args:
            image_dir: Directory containing processed .npy files
            csv_path: Path to CSV file with image IDs and labels
            transform: Optional transform to apply to images
            device: Device to load data onto ("cpu" or "cuda")
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.device = device
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        self.image_ids = self.df['ID'].values
        self.labels = self.df['label'].values.astype(np.float32)
        
        # Load all images into memory at once
        self.images = []
        self.valid_indices = []
        
        for i, img_id in enumerate(tqdm(self.image_ids, desc="Loading images")):
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
        
        print(f"Loaded {len(self.valid_indices)} valid images out of {len(self.image_ids)}")
        print(f"Total memory usage: {sum(img.element_size() * img.nelement() for img in self.images) / 1024**3:.2f} GB")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get valid index
        valid_idx = self.valid_indices[idx]
        label = self.labels[valid_idx]
        image = self.images[idx]  # Already loaded in memory
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CosineAnnealingWarmupScheduler:
    """Cosine annealing scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy and F1 score."""
    # Convert to numpy arrays
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, average='binary')
    recall = recall_score(y_true, y_pred_binary, average='binary')
    precision = precision_score(y_true, y_pred_binary, average='binary')
    return accuracy, recall, precision, f1

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Images are already on the correct device (loaded in memory)
        targets = targets.to(device).unsqueeze(1)  # Add channel dimension
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Collect predictions and targets
        predictions = torch.sigmoid(outputs).detach()
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        total_loss += loss.item()
    
    # Calculate metrics
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()
    accuracy, recall, precision, f1 = calculate_metrics(all_targets, all_predictions)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, f1, recall, precision

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Images are already on the correct device (loaded in memory)
            targets = targets.to(device).unsqueeze(1)  # Add channel dimension
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Collect predictions and targets
            predictions = torch.sigmoid(outputs)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            total_loss += loss.item()
    
    # Calculate metrics
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()
    accuracy, recall, precision, f1 = calculate_metrics(all_targets, all_predictions)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, f1, recall, precision

def train_model(fc_units=128, dropout=0.4, final_dropout=0.4, lr=0.001, weight_decay=3e-5, show_process=True, save_model=True):
    """Main training function with 5-fold cross-validation and no early stopping."""
    print(f"Using device: {config.DEVICE}")
    device = torch.device(config.DEVICE)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {config.SEED} for reproducibility")

    # Create dataset with data loaded into memory
    print("Loading dataset into memory...")
    dataset = LandslideDataset(
        image_dir=config.PROCESSED_TRAIN_IMAGE_DIR,
        csv_path=config.TRAIN_CSV_PATH,
        device=device
    )
    total_size = len(dataset)
    indices = np.arange(total_size)
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)

    # Prepare model directory
    model_dir = config.MODEL_DIR / config.MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\nFold {fold+1}/5")
        print("-" * 50)
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(
            train_subset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(config.SEED)
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        model = get_multimodal_cnn_model(fc_units=fc_units, dropout=dropout, final_dropout=final_dropout).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        pos_weight = torch.tensor([5.0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        warmup_epochs = int(0.1 * config.EPOCHS)
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=config.EPOCHS,
            min_lr=1e-6
        )
        train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s = [], [], [], [], [], []
        for epoch in range(config.EPOCHS):
            if show_process:
                print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
                print("-" * 50)
            train_loss, train_acc, train_f1, train_recall, train_precision = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            current_lr = scheduler.step(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            train_f1s.append(train_f1)
            if show_process:
                val_loss, val_acc, val_f1, val_recall, val_precision = validate_epoch(
                    model, val_loader, criterion, device
                )
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                val_f1s.append(val_f1)
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Rec: {train_recall:.4f}, Train Prec: {train_precision:.4f}, Train F1: {train_f1:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Rec: {val_recall:.4f}, Val Prec: {val_precision:.4f}, Val F1: {val_f1:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")
        if save_model:
            # Save model at last epoch
            model_path = model_dir / f"{config.MODEL_NAME}_{fold}.pth"
            torch.save({
                'epoch': config.EPOCHS-1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'train_f1s': train_f1s,
                'val_f1s': val_f1s,
            }, model_path)
            print(f"Saved model for fold {fold} to {model_path}")
        val_loss, val_acc, val_f1, val_recall, val_precision = validate_epoch(
                    model, val_loader, criterion, device
                )
        print(f"Final Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        fold_metrics.append({
            'fold': fold,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })
    print("\nCross-validation complete!")
    for m in fold_metrics:
        print(f"Fold {m['fold']}: Val Loss={m['val_loss']:.4f}, Val Acc={m['val_acc']:.4f}, Val F1={m['val_f1']:.4f}")
    return fold_metrics

if __name__ == "__main__":
    train_model()
