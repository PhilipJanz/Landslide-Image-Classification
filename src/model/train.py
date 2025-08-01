import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import sys
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
import warnings
import math
from sklearn.model_selection import KFold
from codecarbon import track_emissions


#warnings.filterwarnings('ignore')

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import config
from model.model_config import get_multimodal_cnn_model
from model.experiment import ImprovedMultiModalFPN
from utils.augmentation import DataAugmentationTransform
from utils.visualizations import create_training_plots, create_summary_plot
from utils.dataset_loader import LandslideDataset, TransformedSubset

    
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

def calculate_metrics(y_true, y_pred, threshold=.5):
    """Calculate accuracy and F1 score."""
    # Convert to numpy arrays
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    if threshold is None: # optimize based on f1
        threshold_ls = np.arange(20, 81) / 100
        f1_ls = [f1_score(y_true, (y_pred > threshold).astype(int), average='binary') for threshold in threshold_ls]
        threshold = threshold_ls[np.argmax(f1_ls)]
        print(f"F1-optimal threshold: {threshold:.4f} F1: {np.max(f1_ls):.4f}")

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, average='binary')
    recall = recall_score(y_true, y_pred_binary, average='binary')
    precision = precision_score(y_true, y_pred_binary, average='binary')
    return accuracy, recall, precision, f1, threshold


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
    accuracy, recall, precision, f1, _ = calculate_metrics(all_targets, all_predictions)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, f1, recall, precision

def validate_epoch(model, dataloader, criterion, device, threshold=0.5):
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

    accuracy, recall, precision, f1, threshold = calculate_metrics(all_targets, all_predictions, threshold=threshold)
    return accuracy, f1, recall, precision, all_predictions, all_targets, threshold


#@track_emissions()
def train_model(fc_units=256, 
                fusioned_kernel_units=128,
                dropout=0.0, 
                final_dropout=0.25, 
                lr=config.LEARNING_RATE, 
                weight_decay=1e-4, 
                bce_weight=1.0,
                batch_size=config.BATCH_SIZE,
                show_process=True, 
                save_model=True):
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
    kf = KFold(n_splits=10, shuffle=True, random_state=config.SEED)

    # Prepare model directory
    model_dir = config.MODEL_DIR / config.MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create data augmentation transform
    augmentation_transform = DataAugmentationTransform()

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\nFOLD {fold+1}/5")
        print("-" * 50)
        
        # Create subsets from the same dataset
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create a transformed subset for training data
        train_subset_with_augmentation = TransformedSubset(train_subset, transform=augmentation_transform)
        
        # The original dataset's transform can remain None
        dataset.transform = None
        
        train_loader = DataLoader(
            train_subset_with_augmentation,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(config.SEED)
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        model = get_multimodal_cnn_model(fc_units=fc_units, fusioned_kernel_units=fusioned_kernel_units, dropout=dropout, final_dropout=final_dropout).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        pos_weight = torch.tensor([bce_weight]).to(device)
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
                val_acc, val_f1, val_recall, val_precision, _, _, _ = validate_epoch(
                    model, val_loader, criterion, device
                )
                val_accuracies.append(val_acc)
                val_f1s.append(val_f1)
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Rec: {train_recall:.4f}, Train Prec: {train_precision:.4f}, Train F1: {train_f1:.4f}")
                print(f"Val Acc: {val_acc:.4f}, Val Rec: {val_recall:.4f}, Val Prec: {val_precision:.4f}, Val F1: {val_f1:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")
            
        _, _, _, _, _, _, f1_opt_threshold = validate_epoch(
                    model, train_loader, criterion, device, threshold=None
                )
        val_acc, val_f1, val_recall, val_precision, all_predictions, all_targets, _ = validate_epoch(
                    model, val_loader, criterion, device, threshold=f1_opt_threshold
                )
    
        if save_model:
            # Save model at last epoch
            model_path = model_dir / f"{config.MODEL_NAME}_{fold}.pth"
            torch.save({
                'epoch': config.EPOCHS-1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "train_idx": train_idx,
                "val_idx": val_idx,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'train_f1s': train_f1s,
                'val_f1s': val_f1s,
                "final_val_predictions": all_predictions,
                "val_targets": all_targets,
                "f1_opt_threshold": f1_opt_threshold,
            }, model_path)
            # Create training plots for this fold
            create_training_plots(train_accuracies, val_accuracies, 
                                train_f1s, val_f1s, fold, model_dir)
        print(f"Final Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} (at threshold {f1_opt_threshold})")
        fold_metrics.append({
            'fold': fold,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_recall': val_recall,
            'val_precision': val_precision
        })
        
    
    # Calculate and display average metrics across all folds
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    # Individual fold results
    print("\nIndividual Fold Results:")
    print("-" * 60)
    for m in fold_metrics:
        print(f"Fold {m['fold']+1}: Acc={m['val_acc']:.4f}, F1={m['val_f1']:.4f}, "
              f"Recall={m['val_recall']:.4f}, Precision={m['val_precision']:.4f}")
    
    # Calculate averages
    avg_acc = np.mean([m['val_acc'] for m in fold_metrics])
    avg_recall = np.mean([m['val_recall'] for m in fold_metrics])
    avg_precision = np.mean([m['val_precision'] for m in fold_metrics])
    avg_f1 = np.mean([m['val_f1'] for m in fold_metrics])
    
    # Calculate standard deviations
    std_acc = np.std([m['val_acc'] for m in fold_metrics])
    std_f1 = np.std([m['val_f1'] for m in fold_metrics])
    std_recall = np.std([m['val_recall'] for m in fold_metrics])
    std_precision = np.std([m['val_precision'] for m in fold_metrics])
    
    # Display average results
    print("\nAverage Results Across All Folds:")
    print("-" * 60)
    print(f"Validation Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Validation Recall:   {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"Validation Precision:{avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Validation F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
    
    # Create summary plot
    create_summary_plot(fold_metrics, model_dir)
    
    return fold_metrics

if __name__ == "__main__":
    train_model()
