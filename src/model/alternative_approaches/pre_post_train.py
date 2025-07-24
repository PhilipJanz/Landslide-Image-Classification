import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
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
from codecarbon import track_emissions

warnings.filterwarnings('ignore')

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import config
from model.architecture_config import get_multimodal_cnn_model
from src.model.fpn_architecture import SARPretrainModel, OpticalPretrainModel
from utils.augmentation import DataAugmentationTransform
from utils.visualizations import create_training_plots, create_summary_plot


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

def calculate_metrics(y_true, y_pred, f1_optimal=False):
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

    if f1_optimal:
        threshold_ls = np.arange(30, 71) / 100
        f1_ls = [f1_score(y_true, (y_pred > threshold).astype(int), average='binary') for threshold in threshold_ls]
        print("F1-optimal threshold: ", threshold_ls[np.argmax(f1_ls)], "F1: ", np.max(f1_ls))
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

def pretrain_epoch(optical_model, sar_model, dataloader, criterion, 
                     optical_optimizer, sar_optimizer, device):
    """Train optical and SAR models for one epoch simultaneously."""
    
    optical_model.train()
    sar_model.train()
    total_loss_optical = 0
    total_loss_sar = 0
    
    all_predictions_optical = []
    all_predictions_sar = []
    all_targets = []

    for batch_idx, (images, targets) in enumerate(dataloader):
        targets = targets.to(device).unsqueeze(1)
        
        # ---- OPTICAL ----
        optical_optimizer.zero_grad()
        output_optical = optical_model(images.to(device))
        loss_optical = criterion(output_optical, targets)
        loss_optical.backward()
        optical_optimizer.step()
        
        # ---- SAR ----
        sar_optimizer.zero_grad()
        output_sar = sar_model(images.to(device))
        loss_sar = criterion(output_sar, targets)
        loss_sar.backward()
        sar_optimizer.step()
        
        # Collect predictions
        all_predictions_optical.extend(torch.sigmoid(output_optical).detach().cpu().numpy())
        all_predictions_sar.extend(torch.sigmoid(output_sar).detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        total_loss_optical += loss_optical.item()
        total_loss_sar += loss_sar.item()

    # Metrics
    pred_opt = np.array(all_predictions_optical).flatten()
    pred_sar = np.array(all_predictions_sar).flatten()
    true = np.array(all_targets).flatten()

    acc_opt, rec_opt, prec_opt, f1_opt = calculate_metrics(true, pred_opt)
    acc_sar, rec_sar, prec_sar, f1_sar = calculate_metrics(true, pred_sar)

    avg_loss_optical = total_loss_optical / len(dataloader)
    avg_loss_sar = total_loss_sar / len(dataloader)

    optical_scores = {
            'loss': avg_loss_optical,
            'accuracy': acc_opt,
            'f1': f1_opt,
            'recall': rec_opt,
            'precision': prec_opt
        }
    sar_scores = {
            'loss': avg_loss_sar,
            'accuracy': acc_sar,
            'f1': f1_sar,
            'recall': rec_sar,
            'precision': prec_sar
        }

    return optical_scores, sar_scores


def validate_epoch(model, dataloader, criterion, device, f1_optimal=False):
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
    accuracy, recall, precision, f1 = calculate_metrics(all_targets, all_predictions, f1_optimal=f1_optimal)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, f1, recall, precision, all_predictions, all_targets

def pretrain_validate_epoch(optical_model, sar_model, dataloader, criterion, device, f1_optimal=False):
    """Validate both optical and SAR models for one epoch."""
    optical_model.eval()
    sar_model.eval()

    total_loss_optical = 0
    total_loss_sar = 0

    all_predictions_optical = []
    all_predictions_sar = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            targets = targets.to(device).unsqueeze(1)

            # ---- OPTICAL ----
            outputs_optical = optical_model(images.to(device))
            loss_optical = criterion(outputs_optical, targets)
            preds_optical = torch.sigmoid(outputs_optical)

            # ---- SAR ----
            outputs_sar = sar_model(images.to(device))
            loss_sar = criterion(outputs_sar, targets)
            preds_sar = torch.sigmoid(outputs_sar)

            # Collect
            total_loss_optical += loss_optical.item()
            total_loss_sar += loss_sar.item()
            all_predictions_optical.extend(preds_optical.cpu().numpy())
            all_predictions_sar.extend(preds_sar.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Metrics
    pred_opt = np.array(all_predictions_optical).flatten()
    pred_sar = np.array(all_predictions_sar).flatten()
    true = np.array(all_targets).flatten()

    acc_opt, rec_opt, prec_opt, f1_opt = calculate_metrics(true, pred_opt, f1_optimal=f1_optimal)
    acc_sar, rec_sar, prec_sar, f1_sar = calculate_metrics(true, pred_sar, f1_optimal=f1_optimal)

    avg_loss_optical = total_loss_optical / len(dataloader)
    avg_loss_sar = total_loss_sar / len(dataloader)


    optical_scores = {
            'loss': avg_loss_optical,
            'accuracy': acc_opt,
            'f1': f1_opt,
            'recall': rec_opt,
            'precision': prec_opt,
            'predictions': pred_opt,
            'targets': true,
        }
    sar_scores = {
            'loss': avg_loss_sar,
            'accuracy': acc_sar,
            'f1': f1_sar,
            'recall': rec_sar,
            'precision': prec_sar,
            'predictions': pred_sar,
            'targets': true,
        }

    return optical_scores, sar_scores


#@track_emissions()
def train_model(fc_units=128, 
                dropout=0.55, 
                final_dropout=0.25, 
                lr=config.LEARNING_RATE, 
                weight_decay=2e-4, 
                bce_weight=2.0,
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
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)

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

        # Initialize multi-modal model 
        model = get_multimodal_cnn_model(fc_units=fc_units, dropout=dropout, final_dropout=final_dropout)
        # Initialize Sub-models
        sar_model = SARPretrainModel(model.sar_encoder, model.sar_change_detector).to(device)
        optical_model = OpticalPretrainModel(model.optical_branch).to(device)

        # PRE-TRAINING MODALITIES ----------------------------------------------------------------------------------

        # loss function & optimizer
        pos_weight = torch.tensor([bce_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optical_optimizer = optim.Adam(
            optical_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        sar_optimizer = optim.Adam(
            sar_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        warmup_epochs = int(0.1 * config.EPOCHS)
        optical_scheduler = CosineAnnealingWarmupScheduler(
            optimizer=optical_optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=config.EPOCHS,
            min_lr=1e-6
        )
        sar_scheduler = CosineAnnealingWarmupScheduler(
            optimizer=sar_optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=config.EPOCHS,
            min_lr=1e-6
        )
        
        optical_train_losses, optical_val_losses, optical_train_accuracies, optical_val_accuracies, optical_train_f1s, optical_val_f1s = [], [], [], [], [], []
        sar_train_losses, sar_val_losses, sar_train_accuracies, sar_val_accuracies, sar_train_f1s, sar_val_f1s = [], [], [], [], [], []
        for epoch in range(config.EPOCHS):
            if show_process:
                print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
                print("-" * 50)
            optical_scores, sar_scores = pretrain_epoch(
                optical_model, sar_model, train_loader, criterion, optical_optimizer, sar_optimizer, device
            )
            optical_current_lr = optical_scheduler.step(epoch)
            sar_current_lr = sar_scheduler.step(epoch)
            optical_train_losses.append(optical_scores["loss"])
            optical_train_accuracies.append(optical_scores["accuracy"])
            optical_train_f1s.append(optical_scores["f1"])
            sar_train_losses.append(sar_scores["loss"])
            sar_train_accuracies.append(sar_scores["accuracy"])
            sar_train_f1s.append(sar_scores["f1"])
            

            if show_process:
                optical_scores, sar_scores = pretrain_validate_epoch(
                    optical_model, sar_model, val_loader, criterion, device
                )
                optical_val_losses.append(optical_scores["loss"])
                optical_val_accuracies.append(optical_scores["accuracy"])
                optical_val_f1s.append(optical_scores["f1"])
                sar_val_losses.append(sar_scores["loss"])
                sar_val_accuracies.append(sar_scores["accuracy"])
                sar_val_f1s.append(sar_scores["f1"])
                print(f"Optical Train Loss: {optical_train_losses[-1]:.4f}, Train Acc: {optical_train_accuracies[-1]:.4f}, Train F1: {optical_train_f1s[-1]:.4f}")
                print(f"Optical Val Loss: {optical_val_losses[-1]:.4f}, Val Acc: {optical_val_accuracies[-1]:.4f}, Val F1: {optical_val_f1s[-1]:.4f}")
                print(f"SAR Train Loss: {sar_train_losses[-1]:.4f}, Train Acc: {sar_train_accuracies[-1]:.4f}, Train F1: {sar_train_f1s[-1]:.4f}")
                print(f"SAR Val Loss: {sar_val_losses[-1]:.4f}, Val Acc: {sar_val_accuracies[-1]:.4f}, Val F1: {sar_val_f1s[-1]:.4f}")
                print(f"Learning Rate (optical / SAR):  {optical_current_lr:.6f} / {sar_current_lr:.6f}")

        # POST-TRAINING MULTI-MODAL MODEL --------------------------------------------------------------------------

        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # copy pretrained weights
        model.sar_encoder.load_state_dict(sar_model.sar_encoder.state_dict())
        model.sar_change_detector.load_state_dict(sar_model.sar_change_detector.state_dict())
        model.optical_branch.load_state_dict(optical_model.optical_branch.state_dict())
        # To freeze pre-trained weights:
        """for p in model.optical_branch.parameters():
            p.requires_grad = False
        for p in model.sar_encoder.parameters():
            p.requires_grad = False
        for p in model.sar_change_detector.parameters():
            p.requires_grad = False"""

        # loss function & optimizer
        pos_weight = torch.tensor([bce_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam([
            {"params": model.optical_branch.parameters(), "lr": 1e-4, "weight_decay": weight_decay},
            {"params": model.sar_encoder.parameters(), "lr": 1e-4, "weight_decay": weight_decay},
            {"params": model.sar_change_detector.parameters(), "lr": 1e-4, "weight_decay": weight_decay},
            {"params": model.final_fc_head.parameters(), "lr": lr, "weight_decay": weight_decay},
        ], lr=lr)

        warmup_epochs = int(0.1 * config.EPOCHS)
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=config.EPOCHS,
            min_lr=1e-6
        )
        train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s = [], [], [], [], [], []
        for epoch in range(config.EPOCHS):
            # Unfreeze branches after some progress
            """if epoch == 20:
                for p in model.parameters():
                    p.requires_grad = True"""

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
                val_loss, val_acc, val_f1, val_recall, val_precision, _, _ = validate_epoch(
                    model, val_loader, criterion, device
                )
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                val_f1s.append(val_f1)
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Rec: {train_recall:.4f}, Train Prec: {train_precision:.4f}, Train F1: {train_f1:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Rec: {val_recall:.4f}, Val Prec: {val_precision:.4f}, Val F1: {val_f1:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")
            
        val_loss, val_acc, val_f1, val_recall, val_precision, all_predictions, all_targets = validate_epoch(
                    model, val_loader, criterion, device, f1_optimal=True
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
            }, model_path)
            # Create training plots for this fold
            create_training_plots(train_accuracies, val_accuracies, 
                                train_f1s, val_f1s, fold, model_dir)
        print(f"Final Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
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
    avg_f1 = np.mean([m['val_f1'] for m in fold_metrics])
    avg_recall = np.mean([m['val_recall'] for m in fold_metrics])
    avg_precision = np.mean([m['val_precision'] for m in fold_metrics])
    
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
