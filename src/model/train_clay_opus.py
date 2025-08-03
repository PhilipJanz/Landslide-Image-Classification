import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
import math
from sklearn.model_selection import KFold
from claymodel.module import ClayMAEModule
from peft import LoraConfig, get_peft_model, TaskType
import yaml

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import config
from utils.dataset_loader import LandslideDataset, TransformedSubset
from utils.augmentation import DataAugmentationTransform

# Default hyperparameters
DEFAULT_PARAMS = {
    # Model configuration
    'use_peft': False, #True,  # If False, will do full fine-tuning
    'lora_r': 32,  # LoRA rank
    'lora_alpha': 64,  # LoRA alpha scaling
    'lora_dropout': 0.1,  # LoRA dropout
    'lora_target_modules': None,  # Will be auto-detected
    
    # Training parameters
    'lr_encoder': 1e-5,  # Learning rate for encoder (pre-trained backbone)
    'lr_head': 1e-3,     # Learning rate for classification head (100x encoder lr)
    'weight_decay': 1e-4,
    'batch_size': 32,
    'epochs': 200,
    'early_stopping_patience': 15,
    'warmup_epochs': 10,
    
    # Head architecture
    'head_hidden_dim': 512,
    'head_dropout': 0.25,
    
    # Other
    'gradient_clip': 1.0,
}

def find_lora_target_modules(model):
    """Find suitable target modules for LoRA in the model"""
    target_modules = []
    
    # Print model structure to understand naming
    print("\nModel structure:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            print(f"  {name}: {module.__class__.__name__}")
    
    # Look for attention-related modules
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Common patterns in transformer models
            if any(pattern in name.lower() for pattern in ['qkv', 'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 'to_q', 'to_k', 'to_v']):
                target_modules.append(name)
                print(f"Found attention module: {name}")
    
    # If no attention modules found, look for any Linear layers in the encoder
    if not target_modules:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'encoder' in name:
                target_modules.append(name)
        
        # Limit to a reasonable number of modules
        if len(target_modules) > 10:
            # Focus on MLP layers which are typically good for LoRA
            mlp_modules = [name for name in target_modules if 'mlp' in name.lower()]
            if mlp_modules:
                target_modules = mlp_modules[:6]  # Take up to 6 MLP modules
    
    print(f"\nSelected LoRA target modules: {target_modules}")
    return target_modules

class ClayClassificationModel(nn.Module):
    """Clay model with classification head"""
    def __init__(self, num_classes, hidden_dim=512, dropout=0.25):
        super().__init__()
        
        # Load pre-trained Clay model
        self.clay = ClayMAEModule.load_from_checkpoint(
            config.FM_MODEL_DIR / "clay-v1.5.ckpt",
            metadata_path=config.PROJECT_ROOT / "configs/metadata.yaml"
        )
        self.clay.eval()  # Set to eval mode initially
        
        # Get the dimension of Clay embeddings
        self.embedding_dim = 1024  
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize head weights
        self._init_head_weights()
    
    def _init_head_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, pixels, waves):
        # Prepare datacube for Clay
        datacube = {
            "pixels": pixels,
            "time": torch.zeros(len(pixels), 6).to(pixels.device),
            "latlon": torch.zeros(len(pixels), 2).to(pixels.device),
            "gsd": torch.tensor([10.0], device=pixels.device),
            "waves": waves,
        }
        
        # Get embeddings from Clay
        with torch.set_grad_enabled(self.training):
            embeddings, _, _, _ = self.clay.model.encoder(datacube)
            # Use CLS token embedding
            cls_embedding = embeddings[:, 0, :]
        
        # Pass through classification head
        logits = self.head(cls_embedding)
        return logits

def setup_model_for_training(model, params, device):
    """Setup model for either PEFT or full fine-tuning"""
    
    if params['use_peft']:
        # Freeze Clay encoder initially
        for param in model.clay.parameters():
            param.requires_grad = False
        
        # Auto-detect target modules if not specified
        if params['lora_target_modules'] is None:
            target_modules = find_lora_target_modules(model.clay)
            if not target_modules:
                print("Warning: No suitable LoRA target modules found. Falling back to full fine-tuning.")
                params['use_peft'] = False
                # Unfreeze all parameters for full fine-tuning
                for param in model.clay.parameters():
                    param.requires_grad = True
            else:
                # Configure LoRA
                lora_config = LoraConfig(
                    r=params['lora_r'],
                    lora_alpha=params['lora_alpha'],
                    lora_dropout=params['lora_dropout'],
                    target_modules=target_modules,
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                
                # Apply LoRA to the Clay model
                model.clay = get_peft_model(model.clay, lora_config)
                print("\nLoRA configuration applied:")
                model.clay.print_trainable_parameters()
        else:
            # Use provided target modules
            lora_config = LoraConfig(
                r=params['lora_r'],
                lora_alpha=params['lora_alpha'],
                lora_dropout=params['lora_dropout'],
                target_modules=params['lora_target_modules'],
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            model.clay = get_peft_model(model.clay, lora_config)
            model.clay.print_trainable_parameters()
    else:
        # Full fine-tuning - unfreeze all parameters
        print("Performing full fine-tuning")
        for param in model.clay.parameters():
            param.requires_grad = True
    
    # Head is always trainable
    for param in model.head.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model.to(device)

class CosineAnnealingWarmupScheduler:
    """Cosine annealing scheduler with warmup for multiple parameter groups"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_encoder=0, min_lr_head=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_encoder = min_lr_encoder
        self.min_lr_head = min_lr_head
        self.base_lr_encoder = optimizer.param_groups[0]['lr']
        self.base_lr_head = optimizer.param_groups[1]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_encoder = self.base_lr_encoder * (epoch + 1) / self.warmup_epochs
            lr_head = self.base_lr_head * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_encoder = self.min_lr_encoder + (self.base_lr_encoder - self.min_lr_encoder) * 0.5 * (1 + math.cos(math.pi * progress))
            lr_head = self.min_lr_head + (self.base_lr_head - self.min_lr_head) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # Update learning rates for both parameter groups
        self.optimizer.param_groups[0]['lr'] = lr_encoder
        self.optimizer.param_groups[1]['lr'] = lr_head
        
        return lr_encoder, lr_head

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate accuracy and F1 score"""
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    y_pred_binary = (y_pred > threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, average='binary')
    recall = recall_score(y_true, y_pred_binary, average='binary')
    precision = precision_score(y_true, y_pred_binary, average='binary')
    
    return accuracy, recall, precision, f1

def train_epoch(model, dataloader, criterion, optimizer, device, gradient_clip=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Get wavelengths for Clay
    with open(config.PROJECT_ROOT / "configs/metadata.yaml", "r") as f:
        metadata = yaml.safe_load(f)
    
    # Clay expects 4 bands for Sentinel-2
    sensor = "sentinel-2"
    bands = metadata[sensor]["band_order"][:4]  # Use first 4 bands
    waves = torch.tensor(
        [metadata[sensor]["bands"]["wavelength"][b] * 1000 for b in bands], 
        dtype=torch.float32
    ).to(device)
    
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        targets = targets.to(device).float()
        
        # Select the appropriate bands
        if images.shape[1] > 4:
            images = images[:, :4, :, :]
        
        optimizer.zero_grad()
        outputs = model(images, waves)
        loss = criterion(outputs.squeeze(), targets)
        
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        predictions = torch.sigmoid(outputs).detach()
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        total_loss += loss.item()
    
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()
    accuracy, recall, precision, f1 = calculate_metrics(all_targets, all_predictions)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, f1, recall, precision

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Get wavelengths for Clay
    with open(config.PROJECT_ROOT / "configs/metadata.yaml", "r") as f:
        metadata = yaml.safe_load(f)
    
    sensor = "sentinel-2"
    bands = metadata[sensor]["band_order"][:4]
    waves = torch.tensor(
        [metadata[sensor]["bands"]["wavelength"][b] * 1000 for b in bands], 
        dtype=torch.float32
    ).to(device)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Validation")):
            images = images.to(device)
            targets = targets.to(device).float()
            
            # Select the appropriate bands
            if images.shape[1] > 4:
                images = images[:, :4, :, :]
            
            outputs = model(images, waves)
            loss = criterion(outputs.squeeze(), targets)
            
            predictions = torch.sigmoid(outputs)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            total_loss += loss.item()
    
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()
    accuracy, recall, precision, f1 = calculate_metrics(all_targets, all_predictions)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, f1, recall, precision, all_predictions, all_targets

def train_model(**params):
    """Main training function that accepts hyperparameters"""
    
    # Setup
    print(f"Using device: {config.DEVICE}")
    device = torch.device(config.DEVICE)
    
    # Set random seeds
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Create dataset
    print("Loading dataset...")
    dataset = LandslideDataset(
        image_dir=config.PROCESSED_TRAIN_IMAGE_DIR,
        csv_path=config.TRAIN_CSV_PATH,
        device='cpu'  # Load to CPU, move to GPU in batches
    )
    
    # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"\nFOLD {fold+1}/5")
        print("-" * 50)
        
        # Create data loaders
        train_subset = TransformedSubset(
            torch.utils.data.Subset(dataset, train_idx),
            transform=DataAugmentationTransform()
        )
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        # Create model
        model = ClayClassificationModel(
            num_classes=1,  # Binary classification
            hidden_dim=params['head_hidden_dim'],
            dropout=params['head_dropout']
        )
        
        # Setup for training (PEFT or full fine-tuning)
        model = setup_model_for_training(model, params, device)
        
        # Loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer with different learning rates for encoder and head
        encoder_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'head' in name:
                    head_params.append(param)
                else:
                    encoder_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': encoder_params, 'lr': params['lr_encoder'], 'weight_decay': params['weight_decay']},
            {'params': head_params, 'lr': params['lr_head'], 'weight_decay': params['weight_decay']}
        ]
        
        optimizer = optim.AdamW(param_groups)
        
        # Print parameter group information
        encoder_param_count = sum(p.numel() for p in encoder_params)
        head_param_count = sum(p.numel() for p in head_params)
        print(f"\nParameter groups:")
        print(f"  Encoder parameters: {encoder_param_count:,} (lr: {params['lr_encoder']})")
        print(f"  Head parameters: {head_param_count:,} (lr: {params['lr_head']})")
        
        # Learning rate scheduler
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=params['warmup_epochs'],
            total_epochs=params['epochs'],
            min_lr_encoder=1e-6,
            min_lr_head=1e-4
        )
        
        # Training loop
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(params['epochs']):
            print(f"\nEpoch {epoch+1}/{params['epochs']}")
            
            # Update learning rate
            current_lr_encoder, current_lr_head = scheduler.step(epoch)
            
            # Train
            train_loss, train_acc, train_f1, train_recall, train_precision = train_epoch(
                model, train_loader, criterion, optimizer, device, params['gradient_clip']
            )
            
            # Validate
            val_loss, val_acc, val_f1, val_recall, val_precision, _, _ = validate_epoch(
                model, val_loader, criterion, device
            )
            
            print(f"Trn Loss: {train_loss:.4f}, Trn Precision: {train_precision:.4f}, Trn Recall: {train_recall:.4f}, Trn F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
            print(f"LR Encoder: {current_lr_encoder:.6f}, LR Head: {current_lr_head:.6f}")
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= params['early_stopping_patience']:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        val_loss, val_acc, val_f1, val_recall, val_precision, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        
        fold_metrics.append({
            'fold': fold,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_recall': val_recall,
            'val_precision': val_precision
        })

        # Explicitly delete the model, optimizer, and scheduler to free memory
        del model
        del optimizer
        del scheduler
        del best_model_state # This is on CPU but good practice
        
        # Empty the CUDA cache to release all unused cached memory
        torch.cuda.empty_cache()
    
    # Calculate average metrics
    avg_f1 = np.mean([m['val_f1'] for m in fold_metrics])
    
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)
    for m in fold_metrics:
        print(f"Fold {m['fold']+1}: F1={m['val_f1']:.4f}")
    print(f"\nAverage F1: {avg_f1:.4f}")
    
    return fold_metrics

if __name__ == "__main__":
    # Use default parameters
    train_model(**DEFAULT_PARAMS)