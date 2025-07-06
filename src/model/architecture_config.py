from model.vit_architecture import ViT, MultiModalViT
from model.cnn_architecture import MultiModalCNN
import config

def get_vit_model():
    """
    Returns a ViT model instance with consistent hyperparameters for training and prediction.
    """
    return ViT(
        img_size=config.IMAGE_HEIGHT,
        patch_size=8,
        in_chans=config.IMAGE_CHANNELS,
        num_classes=1,
        embed_dim=96,
        spatial_depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.2,  # Use 0.1 for prediction if needed, or make this configurable
        attn_dropout=0.2,
        drop_path=0.2,
        mlp_hidden_dim=128
    )

def get_multimodal_vit_model():
    """
    Returns a MultiModalViT model instance with consistent hyperparameters for training and prediction.
    """
    return MultiModalViT(
        img_size=config.IMAGE_HEIGHT,
        patch_size=8,
        in_chans=config.IMAGE_CHANNELS,
        num_classes=1,
        embed_dim=96,
        optical_depth=4,
        sar1_depth=4,
        sar2_depth=4,
        fusion_depth=4,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.2,
        attn_dropout=0.2,
        drop_path=0.2,
        mlp_hidden_dim=128
    )

def get_multimodal_cnn_model(fc_units=128, dropout=0.3, final_dropout=0.3):
    """
    Returns a MultiModalCNN model instance for multi-modal (optical + SAR) input (PyTorch version).
    Args:
        fc_units (int): Number of units in the fully connected layers.
        dropout (float): Dropout rate for feature extraction layers.
        final_dropout (float): Dropout rate after the final fully connected layer.
    """
    return MultiModalCNN(fc_units=fc_units, dropout=dropout, final_dropout=final_dropout)

# Example usage:
# model = get_multimodal_cnn_model(fc_units=128, dropout=0.3, final_dropout=0.5)
