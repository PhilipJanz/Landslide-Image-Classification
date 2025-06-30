from model.architecture import ViT, MultiModalViT
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
        spatial_depth=2,
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
