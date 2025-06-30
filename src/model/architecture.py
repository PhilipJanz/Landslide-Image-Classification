import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import math

class HybridPatchEmbed(nn.Module):
    """
    Hybrid Patch Embedding using multiple Conv2d layers with smaller kernels.
    For 64x64 input, outputs 64 tokens (8x8 spatial) with richer features.
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=12, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 64 for 64x64 with 8x8 patches
        # 2 conv layers: 3x3 stride 2 (->32x32), 3x3 stride 2 (->16x16), 3x3 stride 2 (->8x8)
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),        # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv(x)  # (B, embed_dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DropPath(nn.Module):
    """Stochastic Depth (DropPath) regularization."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class SpatialEncoder(nn.Module):
    """
    Spatial Transformer Encoder operating on spatial patches.
    """
    def __init__(self, embed_dim=192, depth=2, num_heads=6, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.1, drop_path=0.1):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, num_heads, mlp_ratio, dropout, attn_dropout, dpr[i]
            ) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, num_patches, embed_dim)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention block: queries from one modality attend to keys/values of the other.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv):
        # q, kv: (B, N, D)
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm)
        return self.drop(attn_out)

class MultiModalViT(nn.Module):
    """
    Multi-Modal Vision Transformer with three separate encoders:
    - Optical: channels 0-3
    - SAR1: channels 4-7
    - SAR2: channels 8-11
    Cross-attention between all pairs, then fusion and joint encoding.
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=12, num_classes=1, embed_dim=192,
                 optical_depth=2, sar1_depth=2, sar2_depth=2, fusion_depth=2, num_heads=6, mlp_ratio=4.0,
                 dropout=0.1, attn_dropout=0.1, drop_path=0.1, mlp_hidden_dim=768):
        super().__init__()
        # Patch embedding for each modality
        self.optical_patch_embed = HybridPatchEmbed(img_size, patch_size, 4, embed_dim)
        self.sar1_patch_embed = HybridPatchEmbed(img_size, patch_size, 4, embed_dim)
        self.sar2_patch_embed = HybridPatchEmbed(img_size, patch_size, 4, embed_dim)
        self.num_patches = self.optical_patch_embed.num_patches  # 64

        # Positional embeddings (shared for all streams)
        self.optical_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.sar1_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.sar2_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # CLS tokens for each stream
        self.optical_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.sar1_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.sar2_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Separate encoders
        self.optical_encoder = SpatialEncoder(embed_dim, optical_depth, num_heads, mlp_ratio, dropout, attn_dropout, drop_path)
        self.sar1_encoder = SpatialEncoder(embed_dim, sar1_depth, num_heads, mlp_ratio, dropout, attn_dropout, drop_path)
        self.sar2_encoder = SpatialEncoder(embed_dim, sar2_depth, num_heads, mlp_ratio, dropout, attn_dropout, drop_path)

        # Cross-attention blocks (symmetric, all pairs)
        self.cross_attn_opt_sar1 = CrossAttentionBlock(embed_dim, num_heads, dropout, attn_dropout)
        self.cross_attn_sar1_opt = CrossAttentionBlock(embed_dim, num_heads, dropout, attn_dropout)
        self.cross_attn_opt_sar2 = CrossAttentionBlock(embed_dim, num_heads, dropout, attn_dropout)
        self.cross_attn_sar2_opt = CrossAttentionBlock(embed_dim, num_heads, dropout, attn_dropout)
        self.cross_attn_sar1_sar2 = CrossAttentionBlock(embed_dim, num_heads, dropout, attn_dropout)
        self.cross_attn_sar2_sar1 = CrossAttentionBlock(embed_dim, num_heads, dropout, attn_dropout)

        # Fusion: concatenate all three streams, then joint transformer blocks
        self.fusion_depth = fusion_depth
        self.joint_encoder = SpatialEncoder(embed_dim * 3, fusion_depth, num_heads, mlp_ratio, dropout, attn_dropout, drop_path)

        # CLS token and positional embedding for fusion
        self.fusion_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim * 3))
        self.fusion_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim * 3))

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim * 3, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.optical_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.sar1_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.sar2_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.optical_cls_token, std=0.02)
        nn.init.trunc_normal_(self.sar1_cls_token, std=0.02)
        nn.init.trunc_normal_(self.sar2_cls_token, std=0.02)
        nn.init.trunc_normal_(self.fusion_cls_token, std=0.02)
        nn.init.trunc_normal_(self.fusion_pos_embed, std=0.02)
        for m in self.classification_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        # Split modalities
        x_opt = x[:, :4, :, :]    # (B, 4, H, W)
        x_sar1 = x[:, 4:8, :, :]  # (B, 4, H, W)
        x_sar2 = x[:, 8:12, :, :] # (B, 4, H, W)
        # Patch embedding
        patches_opt = self.optical_patch_embed(x_opt)    # (B, num_patches, embed_dim)
        patches_sar1 = self.sar1_patch_embed(x_sar1)     # (B, num_patches, embed_dim)
        patches_sar2 = self.sar2_patch_embed(x_sar2)     # (B, num_patches, embed_dim)
        # Add CLS tokens
        cls_opt = self.optical_cls_token.expand(B, -1, -1)
        cls_sar1 = self.sar1_cls_token.expand(B, -1, -1)
        cls_sar2 = self.sar2_cls_token.expand(B, -1, -1)
        tokens_opt = torch.cat((cls_opt, patches_opt), dim=1)
        tokens_sar1 = torch.cat((cls_sar1, patches_sar1), dim=1)
        tokens_sar2 = torch.cat((cls_sar2, patches_sar2), dim=1)
        # Add positional embeddings
        tokens_opt = tokens_opt + self.optical_pos_embed
        tokens_sar1 = tokens_sar1 + self.sar1_pos_embed
        tokens_sar2 = tokens_sar2 + self.sar2_pos_embed
        # Separate encoders
        encoded_opt = self.optical_encoder(tokens_opt)
        encoded_sar1 = self.sar1_encoder(tokens_sar1)
        encoded_sar2 = self.sar2_encoder(tokens_sar2)
        # Remove CLS for cross-attn, keep for later
        opt_cls, opt_patches = encoded_opt[:, :1, :], encoded_opt[:, 1:, :]
        sar1_cls, sar1_patches = encoded_sar1[:, :1, :], encoded_sar1[:, 1:, :]
        sar2_cls, sar2_patches = encoded_sar2[:, :1, :], encoded_sar2[:, 1:, :]
        # Cross-attention (all pairs, symmetric)
        # Optical
        opt_cross = opt_patches \
            + self.cross_attn_opt_sar1(opt_patches, sar1_patches) \
            + self.cross_attn_opt_sar2(opt_patches, sar2_patches)
        # SAR1
        sar1_cross = sar1_patches \
            + self.cross_attn_sar1_opt(sar1_patches, opt_patches) \
            + self.cross_attn_sar1_sar2(sar1_patches, sar2_patches)
        # SAR2
        sar2_cross = sar2_patches \
            + self.cross_attn_sar2_opt(sar2_patches, opt_patches) \
            + self.cross_attn_sar2_sar1(sar2_patches, sar1_patches)
        # Re-attach CLS tokens
        fused_opt = torch.cat((opt_cls, opt_cross), dim=1)
        fused_sar1 = torch.cat((sar1_cls, sar1_cross), dim=1)
        fused_sar2 = torch.cat((sar2_cls, sar2_cross), dim=1)
        # Concatenate along feature dim (for each patch)
        fused = torch.cat((fused_opt, fused_sar1, fused_sar2), dim=2)  # (B, num_patches+1, embed_dim*3)
        # Fusion positional embedding and CLS
        fusion_cls = self.fusion_cls_token.expand(B, -1, -1)
        fused = fused + self.fusion_pos_embed
        fused = torch.cat((fusion_cls, fused[:, 1:, :]), dim=1)  # Replace first token with fusion CLS
        # Joint encoder
        fused = self.joint_encoder(fused)
        # CLS pooling
        cls_out = fused[:, 0]
        logits = self.classification_head(cls_out)
        return logits

class ViT(nn.Module):
    """
    Multi-Spatial Vision Transformer (ViT) - Simplified architecture
    
    Architecture:
    1. Hybrid patch embedding: 8x8 patches (64 patches for 64x64 images)
    2. Spatial encoder: 2 layers for spatial feature extraction
    3. CLS pooling and classification head for image-wise prediction
    4. Sigmoid for binary classification probabilities
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=12, num_classes=1, embed_dim=192, 
                 spatial_depth=2, num_heads=6, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.1, drop_path=0.1,
                 mlp_hidden_dim=768):
        super().__init__()
        self.patch_embed = HybridPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches  # 64

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embeddings (add 1 for CLS token)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Spatial Encoder
        self.spatial_encoder = SpatialEncoder(embed_dim, spatial_depth, num_heads, mlp_ratio, dropout, attn_dropout, drop_path)
        
        # Classification head for image-wise prediction
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Initialize classification head
        for m in self.classification_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - batch of single images
        Returns:
            logits: (B, num_classes) - image-wise class logits
        """
        B, C, H, W = x.shape
        
        # Step 1: Patch embedding
        patches = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Expand and prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, patches), dim=1)    # (B, num_patches+1, embed_dim)
        
        # Add positional embeddings
        x = x + self.spatial_pos_embed  # (B, num_patches+1, embed_dim)
        
        # Step 2: Spatial encoding
        encoded = self.spatial_encoder(x)  # (B, num_patches+1, embed_dim)
        
        # Step 3: CLS pooling and classification head
        cls_out = encoded[:, 0]  # (B, embed_dim)
        
        # Classification head
        logits = self.classification_head(cls_out)  # (B, num_classes)
        
        return logits
