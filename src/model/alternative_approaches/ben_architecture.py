import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from peft import LoraConfig, TaskType
from peft.peft_model import PeftModelForFeatureExtraction
import sys

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from config import FM_MODEL_DIR

def apply_lora_to_resnet(model, r=4, alpha=1.0, dropout=0.1):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
            parent = model
            *path, last = name.split('.')
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, LoRAConv2d(module, r=r, lora_alpha=alpha, dropout=dropout))
    return model


def map_channels(tensor, channel_mapping):
    device = tensor.device  # Get device (CPU or CUDA)
    new_tensor = torch.zeros((tensor.shape[0], 12, tensor.shape[2], tensor.shape[3]),
                             dtype=tensor.dtype, device=device)
    for src_idx, dst_idx in channel_mapping.items():
        new_tensor[:, dst_idx, :, :] = tensor[:, src_idx, : ,:]
    return new_tensor

class LoRAConv2d(nn.Module):
    def __init__(self, original_layer, r=4, lora_alpha=1.0, dropout=0.0):
        super().__init__()
        self.original = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.scale = lora_alpha / r

        # Grab parameters from the original conv layer
        in_channels = original_layer.in_channels
        out_channels = original_layer.out_channels
        stride = original_layer.stride
        padding = original_layer.padding
        kernel_size = original_layer.kernel_size
        dilation = original_layer.dilation
        groups = original_layer.groups

        # Important: preserve all spatial properties
        self.lora_down = nn.Conv2d(in_channels, r, kernel_size=1, stride=1, padding=0, bias=False)
        self.lora_up = nn.Conv2d(r, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        # Weight init
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.original(x) + self.scale * self.lora_up(self.dropout(self.lora_down(x)))

class BigEarthNetFM(nn.Module):
    def __init__(self, fc_units=128, dropout=0.4):
        super().__init__()

        # load BigEarth Net foundation model
        pretrained_model_path = FM_MODEL_DIR / 'reBEN_resnet18-all-v0.2.0'
        pretrained_model = torch.load(pretrained_model_path, weights_only=False) # -> (B, 512)
        # remove BENs classification head
        pretrained_model.fc = nn.Identity()
        self.pretrained_model = apply_lora_to_resnet(pretrained_model, r=8, alpha=16, dropout=0.3)

        self.desc_post_dim_reduction = nn.Sequential(
            nn.Linear(512, fc_units),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.desc_pre_dim_reduction = nn.Sequential(
            nn.Linear(512, fc_units),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.asc_post_dim_reduction = nn.Sequential(
            nn.Linear(512, fc_units),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.asc_pre_dim_reduction = nn.Sequential(
            nn.Linear(512, fc_units),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.final_fc_head = nn.Sequential(
            nn.Linear(fc_units * 4, fc_units * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_units * 2, fc_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_units, 1),
        )

    def forward(self, x):
        # x: (B, 12, 120, 120)
        # bring x in right format for BEN
        desc_post_image = map_channels(x, {0: 3, 1: 2, 2: 1, 3: 7, 4: 11, 5: 10})
        desc_pre_image = map_channels(x, {0: 3, 1: 2, 2: 1, 3: 7, 6: 11, 7: 10})
        asc_post_image = map_channels(x, {0: 3, 1: 2, 2: 1, 3: 7, 8: 11, 9: 10})
        asc_pre_image = map_channels(x, {0: 3, 1: 2, 2: 1, 3: 7, 10: 11, 11: 10})

        # apply pretrained model and dimension reduction
        desc_post_emb = self.desc_post_dim_reduction(self.pretrained_model.forward(desc_post_image))
        desc_pre_emb = self.desc_pre_dim_reduction(self.pretrained_model.forward(desc_pre_image))
        asc_post_emb = self.asc_post_dim_reduction(self.pretrained_model.forward(asc_post_image))
        asc_pre_emb = self.asc_pre_dim_reduction(self.pretrained_model.forward(asc_pre_image))

        x = torch.cat([desc_post_emb, desc_pre_emb, asc_post_emb, asc_pre_emb], dim=1)  # (B, 4 * fc_units)
        x = self.final_fc_head(x)                             # (B, 1)
        return x
