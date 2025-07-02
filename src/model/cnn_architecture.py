import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBranch(nn.Module):
    def __init__(self, in_channels, dropout=0.4):
        super().__init__()
        # Input: (B, in_channels, 64, 64)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # -> (B, 16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),           # -> (B, 16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # -> (B, 16, 32, 32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),           # -> (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),           # -> (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # -> (B, 32, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # -> (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),           # -> (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # -> (B, 64, 8, 8)
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class MultiModalCNN(nn.Module):
    def __init__(self, fc_units=128, dropout=0.4, final_dropout=0.4):
        super().__init__()
        self.optical_branch = CNNBranch(4, dropout)
        self.sar_desc_branch = CNNBranch(3, dropout)
        self.sar_asc_branch = CNNBranch(3, dropout)
        # After concat: (B, 192, 8, 8)
        self.fusion_block = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # -> (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # -> (B, 128, 4, 4)
            nn.Dropout(dropout),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),   # -> (B, 64, 4, 4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)         # -> (B, 64, 1, 1)
        self.final_fc = nn.Linear(64, fc_units)
        self.final_dropout = nn.Dropout(final_dropout)
        self.classifier = nn.Linear(fc_units, 1)

    def forward(self, x):
        # x: (B, 10, 64, 64)
        x_opt = self.optical_branch(x[:, 0:4, :, :])      # (B, 64, 8, 8)
        x_sar_desc = self.sar_desc_branch(x[:, 4:7, :, :])# (B, 64, 8, 8)
        x_sar_asc = self.sar_asc_branch(x[:, 7:10, :, :]) # (B, 64, 8, 8)
        x = torch.cat([x_opt, x_sar_desc, x_sar_asc], dim=1)  # (B, 192, 8, 8)
        x = self.fusion_block(x)                              # (B, 64, 4, 4)
        x = self.global_pool(x)                               # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)                             # (B, 64)
        x = F.relu(self.final_fc(x))                          # (B, fc_units)
        x = self.final_dropout(x)
        x = self.classifier(x)
        return x

