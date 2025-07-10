import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        out += x  # Residual connection
        return self.relu(out)

class CNNBranch(nn.Module):
    def __init__(self, in_channels, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Residual Block 1
            ResidualBlock(32),
            nn.MaxPool2d(2),  # -> (B, 32, 32, 32)

            # Increase channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Residual Block 2
            ResidualBlock(64),
            nn.MaxPool2d(2),  # -> (B, 64, 16, 16)

            nn.Dropout(dropout),

            # Increase channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Residual Block 3
            ResidualBlock(128),
            nn.MaxPool2d(2),  # -> (B, 128, 8, 8)

            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.features(x)


class MultiModalCNN(nn.Module):
    def __init__(self, fc_units=128, dropout=0.4, final_dropout=0.4):
        super().__init__()
        self.optical_branch = CNNBranch(4, dropout)
        self.sar_desc_branch = CNNBranch(4, dropout)
        self.sar_asc_branch = CNNBranch(4, dropout)

        self.fusion_block = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.MaxPool2d(2),  # -> (B, 256, 4, 4)
            nn.Dropout(dropout),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.final_fc = nn.Linear(128, fc_units)
        self.final_dropout = nn.Dropout(final_dropout)
        self.classifier = nn.Linear(fc_units, 1)

    def forward(self, x):
        x_opt = self.optical_branch(x[:, 0:4, :, :])
        x_sar_desc = self.sar_desc_branch(x[:, 4:8, :, :])
        x_sar_asc = self.sar_asc_branch(x[:, 8:, :, :])
        x = torch.cat([x_opt, x_sar_desc, x_sar_asc], dim=1)  # (B, 384, 8, 8)

        x = self.fusion_block(x)        # (B, 128, 4, 4)
        x = self.global_pool(x)         # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)       # (B, 128)
        x = F.relu(self.final_fc(x))    # (B, fc_units)
        x = self.final_dropout(x)
        x = self.classifier(x)          # (B, 1)
        return x
