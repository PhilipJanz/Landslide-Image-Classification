import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBranch(nn.Module):
    def __init__(self, in_channels, dropout=0.4):
        super().__init__()
        # Input: (B, in_channels, 64, 64)
        self.features = nn.Sequential(
            # BLOCK 1
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # -> (B, 16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # BLOCK 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),           # -> (B, 32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        x = self.features(x)
        return x
    
class MultiModalCNN(nn.Module):
    def __init__(self, fc_units=128, fusioned_kernel_units=64, dropout=0.4, final_dropout=0.4):
        super().__init__()
        self.optical_branch = CNNBranch(5, dropout)
        #self.sar_desc_branch = CNNBranch(4, dropout)
        self.sar_branch = CNNBranch(4, dropout)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32 * 3, 64, kernel_size=3, padding=1),           # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 32x32
            nn.Dropout2d(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),          # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 16x16
            nn.Dropout2d(dropout),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),         # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 8x8
            nn.Dropout2d(dropout),
        )

        # Lateral connections
        self.lat4 = nn.Conv2d(256, fusioned_kernel_units, kernel_size=1)  # -> P4
        self.lat3 = nn.Conv2d(128, fusioned_kernel_units, kernel_size=1)  # -> P3
        self.lat2 = nn.Conv2d(64, fusioned_kernel_units, kernel_size=1)   # -> P2

        # Output smoothing convs
        self.smooth2 = nn.Conv2d(fusioned_kernel_units, fusioned_kernel_units, kernel_size=3, padding=1)

        # Classifier head
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusioned_kernel_units * 3, fc_units),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            nn.Linear(fc_units, 1)  # Binary classification
        )

    def forward(self, x):
        # x: (B, 13, 64, 64)
        # filter visual spectrum based on cloud mask
        #x[:, 1:5, :, :] = x[:, 1:5, :, :] - (x[:, 1:5, :, :] * x[:, [0], :, :])
        x_opt = self.optical_branch(x[:, 0:5, :, :])      # (B, 32, 32, 32)
        x_sar_desc = self.sar_branch(x[:, 5:9, :, :]) # (B, 32, 32, 32)
        x_sar_asc = self.sar_branch(x[:, 9:, :, :]) # (B, 32, 32, 32)
        c1 = torch.cat([x_opt, x_sar_desc, x_sar_asc], dim=1)  # (B, 96, 32, 32) 

        # Bottom-up pathway
        c2 = self.conv2(c1)    # 16x16
        c3 = self.conv3(c2)    # 8×8
        c4 = self.conv4(c3)    # 4x4

        # Top-down pathway
        p4 = self.lat4(c4)  # 64×4x4
        p3 = self.lat3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')  # 64×8×8
        p2 = self.lat2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')  # 64×16x16

        # Final output smoothing
        p4 = self.global_pool(self.smooth2(p4))  # B x fusioned_kernel_units
        p3 = self.global_pool(self.smooth2(p3))  # B x fusioned_kernel_units
        p2 = self.global_pool(self.smooth2(p2))  # B x fusioned_kernel_units

        # Unite and classify classifier
        out = torch.cat([p4, p3, p2], dim=1)  # (B, fusioned_kernel_units * 3)
        out = self.classifier(out)   # B×1
        return out
