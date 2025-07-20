import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBranch(nn.Module):
    def __init__(self, in_channels, dropout=0.4):
        super().__init__()
        # Input: (B, in_channels, 64, 64)
        self.convolution_blocks = nn.Sequential(
            # BLOCK 1
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # -> (B, 16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # BLOCK 2
            nn.Conv2d(16, 16, kernel_size=3, padding=1),           # -> (B, 16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            # BLOCK 3                                    # -> (B, 16, 32, 32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),           # -> (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # BLOCK 4
            nn.Conv2d(32, 32, kernel_size=3, padding=1),           # -> (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),    
            # BLOCK 5                                  # -> (B, 32, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # -> (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # -> (B, 64, 8, 8)
            nn.Dropout(dropout),
            # BLOCK 6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # -> (B, 64, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # -> (B, 128, 4, 4)
            nn.Dropout(dropout),
            # POOLING
            nn.AdaptiveAvgPool2d(1),     # -> (B, 128, 1, 1)
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.convolution_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

class MultiModalCNN(nn.Module):
    def __init__(self, fc_units=128, dropout=0.4, final_dropout=0.4):
        super().__init__()
        self.optical_branch = CNNBranch(4, dropout) # -> (B, 128)
        self.sar_encoder = CNNBranch(4, dropout) # -> (B, 128)
        self.sar_change_detector = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.final_fc_head = nn.Sequential(
            nn.Linear(256, fc_units),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            nn.Linear(fc_units, 1),
        )

    def forward(self, x):
        # x: (B, 10, 64, 64)
        optical_features = self.optical_branch(x[:, 0:4, :, :])      # (B, 128)
        
        # Process SAR pre and post imagery using the shared encoder
        sar_pre_features = self.sar_encoder(x[:, [6, 7, 10, 11], :, :])
        sar_post_features = self.sar_encoder(x[:, [4, 5, 8, 9], :, :])

        # Concatenate SAR features and pass through the change detection module
        combined_sar_features = torch.cat((sar_pre_features, sar_post_features), dim=1)
        sar_change_features = self.sar_change_detector(combined_sar_features)

        # Concatenate optical features with the learned SAR change features
        final_combined_features = torch.cat((optical_features, sar_change_features), dim=1)

        # Pass through the final classification head
        output = self.final_fc_head(final_combined_features)                           # (B, 1)
        return output

class SARPretrainModel(nn.Module):
    def __init__(self, sar_encoder, sar_change_detector):
        super().__init__()
        self.sar_encoder = sar_encoder
        self.sar_change_detector = sar_change_detector
        self.sar_output_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Assume x shape: (B, 10, 64, 64)
        sar_pre = self.sar_encoder(x[:, [6, 7, 10, 11], :, :])
        sar_post = self.sar_encoder(x[:, [4, 5, 8, 9], :, :])
        sar_combined = torch.cat([sar_pre, sar_post], dim=1)
        features = self.sar_change_detector(sar_combined)
        output = self.sar_output_head(features)
        return output

class OpticalPretrainModel(nn.Module):
    def __init__(self, optical_branch):
        super().__init__()
        self.optical_branch = optical_branch
        self.optical_output_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (B, 10, 64, 64)
        optical_input = x[:, 0:4, :, :]
        features = self.optical_branch(optical_input)
        return self.optical_output_head(features)
