import torch
import numpy as np


class DataAugmentationTransform:
    """Data augmentation transform for landslide images."""
    
    def __init__(self, p_hflip=0.5, p_vflip=0.5):
        """
        Args:
            p_hflip: Probability of horizontal flip (default: 0.5)
            p_vflip: Probability of vertical flip (default: 0.5)
        """
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
    
    def __call__(self, image, random=True, hflip=False, vflip=False):
        """
        Apply data augmentation to the image.
        
        Args:
            image: Tensor of shape (C, H, W)
            
        Returns:
            Augmented image tensor
        """
        if random:
            # Horizontal flip
            if torch.rand(1) < self.p_hflip:
                image = torch.flip(image, dims=[2])  # Flip along width dimension
            
            # Vertical flip
            if torch.rand(1) < self.p_vflip:
                image = torch.flip(image, dims=[1])  # Flip along height dimension
        else:
            if hflip:
                image = torch.flip(image, dims=[2])
            if vflip:
                image = torch.flip(image, dims=[1])
        
        return image
    