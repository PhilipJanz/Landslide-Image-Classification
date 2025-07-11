import torch
import torchvision.transforms as T


class DataAugmentationTransform:
    """Data augmentation transform for landslide images using PyTorch utilities."""

    def __init__(self,
                 p_hflip=0.5,
                 p_vflip=0.5,
                 p_erase=0.3,
                 p_rotate=0.5,
                 p_crop=0.5,
                 crop_scale=(0.6, 1.0)):
        """
        Args:
            p_hflip: Probability of horizontal flip
            p_vflip: Probability of vertical flip
            p_erase: Probability of patch erasing
            p_rotate: Probability of random 90-degree rotation
            p_crop: Probability of applying RandomResizedCrop
            crop_scale: Tuple (min_scale, max_scale) for crop area
        """
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_erase = p_erase
        self.p_rotate = p_rotate
        
        """
        self.p_crop = p_crop
        self.random_resized_crop = T.RandomResizedCrop(
            size=(64, 64),
            scale=crop_scale,
            ratio=(1.0, 1.0),  # Keep square aspect ratio
            antialias=True
        )"""

    def __call__(self, image, random=True, hflip=False, vflip=False, rotate=False):
        """
        Apply random augmentations to the input image.
        Args:
            image (Tensor): Image tensor of shape (C, H, W)
        Returns:
            Tensor: Augmented image
        """
        if random:
            # Horizontal flip
            if torch.rand(1) < self.p_hflip:
                image = torch.flip(image, dims=[2])

            # Vertical flip
            if torch.rand(1) < self.p_vflip:
                image = torch.flip(image, dims=[1])

            # Random 90° rotation
            if torch.rand(1) < self.p_rotate:
                image = torch.rot90(image, k=1, dims=[1, 2])

            # RandomResizedCrop
            #if torch.rand(1) < self.p_crop:
            #    image = self.random_resized_crop(image)

            # Erase patch
            if torch.rand(1) < self.p_erase:
                image = self.erase_patch(image)
        else:
            if hflip:
                image = torch.flip(image, dims=[2])
            if vflip:
                image = torch.flip(image, dims=[1])
            if rotate:
                image = torch.rot90(image, k=1, dims=[1, 2])

        return image

    def erase_patch(self, image, size=4):
        """Randomly zero out a small patch."""
        C, H, W = image.shape
        y = torch.randint(0, H - size, (1,)).item()
        x = torch.randint(0, W - size, (1,)).item()
        image[:, y:y + size, x:x + size] = 0
        return image
