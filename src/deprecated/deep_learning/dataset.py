import os
import json
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


class MinecraftTextureDataset(Dataset):
    """
    PyTorch Dataset for Minecraft block textures.
    
    Loads PNG images from minecraft/blocks/ and maps them to class labels.
    Supports train/val/test splits and optional augmentation.
    
    Args:
        root_dir (str): Path to minecraft/blocks/ directory
        split (str): One of 'train', 'val', 'test', or 'all'. Default: 'all'
        split_ratios (Tuple[float, float, float]): Train/val/test split ratios. Default: (0.7, 0.15, 0.15)
        augmentation (Optional[callable]): Augmentation function to apply. Default: None
        transform (Optional[callable]): Additional transforms (torchvision). Default: None
        size (Tuple[int, int]): Target image size. Default: (16, 16)
    """
    
    def __init__(self, root_dir: str, split: str = 'all', 
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 augmentation=None, transform=None, size: Tuple[int, int] = (16, 16)):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_ratios = split_ratios
        self.augmentation = augmentation
        self.transform = transform
        self.size = size
        
        # Find all PNG files
        self.png_files = sorted([f for f in self.root_dir.glob('*.png')])
        
        if not self.png_files:
            raise ValueError(f"No PNG files found in {self.root_dir}")
        
        print(f"Found {len(self.png_files)} PNG files")
        
        # Create class mapping (filename -> class index)
        self.class_to_idx = {f.stem: idx for idx, f in enumerate(self.png_files)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        # Handle data split
        if split != 'all':
            self._apply_split()
    
    def _apply_split(self):
        n_samples = len(self.png_files)
        
        # Create deterministic split using fixed random seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        
        n_train = int(n_samples * self.split_ratios[0])
        n_val = int(n_samples * self.split_ratios[1])
        
        # All classes are included in each split
        # Split divides into indices that determine augmentation seed/variant
        if self.split == 'train':
            self.split_indices = indices[:n_train]
        elif self.split == 'val':
            self.split_indices = indices[n_train:n_train+n_val]
        elif self.split == 'test':
            self.split_indices = indices[n_train+n_val:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        print(f"Split '{self.split}': {len(self.png_files)} classes (all 989 included)")
        print(f"  Augmentation variants: {len(self.split_indices)}")
    
    def __len__(self) -> int:
        """Return number of augmentation samples in this split."""
        if hasattr(self, 'split_indices'):
            return len(self.split_indices)
        return len(self.png_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load PNG image and return tensor with class label.
        
        With multiple augmentation variants per class:
        - idx ranges from 0 to len(self.split_indices) - 1
        - Each idx maps to (class_index, augmentation_seed)
        
        Args:
            idx (int): Index within split (0 to num_samples_in_split - 1)
        
        Returns:
            Tuple[torch.Tensor, int]: (image tensor of shape (3, H, W), class index)
        """
        # Map idx to class and augmentation seed
        if hasattr(self, 'split_indices'):
            # Multiple variants per class based on split assignment
            aug_seed = self.split_indices[idx]
            class_idx = idx % len(self.png_files)  # Cycle through all classes
        else:
            # If split='all', no splitting applied
            class_idx = idx
            aug_seed = idx
        
        img_path = self.png_files[class_idx]
        class_name = img_path.stem
        class_idx = self.class_to_idx[class_name]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Resize if needed
        if image.size != self.size:
            image = image.resize(self.size, Image.LANCZOS)
        
        # Convert to tensor and normalize
        image = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        
        # Convert from (H, W, C) to (C, H, W)
        image = image.permute(2, 0, 1)
        
        # Set augmentation seed for deterministic but varied augmentations
        if self.augmentation is not None:
            np.random.seed(aug_seed)
            image = self.augmentation(image, class_idx)
        
        # Apply additional transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, class_idx
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index."""
        return self.idx_to_class.get(class_idx, f"unknown_{class_idx}")
    
    def get_class_idx(self, class_name: str) -> int:
        """Get class index from name."""
        return self.class_to_idx.get(class_name, -1)


def create_dataloaders(root_dir: str, batch_size: int = 32, 
                       split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                       augmentation=None, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train, val, and test dataloaders.
    
    Args:
        root_dir (str): Path to minecraft/blocks/ directory
        batch_size (int): Batch size. Default: 32
        split_ratios (Tuple): Train/val/test split ratios. Default: (0.7, 0.15, 0.15)
        augmentation (Optional[callable]): Augmentation for training. Default: None
        num_workers (int): Number of workers for DataLoader. Default: 0
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]: 
            (train_loader, val_loader, test_loader, num_classes)
    """
    # Create datasets
    train_dataset = MinecraftTextureDataset(root_dir, split='train', 
                                            split_ratios=split_ratios, 
                                            augmentation=augmentation)
    val_dataset = MinecraftTextureDataset(root_dir, split='val', 
                                          split_ratios=split_ratios, 
                                          augmentation=None)
    test_dataset = MinecraftTextureDataset(root_dir, split='test', 
                                           split_ratios=split_ratios, 
                                           augmentation=None)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers)
    
    num_classes = len(train_dataset)
    
    return train_loader, val_loader, test_loader, num_classes


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    blocks_dir = Path(__file__).parent.parent.parent / "minecraft" / "blocks"
    
    if blocks_dir.exists():
        dataset = MinecraftTextureDataset(str(blocks_dir), split='all')
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of classes: {len(dataset.class_to_idx)}")
        
        image, class_idx = dataset[0]
        print(f"Sample image shape: {image.shape}")
        print(f"Sample class index: {class_idx}")
        print(f"Sample class name: {dataset.get_class_name(class_idx)}")
    else:
        print(f"Blocks directory not found at {blocks_dir}")
