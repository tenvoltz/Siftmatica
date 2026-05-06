import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from minecraft.block_database_query import BlockDatabase, get_database
from src.util.image_transforms import pil_to_tensor, ensure_tensor, ensure_3channel


class EdgeBleeding:
    """
    Simulates texture bleeding when adjacent blocks are visible.
    
    Samples 8 random block textures and places them around the target image.
    A random shift vector is applied to all 9 images together, simulating
    the visual effect of adjacent block textures "bleeding" into the edges
    of the target block.
    
    Args:
        max_shift (int): Maximum pixel shift in any direction. Default: 4
        bias_factor (float): Bias factor toward target texture for sampling. Default: 3.0
        num_textures (int): Total number of available textures for sampling. Default: 989
    """
    
    def __init__(self,image_database: BlockDatabase, max_shift: int = 4, bias_factor: float = 3.0, num_textures: int = 989):
        self.max_shift = max_shift
        self.bias_factor = bias_factor
        self.num_textures = len(image_database.get_all_valid_textures())
        self.image_database = image_database
    
    def __call__(self, image: torch.Tensor, target_idx: int) -> torch.Tensor:
        image = ensure_tensor(image)
        image = ensure_3channel(image)
        
        c, h, w = image.shape
        assert h == 16 and w == 16, f"Expected 16x16 image, got {h}x{w}"
        
        canvas = torch.zeros(c, 48, 48, dtype=image.dtype)
        canvas[:, 16:32, 16:32] = image
        
        surrounding_classes = self._sample_biased_classes(target_idx, num_samples=8)
        surrounding_textures = []
        for class_idx in surrounding_classes:
            png_filename = self.image_database.get_valid_texture_by_index(class_idx)
            if png_filename is not None:
                pil_image = self.image_database.get_PNG_from_filename(png_filename)
                if pil_image is not None:
                    tex_tensor = pil_to_tensor(pil_image, size=(w, h))
                    surrounding_textures.append(tex_tensor)
                else:
                    surrounding_textures.append(torch.randn(c, h, w) * 0.2 + 0.5)
            else:
                surrounding_textures.append(torch.randn(c, h, w) * 0.2 + 0.5)
        
        positions = [(0, 0), (0, 1), (0, 2),
                     (1, 0),         (1, 2),
                     (2, 0), (2, 1), (2, 2)]
        
        for idx, (grid_y, grid_x) in enumerate(positions):
            y_start = grid_y * h
            y_end = y_start + h
            x_start = grid_x * w
            x_end = x_start + w
            
            surrounding_texture = ensure_tensor(surrounding_textures[idx])
            surrounding_texture = torch.clamp(surrounding_texture, 0, 1)
            canvas[:, y_start:y_end, x_start:x_end] = surrounding_texture
        
        shift_y = np.random.randint(-self.max_shift, self.max_shift + 1)
        shift_x = np.random.randint(-self.max_shift, self.max_shift + 1)
        
        if shift_y != 0:
            canvas = torch.roll(canvas, shifts=shift_y, dims=1)
        if shift_x != 0:
            canvas = torch.roll(canvas, shifts=shift_x, dims=2)
        
        center_y = 16
        center_x = 16
        result = canvas[:, center_y:center_y+h, center_x:center_x+w]
        
        return result
    
    def _sample_biased_classes(self, target_idx: int, num_samples: int = 8) -> np.ndarray:
        # Clamp target_idx to valid range
        target_idx = min(target_idx, self.num_textures - 1)
        
        weights = np.ones(self.num_textures)
        weights[target_idx] *= self.bias_factor
        weights /= weights.sum()
        
        return np.random.choice(self.num_textures, size=num_samples, p=weights)


class AdjacentNoise:
    """
    Adds noise by randomly copying pixels from adjacent pixels in the image.
    
    Simulates compression artifacts and color bleeding by randomly copying
    pixels from adjacent pixels (neighbors) in the image to other locations.
    
    Args:
        probability (float): Probability of applying augmentation. Default: 0.5
        intensity (float): Strength of noise injection (0.0 to 1.0). Default: 0.3
    """
    
    def __init__(self, probability: float = 0.5, intensity: float = 0.3):
        self.probability = probability
        self.intensity = intensity
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = ensure_tensor(image)
        
        if np.random.random() > self.probability:
            return image
        
        image = ensure_3channel(image)
        image = image.clone()
        c, h, w = image.shape
        
        num_operations = np.random.randint(5, 15)
        
        for _ in range(num_operations):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            dy, dx = directions[np.random.randint(0, len(directions))]
            
            ny = np.clip(y + dy, 0, h - 1)
            nx = np.clip(x + dx, 0, w - 1)
            
            image[:, y, x] = self.intensity * image[:, ny, nx] + (1 - self.intensity) * image[:, y, x]
        
        return image


class IrregularHoles:
    """
    Creates irregular holes by randomly removing pixels in irregular patches.
    
    Simulates occlusion or damage to textures by randomly removing pixels
    in irregular patches and setting them to a fill value (black).
    
    Args:
        probability (float): Probability of applying augmentation. Default: 0.4
        patch_size_range (Tuple[int, int]): Range of patch sizes in pixels. Default: (1, 6)
        num_patches (Optional[int]): Number of patches to remove. If None, random 1-3. Default: None
        fill_value (float): Value to fill holes with. Default: 0.0 (black)
    """
    
    def __init__(self, probability: float = 0.4, patch_size_range: Tuple[int, int] = (1, 6),
                 num_patches: Optional[int] = None, fill_value: float = 0.0):
        self.probability = probability
        self.patch_size_range = patch_size_range
        self.num_patches = num_patches
        self.fill_value = fill_value
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = ensure_tensor(image)
        
        if np.random.random() > self.probability:
            return image
        
        image = ensure_3channel(image)
        image = image.clone()
        c, h, w = image.shape
        
        num_patches = self.num_patches if self.num_patches is not None else np.random.randint(1, 4)
        
        for _ in range(num_patches):
            patch_h = np.random.randint(self.patch_size_range[0], self.patch_size_range[1] + 1)
            patch_w = np.random.randint(self.patch_size_range[0], self.patch_size_range[1] + 1)
            
            y_start = np.random.randint(0, max(1, h - patch_h))
            x_start = np.random.randint(0, max(1, w - patch_w))
            
            y_end = min(y_start + patch_h, h)
            x_end = min(x_start + patch_w, w)
            
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if np.random.random() > 0.3:
                        image[:, y, x] = self.fill_value
        
        return image


class AugmentationPipeline:
    def __init__(self, database: BlockDatabase = None, edge_bleeding_prob: float = 0.5, adjacent_noise_prob: float = 0.5,
                 irregular_holes_prob: float = 0.5):
        if database is None:
            database = get_database()
        self.edge_bleeding = EdgeBleeding(database)
        self.adjacent_noise = AdjacentNoise(probability=adjacent_noise_prob)
        self.irregular_holes = IrregularHoles(probability=irregular_holes_prob)
        self.edge_bleeding_prob = edge_bleeding_prob

    def __call__(self, image: torch.Tensor, class_idx: int = 0) -> torch.Tensor:
        if np.random.random() < self.edge_bleeding_prob:
            image = self.edge_bleeding(image, class_idx)

        image = self.adjacent_noise(image)
        image = self.irregular_holes(image)

        return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    database = get_database()
    
    # Load 10 real texture images from database
    num_samples = 10
    all_textures = list(database.get_all_valid_textures())[:num_samples]
    test_images = []
    
    for texture_name in all_textures:
        pil_image = database.get_PNG_from_filename(texture_name)
        if pil_image is not None:
            test_images.append(pil_to_tensor(pil_image, size=(16, 16)))
    
    if len(test_images) < num_samples:
        print(f"Warning: Only found {len(test_images)} valid textures, expected {num_samples}")
    
    # Initialize augmentation methods
    eb = EdgeBleeding(database)
    an = AdjacentNoise(probability=1.0)  # Always apply for visualization
    ih = IrregularHoles(probability=1.0)  # Always apply for visualization
    pipeline = AugmentationPipeline(database, edge_bleeding_prob=1.0, adjacent_noise_prob=1.0, irregular_holes_prob=1.0)
    
    def tensor_to_image(t):
        """Convert torch tensor to numpy image for display"""
        if isinstance(t, torch.Tensor):
            t = t.numpy()
        t = np.transpose(t, (1, 2, 0))  # CHW -> HWC
        return np.clip(t, 0, 1)
    
    # Plot EdgeBleeding
    fig, axes = plt.subplots(2, len(test_images), figsize=(20, 4))
    fig.suptitle('EdgeBleeding Augmentation', fontsize=16)
    for i, test_img in enumerate(test_images):
        axes[0, i].imshow(tensor_to_image(test_img))
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        augmented = eb(test_img, target_idx=i % 989)
        axes[1, i].imshow(tensor_to_image(augmented))
        axes[1, i].set_title("EdgeBleeding")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig('output/augmentation_edge_bleeding.png', dpi=100, bbox_inches='tight')
    print("Saved: output/augmentation_edge_bleeding.png")
    plt.close()
    
    # Plot AdjacentNoise
    fig, axes = plt.subplots(2, len(test_images), figsize=(20, 4))
    fig.suptitle('AdjacentNoise Augmentation', fontsize=16)
    for i, test_img in enumerate(test_images):
        axes[0, i].imshow(tensor_to_image(test_img))
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        augmented = an(test_img)
        axes[1, i].imshow(tensor_to_image(augmented))
        axes[1, i].set_title("AdjacentNoise")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig('output/augmentation_adjacent_noise.png', dpi=100, bbox_inches='tight')
    print("Saved: output/augmentation_adjacent_noise.png")
    plt.close()
    
    # Plot IrregularHoles
    fig, axes = plt.subplots(2, len(test_images), figsize=(20, 4))
    fig.suptitle('IrregularHoles Augmentation', fontsize=16)
    for i, test_img in enumerate(test_images):
        axes[0, i].imshow(tensor_to_image(test_img))
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        augmented = ih(test_img)
        axes[1, i].imshow(tensor_to_image(augmented))
        axes[1, i].set_title("IrregularHoles")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig('output/augmentation_irregular_holes.png', dpi=100, bbox_inches='tight')
    print("Saved: output/augmentation_irregular_holes.png")
    plt.close()
    
    # Plot AugmentationPipeline
    fig, axes = plt.subplots(2, len(test_images), figsize=(20, 4))
    fig.suptitle('AugmentationPipeline (All Augmentations)', fontsize=16)
    for i, test_img in enumerate(test_images):
        axes[0, i].imshow(tensor_to_image(test_img))
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        augmented = pipeline(test_img, class_idx=i % 989)
        axes[1, i].imshow(tensor_to_image(augmented))
        axes[1, i].set_title("Pipeline")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig('output/augmentation_pipeline.png', dpi=100, bbox_inches='tight')
    print("Saved: output/augmentation_pipeline.png")
    plt.close()
    
    print("\nAll augmentation visualizations complete!")
