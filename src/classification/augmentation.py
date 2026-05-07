import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from minecraft.block_database import BlockDatabase, get_database
from src.util.image_transforms import (
    pil_to_tensor,
    ensure_tensor,
    ensure_3channel,
    tensor_to_image,
)

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

    GRID_POSITIONS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2))
    LOW_BLEEDING = (3, 2.0)
    MEDIUM_BLEEDING = (5, 2.0)
    HIGH_BLEEDING = (8, 2.0)

    def __init__(
        self,
        image_database: BlockDatabase = None,
        max_shift: int = 3,
        bias_factor: float = 2.0,
    ):
        self.max_shift = max_shift
        self.bias_factor = bias_factor
        self.image_database = image_database or get_database()
        self.texture_cache = self._build_texture_cache()
        self.num_textures = len(self.texture_cache)

    def _build_texture_cache(self) -> list[torch.Tensor]:
        textures = []
        for texture_name in self.image_database.get_all_valid_textures():
            image = self.image_database.get_image(texture_name)
            if image is None:
                continue
            tensor = ensure_3channel(pil_to_tensor(image, size=(16, 16))).float()
            textures.append(tensor)
        if not textures:
            raise RuntimeError("No valid textures loaded.")
        return textures

    def _sample_indices(
        self, target_idx: int, count: int, device: torch.device
    ) -> torch.Tensor:
        weights = torch.ones(self.num_textures, dtype=torch.float32, device=device)
        target_idx = min(target_idx, self.num_textures - 1)
        weights[target_idx] *= self.bias_factor * (self.num_textures - 1)
        return torch.multinomial(weights, count, replacement=True)

    def __call__(self, image: torch.Tensor, image_filename: str) -> torch.Tensor:
        image = ensure_3channel(ensure_tensor(image)).float()        
        c, h, w = image.shape
        device = image.device

        canvas = torch.zeros((c, h * 3, w * 3), dtype=image.dtype, device=device)
        canvas[:, h : 2 * h, w : 2 * w] = image

        target_idx = self.image_database.get_texture_index(image_filename)
        sampled = self._sample_indices(target_idx=target_idx, count=8, device=device)
        for idx, (gy, gx) in enumerate(self.GRID_POSITIONS):
            tex = self.texture_cache[sampled[idx].item()].to(
                device=device, dtype=image.dtype
            )
            y0, x0 = gy * h, gx * w
            canvas[:, y0 : y0 + h, x0 : x0 + w] = tex

        shift_y = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        shift_x = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        canvas = torch.roll(canvas, shifts=(shift_y, shift_x), dims=(1, 2))

        return canvas[:, h : 2 * h, w : 2 * w].clamp(0, 1)

    def set_noise_level(self, level: str):
        if level == 'low':
            self.max_shift, self.bias_factor = self.LOW_BLEEDING
        elif level == 'medium':
            self.max_shift, self.bias_factor = self.MEDIUM_BLEEDING
        elif level == 'high':
            self.max_shift, self.bias_factor = self.HIGH_BLEEDING
        else:
            raise ValueError(f"Invalid bleeding level: {level}. Choose from 'low', 'medium', 'high'.")


class AdjacentNoise:
    """
    Adds noise by randomly copying pixels from adjacent pixels in the image.

    Simulates compression artifacts and color bleeding by randomly copying
    pixels from adjacent pixels (neighbors) in the image to other locations.

    Args:
        intensity (float): Strength of noise injection (0.0 to 1.0). Default: 0.3
        operations_range (Tuple[int, int]): Range of number of noise operations to apply. Default: (100, 150)
    """
    NEIGHBORS = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    LOW_NOISE = (0.2, (50, 80))
    MEDIUM_NOISE = (0.5, (100, 150))
    HIGH_NOISE = (1.0, (150, 200))

    def __init__(self, intensity: float = 0.3, operations_range: tuple[int, int] = (100, 150)):
        self.intensity = intensity
        self.operations_range = operations_range
        
    def set_noise_level(self, level: str):
        if level == 'low':
            self.intensity, self.operations_range = self.LOW_NOISE
        elif level == 'medium':
            self.intensity, self.operations_range = self.MEDIUM_NOISE
        elif level == 'high':
            self.intensity, self.operations_range = self.HIGH_NOISE
        else:
            raise ValueError(f"Invalid noise level: {level}. Choose from 'low', 'medium', 'high'.")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = ensure_3channel(ensure_tensor(image)).clone()
        _, h, w = image.shape
        ops = torch.randint(self.operations_range[0], self.operations_range[1] + 1, (1,)).item()
        for _ in range(ops):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            direction = self.NEIGHBORS[torch.randint(0, 8, (1,))][0]
            ny = int(torch.clamp((y + direction[0]).detach().clone(), 0, h - 1))
            nx = int(torch.clamp((x + direction[1]).detach().clone(), 0, w - 1))
            image[:, y, x] = self.intensity * image[:, ny, nx] + (1 - self.intensity) * image[:, y, x]
        return image.clamp(0, 1)


class IrregularHoles:
    """
    Creates irregular holes by randomly removing pixels in irregular patches.

    Simulates occlusion or damage to textures by randomly removing pixels
    in irregular patches and setting them to a fill value (black).

    Args:
        patch_size_range (Tuple[int, int]): Range of patch sizes in pixels. Default: (1, 6)
        operation_ranges (Tuple[int, int]): Range of number of operations to apply. Default: (5, 8)
        fill_value (float): Value to fill holes with. Default: 0.0 (black)
        density (float): Density of holes within each patch (0.0 to 1.0). Default: 0.5
    """

    LOW_HOLES = ((1, 3), (5, 8), 0.5)
    MEDIUM_HOLES = ((2, 5), (8, 12), 0.7)
    HIGH_HOLES = ((3, 8), (12, 20), 0.9)

    def __init__(self, patch_size_range: tuple[int, int] = (1, 6), operation_ranges: tuple[int, int] = (5, 8), fill_value: float = 0.0, density: float = 0.5):
        self.patch_size_range = patch_size_range
        self.operations_range = operation_ranges
        self.fill_value = fill_value
        self.density = density

    def set_noise_level(self, level: str):
        if level == 'low':
            self.patch_size_range, self.operations_range, self.density = self.LOW_HOLES
        elif level == 'medium':
            self.patch_size_range, self.operations_range, self.density = self.MEDIUM_HOLES
        elif level == 'high':
            self.patch_size_range, self.operations_range, self.density = self.HIGH_HOLES
        else:
            raise ValueError(f"Invalid hole level: {level}. Choose from 'low', 'medium', 'high'.")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = ensure_3channel(ensure_tensor(image)).clone()
        _, h, w = image.shape
        ops = torch.randint(self.operations_range[0], self.operations_range[1] + 1, (1,)).item()
        for _ in range(ops):
            ph = torch.randint(self.patch_size_range[0], self.patch_size_range[1] + 1, (1,)).item()
            pw = torch.randint(self.patch_size_range[0], self.patch_size_range[1] + 1, (1,)).item()
            y0 = torch.randint(0, max(1, h - ph + 1), (1,)).item()
            x0 = torch.randint(0, max(1, w - pw + 1), (1,)).item()
            mask = torch.rand((ph, pw)) < self.density
            image[:, y0 : y0 + ph, x0 : x0 + pw][:, mask] = self.fill_value
        return image


class AugmentationPipeline:
    def __init__(
        self,
        database: BlockDatabase = None,
        edge_bleeding_prob: float = 0.5,
        adjacent_noise_prob: float = 1.0,
        irregular_holes_prob: float = 0.5,
    ):
        if database is None: database = get_database()
        self.edge_bleeding = EdgeBleeding(database)
        self.adjacent_noise = AdjacentNoise()
        self.irregular_holes = IrregularHoles()
        self.edge_bleeding_prob = edge_bleeding_prob
        self.adjacent_noise_prob = adjacent_noise_prob
        self.irregular_holes_prob = irregular_holes_prob

    def __call__(self, image: torch.Tensor, image_filename: str = "") -> torch.Tensor:
        if torch.rand(1) < self.edge_bleeding_prob:
            image = self.edge_bleeding(image, image_filename)
        if torch.rand(1) < self.adjacent_noise_prob:
            image = self.adjacent_noise(image)
        if torch.rand(1) < self.irregular_holes_prob:
            image = self.irregular_holes(image)
        return image

    def set_noise_levels(self, bleeding_level: str, noise_level: str, hole_level: str):
        self.edge_bleeding.set_noise_level(bleeding_level)
        self.adjacent_noise.set_noise_level(noise_level)
        self.irregular_holes.set_noise_level(hole_level)


LEVELS = ["low", "medium", "high"]
NUM_SAMPLES = 10
OUTPUT_DIR = Path("output/augmentation")

def load_test_images(database, num_samples=NUM_SAMPLES, size=(16, 16)):
    textures = list(database.get_all_valid_textures())[:num_samples]
    return [
        {
            "texture_name": texture_name,
            "image": pil_to_tensor(database.get_image(texture_name), size=size).float()
        }
        for texture_name in textures
    ]


def configure_augmentation(augmentation, level):
    if isinstance(augmentation, EdgeBleeding):
        augmentation.set_noise_level(level)
    if isinstance(augmentation, AdjacentNoise):
        augmentation.set_noise_level(level)
    if isinstance(augmentation, IrregularHoles):
        augmentation.set_noise_level(level)
    if isinstance(augmentation, AugmentationPipeline):
        augmentation.set_noise_levels(
            bleeding_level=level,
            noise_level=level,
            hole_level=level,
        )


def apply_augmentation(augmentation, image, image_filename):
    """
    Handle differing augmentation call signatures.
    """
    if isinstance(augmentation, EdgeBleeding):
        return augmentation(image, image_filename)

    if isinstance(augmentation, AugmentationPipeline):
        return augmentation(image, image_filename)

    return augmentation(image)


def visualize_augmentation(augmentation, test_images, title, output_filename):
    rows = 1 + len(LEVELS)
    cols = len(test_images)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    fig.suptitle(title, fontsize=16)

    row_labels = [
        "Original",
        *[f"{title} ({level.capitalize()})" for level in LEVELS],
    ]

    for col, test_image in enumerate(test_images):
        axes[0, col].imshow(tensor_to_image(test_image["image"]))
        axes[0, col].axis("off")
        if col == 0: axes[0, col].set_title(row_labels[0],fontsize=14,pad=20, loc="left")
        
        for row, level in enumerate(LEVELS, start=1):
            configure_augmentation(augmentation, level)
            augmented = apply_augmentation(augmentation, test_image["image"], test_image["texture_name"])
            axes[row, col].imshow(tensor_to_image(augmented))
            axes[row, col].axis("off")
            if col == 0: axes[row, col].set_title(row_labels[row],fontsize=14,pad=20, loc="left")

    plt.tight_layout()
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    database = get_database()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_images = load_test_images(database)
    augmentations = [
        (
            EdgeBleeding(),
            "EdgeBleeding",
            "augmentation_edge_bleeding.png",
        ),
        (
            AdjacentNoise(),
            "AdjacentNoise",
            "augmentation_adjacent_noise.png",
        ),
        (
            IrregularHoles(),
            "IrregularHoles",
            "augmentation_irregular_holes.png",
        ),
        (
            AugmentationPipeline(
                edge_bleeding_prob=1.0,
                adjacent_noise_prob=1.0,
                irregular_holes_prob=1.0,
            ),
            "Pipeline",
            "augmentation_pipeline.png",
        ),
    ]
    for augmentation, title, filename in augmentations:
        visualize_augmentation(
            augmentation=augmentation,
            test_images=test_images,
            title=title,
            output_filename=filename,
        )
        
    print("\nAll augmentation visualizations complete!")
