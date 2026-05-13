import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
import numpy as np

from minecraft.block_database import BlockDatabase, get_database
from src.classification.augmentation import AugmentationPipeline, LEVELS
from src.util.image_transforms import pil_to_tensor


class TripletTextureDataset(Dataset):

    def __init__(self, database: Optional[BlockDatabase] = None, semi_hard_negative: bool = False):
        self.database = database or get_database()
        self.textures = self.database.get_all_valid_textures()
        self.num_textures = len(self.textures)
        self.semi_hard_negative = semi_hard_negative

        self.augmentation = AugmentationPipeline()

        self.precomputed_embeddings = None
        self.margin = 0.3

    def __len__(self) -> int:
        return self.num_textures

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_idx = idx
        texture_name = self.textures[anchor_idx]

        pil_image = self.database.get_image(texture_name)
        if pil_image is None:
            return self.__getitem__((idx + 1) % len(self))

        anchor = pil_to_tensor(pil_image, size=(16, 16)).float()
        positive = self._get_augmented_positive(anchor, image_filename=texture_name)
        negative = self._get_negative(anchor_idx, anchor)

        return anchor, positive, negative

    def _get_augmented_positive(self, anchor: torch.Tensor, image_filename: str, level_distribution: List[float] = [0.2, 0.4, 0.4]) -> torch.Tensor:
        level = LEVELS[torch.multinomial(torch.tensor(level_distribution), 1).item()]
        self.augmentation.set_noise_levels(level, level, level)
        return self.augmentation(anchor, image_filename=image_filename)

    def set_precomputed_embeddings(self, embeddings: torch.Tensor, margin: float = 0.3):
        self.precomputed_embeddings = embeddings  # Shape: (num_textures, embedding_dim)
        self.margin = margin

    def _get_negative(self, anchor_idx: int, anchor: torch.Tensor) -> torch.Tensor:
        if self.semi_hard_negative and self.precomputed_embeddings is not None:
            return self._get_semi_hard_negative(anchor_idx, anchor, self.textures[anchor_idx])
        else:
            return self._get_random_negative(anchor_idx)

    def _get_semi_hard_negative(self, anchor_idx: int, anchor: torch.Tensor, texture_name: str) -> torch.Tensor:
        anchor_emb = self.precomputed_embeddings[anchor_idx]
        anchor_emb = anchor_emb / anchor_emb.norm(p=2)
        
        pos_aug = self._get_augmented_positive(anchor, image_filename=texture_name)
        pos_dist = 1 - torch.sum(anchor_emb * anchor_emb)

        all_indices = torch.arange(self.num_textures)
        neg_indices = all_indices[all_indices != anchor_idx]
        neg_embeddings = self.precomputed_embeddings[neg_indices]

        similarities = torch.sum(anchor_emb.unsqueeze(0) * neg_embeddings, dim=1)
        neg_dists = 1 - similarities

        lower_bound = pos_dist
        upper_bound = pos_dist + self.margin
        semi_hard_mask = (neg_dists > lower_bound) & (neg_dists < upper_bound)

        if semi_hard_mask.any():
            semi_hard_indices = neg_indices[semi_hard_mask]
            rand_idx = torch.randint(0, len(semi_hard_indices), (1,)).item()
            neg_idx = semi_hard_indices[rand_idx].item()
        else:
            rand_idx = torch.randint(0, len(neg_indices), (1,)).item()
            neg_idx = neg_indices[rand_idx].item()

        neg_texture_name = self.textures[neg_idx]
        neg_pil = self.database.get_image(neg_texture_name)
        if neg_pil is None:
            return self._get_negative(anchor_idx, anchor)

        return pil_to_tensor(neg_pil, size=(16, 16)).float()

    def _get_random_negative(self, anchor_idx: int) -> torch.Tensor:
        neg_idx = torch.randint(0, self.num_textures, (1,)).item()
        while neg_idx == anchor_idx:
            neg_idx = torch.randint(0, self.num_textures, (1,)).item()

        neg_texture_name = self.textures[neg_idx]
        neg_pil = self.database.get_image(neg_texture_name)
        if neg_pil is None:
            return self._get_random_negative(anchor_idx)

        return pil_to_tensor(neg_pil, size=(16, 16)).float()


class AugmentedValidationDataset(Dataset):
    
    def __init__(self, level: str = 'medium', database: Optional[BlockDatabase] = None):
        if level not in LEVELS:
            raise ValueError(f"Invalid level: {level}. Choose from {LEVELS}")
        
        self.database = database or get_database()
        self.textures = self.database.get_all_valid_textures()
        self.level = level
        
        self.augmentation = AugmentationPipeline()
        self.augmentation.set_noise_levels(level, level, level)
    
    def __len__(self) -> int:
        return len(self.textures)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        texture_name = self.textures[idx]
        pil_image = self.database.get_image(texture_name)
        
        if pil_image is None:
            return self.__getitem__((idx + 1) % len(self))
        
        clean = pil_to_tensor(pil_image, size=(16, 16)).float()
        augmented = self.augmentation(clean, image_filename=texture_name)
        
        return augmented, texture_name


def create_validation_datasets() -> Dict[str, AugmentedValidationDataset]:
    return {level: AugmentedValidationDataset(level=level) for level in LEVELS}


if __name__ == "__main__":
    train_dataset = TripletTextureDataset()
    print(f"Train dataset size: {len(train_dataset)}")
    
    val_datasets = create_validation_datasets()
    for level, dataset in val_datasets.items():
        print(f"Validation dataset ({level}): {len(dataset)} samples")
    
    anchor, positive, negative = train_dataset[0]
    print(f"Anchor shape: {anchor.shape}")
    print(f"Positive shape: {positive.shape}")
    print(f"Negative shape: {negative.shape}")
