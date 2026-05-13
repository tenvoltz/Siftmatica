import numpy as np
import torch
from typing import Tuple, Dict, List

from minecraft.block_database import BlockDatabase
from src.util.image_transforms import pil_to_tensor


class MaskedNearestNeighbor:
    """
        Masked Nearest Neighbor classifier for texture classification.
        
        Masks out pixels near 0 (black/transparent) and computes distances using
        L1, L2, or cosine similarity on valid pixels only.
        
        Args:
            distance_metric (str): One of 'l1', 'l2', or 'cosine'. Default: 'l2'
            black_threshold (float): Pixel value threshold for masking (0.0 to 1.0). Default: 0.05
    """
    def __init__(self, database: BlockDatabase, distance_metric: str = 'l2', black_threshold: float = 0.05):
        if distance_metric not in ['l1', 'l2', 'cosine']:
            raise ValueError(f"distance_metric must be 'l1', 'l2', or 'cosine', got {distance_metric}")

        self.database = database
        self.distance_metric = distance_metric
        self.black_threshold = black_threshold
        self.image_size = (16, 16)

        self.reference_vectors = None
        self.reference_masks = None
        self.reference_labels = None
        self.label_to_name = {}
        self.name_to_label = {}
        self.spatial_weights = self._create_spatial_weights()

    def _create_spatial_weights(self) -> np.ndarray:
        h, w = self.image_size
        y, x = np.ogrid[:h, :w]

        cy = (h - 1) / 2
        cx = (w - 1) / 2

        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        dist /= dist.max()
        
        # Linear decay from center to edges
        # weights = 1.0 - dist
        # weights = 0.2 + 0.8 * weights
        # weights = np.repeat(weights[np.newaxis, :, :], 3, axis=0)
        
        # Gaussian decay from center to edges
        sigma = 0.5
        weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))
        weights = np.repeat(weights[np.newaxis, :, :], 3, axis=0)

        return weights.reshape(-1)

    def _create_mask(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3: raise ValueError(f"Expected image with shape (C, H, W), got {image.shape}")
        channels = image.shape[0]
        intensity = image.mean(axis=0)
        spatial_mask = intensity > self.black_threshold
        return np.repeat(spatial_mask[np.newaxis, :, :], channels, axis=0).reshape(-1)

    def _flatten_image(self, image: np.ndarray) -> np.ndarray:
        return image.reshape(-1)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if image.shape[1:] == self.image_size:
            return image

        tensor = torch.from_numpy(image).unsqueeze(0).float()
        resized = torch.nn.functional.interpolate(
            tensor,
            size=self.image_size,
            mode='bilinear',
            align_corners=False,
        )
        return resized.squeeze(0).cpu().numpy()

    def _vectorize_with_mask(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flat_image = self._flatten_image(image)
        valid_vector = flat_image[mask]
        return valid_vector, mask

    def add_reference_images_from_database(self):
        textures = list(self.database.get_all_valid_textures())
        if not textures: raise ValueError("No valid textures found in database")
        print(f"Loading {len(textures)} reference images from database...")

        label_mapping = {texture: idx for idx, texture in enumerate(textures)}
        self.label_to_name = {v: k for k, v in label_mapping.items()}
        self.name_to_label = label_mapping

        reference_vectors = []
        reference_masks = []
        reference_labels = []

        for idx, texture_name in enumerate(textures):
            pil_image = self.database.get_image(texture_name)
            if pil_image is None: continue

            try:
                image = pil_to_tensor(pil_image, size=self.image_size).cpu().numpy()
                image = self._resize_image(image)

                mask = self._create_mask(image)
                vector = self._flatten_image(image)

                reference_vectors.append(vector)
                reference_masks.append(mask)
                reference_labels.append(label_mapping[texture_name])
            except Exception as e:
                print(f"Warning: Failed to load {texture_name}: {e}")
                continue

        self.reference_vectors = reference_vectors
        self.reference_masks = reference_masks
        self.reference_labels = np.array(reference_labels)

        print(f"Loaded {len(reference_vectors)} reference images")

    def _compute_distance(self, query_vector: np.ndarray,
                         reference_vector: np.ndarray,
                         query_mask: np.ndarray,
                         reference_mask: np.ndarray) -> float:
        valid_mask = query_mask & reference_mask
        if not np.any(valid_mask): return float('inf')

        weights = self.spatial_weights[valid_mask]
        q_valid = query_vector[valid_mask]
        r_valid = reference_vector[valid_mask]

        if self.distance_metric == 'l2':
            diff = weights * (q_valid - r_valid)
            distance = np.linalg.norm(diff, ord=2)
        elif self.distance_metric == 'l1':
            diff = weights * (q_valid - r_valid)
            distance = np.linalg.norm(diff, ord=1)
        elif self.distance_metric == 'cosine':
            q_weighted = weights * q_valid
            r_weighted = weights * r_valid
            q_norm = np.linalg.norm(q_weighted)
            r_norm = np.linalg.norm(r_weighted)
            if q_norm == 0 or r_norm == 0: distance = float('inf')
            else:
                similarity = np.dot(q_weighted, r_weighted) / (q_norm * r_norm)
                distance = 1.0 - similarity
        return distance
    
    def _create_center_crop_mask(self, image: np.ndarray, crop_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        c, h, w = image.shape
        mask = np.zeros((h, w), dtype=bool)
        start_y = (h - crop_size[0]) // 2
        start_x = (w - crop_size[1]) // 2
        mask[start_y:start_y + crop_size[0], start_x:start_x + crop_size[1]] = True
        return np.repeat(mask[np.newaxis, :, :], c, axis=0).reshape(-1)

    def predict(self, image: np.ndarray) -> Tuple[int, str, float]:
        if self.reference_vectors is None:
            raise ValueError("No reference images loaded. Call add_reference_images_from_database() first.")

        image = self._resize_image(image)
        query_mask = self._create_mask(image)
        query_vector = self._flatten_image(image)
        # center_crop_mask = self._create_center_crop_mask(image)
        # query_mask = query_mask & center_crop_mask

        distances = []
        for ref_vector, ref_mask in zip(self.reference_vectors, self.reference_masks):
            # ref_mask = ref_mask & center_crop_mask
            distance = self._compute_distance(query_vector, ref_vector, query_mask, ref_mask)
            distances.append(distance)

        distances = np.array(distances)

        nearest_idx = np.argmin(distances)
        nearest_label = self.reference_labels[nearest_idx]
        nearest_distance = distances[nearest_idx]
        class_name = self.label_to_name[nearest_label]

        return nearest_label, class_name, nearest_distance

    def predict_batch(self, images: List[np.ndarray], return_distances: bool = False) -> Dict:
        predictions = []
        distances = []

        for image in images:
            label, _, dist = self.predict(image)
            predictions.append(label)
            distances.append(dist)

        result = {
            'labels': np.array(predictions),
            'names': [self.label_to_name[l] for l in predictions]
        }

        if return_distances:
            result['distances'] = np.array(distances)

        return result

    def get_config(self) -> Dict:
        return {
            'distance_metric': self.distance_metric,
            'black_threshold': self.black_threshold,
            'num_reference_images': len(self.reference_vectors) if self.reference_vectors else 0
        }


if __name__ == "__main__":
    print("MaskedNearestNeighbor module loaded successfully")
