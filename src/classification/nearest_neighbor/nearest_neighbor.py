import numpy as np
import torch
from typing import Tuple, Optional, Dict, List

from minecraft.block_database_query import BlockDatabase
from src.util.image_transforms import pil_to_tensor, ensure_numpy


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
        
        self.reference_vectors = None
        self.reference_masks = None
        self.reference_labels = None
        self.label_to_name = {}
        self.name_to_label = {}
    
    def _create_mask(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            c, h, w = image.shape
            intensity = image.mean(axis=0)
        else:
            c = 1
            h, w = image.shape
            intensity = image
        
        spatial_mask = (intensity > self.black_threshold)
        
        mask = np.repeat(spatial_mask[np.newaxis, :, :], c, axis=0)
        mask = mask.flatten()
        
        return mask
    
    def _vectorize_with_mask(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flat_image = image.reshape(-1)
        valid_vector = flat_image[mask]
        
        return valid_vector, mask
    
    def add_reference_images_from_database(self):
        textures = list(self.database.get_all_valid_textures())
        
        if not textures:
            raise ValueError("No valid textures found in database")
        
        print(f"Loading {len(textures)} reference images from database...")
        
        label_mapping = {texture: idx for idx, texture in enumerate(textures)}
        self.label_to_name = {v: k for k, v in label_mapping.items()}
        self.name_to_label = label_mapping
        
        reference_vectors = []
        reference_masks = []
        reference_labels = []
        
        for idx, texture_name in enumerate(textures):
            pil_image = self.database.get_PNG_from_filename(texture_name)
            if pil_image is None:
                continue
            
            try:
                image = pil_to_tensor(pil_image).numpy()
                
                mask = self._create_mask(image)
                vector, _ = self._vectorize_with_mask(image, mask)
                
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
        q_valid = query_vector
        r_valid = reference_vector
        
        if len(q_valid) != len(r_valid):
            min_len = min(len(q_valid), len(r_valid))
            q_valid = q_valid[:min_len]
            r_valid = r_valid[:min_len]
        
        if len(q_valid) == 0:
            return float('inf')
        
        if self.distance_metric == 'l2':
            distance = np.linalg.norm(q_valid - r_valid, ord=2)
        elif self.distance_metric == 'l1':
            distance = np.linalg.norm(q_valid - r_valid, ord=1)
        elif self.distance_metric == 'cosine':
            q_norm = np.linalg.norm(q_valid)
            r_norm = np.linalg.norm(r_valid)
            if q_norm == 0 or r_norm == 0:
                distance = float('inf')
            else:
                similarity = np.dot(q_valid, r_valid) / (q_norm * r_norm)
                distance = 1.0 - similarity
        
        return distance
    
    def predict(self, image: np.ndarray) -> Tuple[int, str, float]:
        if self.reference_vectors is None:
            raise ValueError("No reference images loaded. Call add_reference_images_from_database() first.")
        
        query_mask = self._create_mask(image)
        query_vector, _ = self._vectorize_with_mask(image, query_mask)
        
        distances = []
        for ref_vector, ref_mask in zip(self.reference_vectors, self.reference_masks):
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
            label, name, dist = self.predict(image)
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
