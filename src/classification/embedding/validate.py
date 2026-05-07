import torch
import numpy as np
from typing import Dict, List
import json
from pathlib import Path

from .inference import EmbeddingMatcher
from .model import create_embedding_model
from src.classification.nearest_neighbor.nearest_neighbor import MaskedNearestNeighbor
from src.classification.augmentation import AugmentationPipeline, LEVELS
from minecraft.block_database import get_database


class EmbeddingValidator:
    
    def __init__(self, model_path: str = "checkpoints/embedding_model.pt"):
        self.database = get_database()
        self.nn_classifier = MaskedNearestNeighbor(self.database, distance_metric='cosine')
        self.nn_classifier.add_reference_images_from_database()
        
        self.embedding_model = create_embedding_model()
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            self.embedding_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        self.matcher = EmbeddingMatcher(self.embedding_model)
    
    def validate_original_images(self) -> Dict:
        print("\n=== Validating on Original Images ===")
        
        nn_correct = 0
        emb_correct = 0
        total = 0
        
        self.matcher.precompute_embeddings()
        
        for idx, texture_name in enumerate(self.database.get_all_valid_textures()):
            pil_image = self.database.get_image(texture_name)
            if pil_image is None:
                continue
            
            from src.util.image_transforms import pil_to_tensor
            image = pil_to_tensor(pil_image, size=(16, 16)).float().numpy()
            
            nn_label, nn_name, nn_dist = self.nn_classifier.predict(image)
            emb_name, emb_score, emb_idx = self.matcher.find_best_match(
                torch.from_numpy(image)
            )
            
            true_idx = self.nn_classifier.name_to_label.get(texture_name, -1)
            
            if nn_label == true_idx:
                nn_correct += 1
            if emb_name == texture_name:
                emb_correct += 1
            
            total += 1
        
        nn_acc = nn_correct / total if total > 0 else 0
        emb_acc = emb_correct / total if total > 0 else 0
        
        print(f"NearestNeighbor accuracy: {nn_acc:.4f} ({nn_correct}/{total})")
        print(f"Embedding accuracy: {emb_acc:.4f} ({emb_correct}/{total})")
        
        return {
            'nearest_neighbor': {'accuracy': nn_acc, 'correct': nn_correct, 'total': total},
            'embedding': {'accuracy': emb_acc, 'correct': emb_correct, 'total': total}
        }
    
    def validate_with_augmentation(self, num_samples: int = 100) -> Dict:
        print("\n=== Validating with Augmentation ===")
        
        results = {}
        augmentation = AugmentationPipeline()
        
        indices = np.random.choice(
            len(self.nn_classifier.reference_labels), 
            size=min(num_samples, len(self.nn_classifier.reference_labels)), 
            replace=False
        )
        
        for level in LEVELS:
            print(f"\n--- Testing level: {level} ---")
            augmentation.set_noise_levels(level, level, level)
            
            nn_correct = 0
            emb_correct = 0
            total = 0
            
            for idx in indices:
                label = self.nn_classifier.reference_labels[idx]
                texture_name = self.nn_classifier.label_to_name[label]
                pil_image = self.database.get_image(texture_name)
                
                if pil_image is None:
                    continue
                
                from src.util.image_transforms import pil_to_tensor
                image = pil_to_tensor(pil_image, size=(16, 16)).float()
                augmented = augmentation(image, image_filename=texture_name).numpy()
                
                nn_label, nn_name, nn_dist = self.nn_classifier.predict(augmented)
                
                emb_name, emb_score, emb_idx = self.matcher.find_best_match(
                    torch.from_numpy(augmented)
                )
                
                if nn_label == label:
                    nn_correct += 1
                if emb_name == texture_name:
                    emb_correct += 1
                
                total += 1
            
            nn_acc = nn_correct / total if total > 0 else 0
            emb_acc = emb_correct / total if total > 0 else 0
            
            results[level] = {
                'nearest_neighbor': {'accuracy': nn_acc, 'correct': nn_correct, 'total': total},
                'embedding': {'accuracy': emb_acc, 'correct': emb_correct, 'total': total}
            }
            
            print(f"  NN accuracy: {nn_acc:.4f}")
            print(f"  Embedding accuracy: {emb_acc:.4f}")
        
        return results
    
    def run_full_validation(self, num_samples: int = 100) -> Dict:
        print("=" * 60)
        print("Embedding Model Validation")
        print("=" * 60)
        
        all_results = {}
        all_results['original'] = self.validate_original_images()
        all_results['augmented'] = self.validate_with_augmentation(num_samples)
        
        output_path = Path("output/embedding_validation_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        return all_results


if __name__ == "__main__":
    validator = EmbeddingValidator()
    results = validator.run_full_validation(num_samples=50)
