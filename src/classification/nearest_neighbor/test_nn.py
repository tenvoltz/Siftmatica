import numpy as np
import torch
from pathlib import Path
from typing import Dict
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from minecraft.block_database import get_database, BlockDatabase
from src.util.image_transforms import pil_to_tensor, tensor_to_image
from .nearest_neighbor import MaskedNearestNeighbor
from src.classification.augmentation import (
    AugmentationPipeline, 
    EdgeBleeding, 
    AdjacentNoise, 
    IrregularHoles,
    LEVELS,
    configure_augmentation,
    apply_augmentation,
)


class NNTester:
    def __init__(self, database: BlockDatabase, distance_metric: str = 'l2', black_threshold: float = 0.05):
        self.database = database
        self.nn = MaskedNearestNeighbor(database=database, distance_metric=distance_metric, black_threshold=black_threshold)
        self.nn.add_reference_images_from_database()
        self.label_to_name = self.nn.label_to_name
        self.name_to_label = self.nn.name_to_label

    def _create_augmentation(self, augmentation_type: str, level: str):
        augmentation_map = {
            'all': AugmentationPipeline(database=self.database),
            'edge_bleeding': EdgeBleeding(image_database=self.database),
            'adjacent_noise': AdjacentNoise(),
            'irregular_holes': IrregularHoles(),
        }
        if augmentation_type not in augmentation_map:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
        aug = augmentation_map[augmentation_type]
        configure_augmentation(aug, level)
        return aug

    def test_original_images(self) -> Dict:
        print("\n=== Testing on Original Images ===")

        correct = 0
        total = 0
        results = {
            'correct': 0,
            'total': 0,
            'accuracy': 0.0,
            'errors': []
        }

        for idx, (ref_vector, label) in enumerate(zip(self.nn.reference_vectors, self.nn.reference_labels)):
            texture_name = self.label_to_name[label]
            pil_image = self.database.get_image(texture_name)
            
            if pil_image is None: continue
            image = pil_to_tensor(pil_image).numpy()
            predicted_label, predicted_name, distance = self.nn.predict(image)
            if predicted_label == label: correct += 1
            
            else:
                results['errors'].append({
                    'true_label': int(label),
                    'true_name': self.label_to_name[label],
                    'predicted_label': int(predicted_label),
                    'predicted_name': predicted_name,
                    'distance': float(distance)
                })

            total += 1
            if (idx + 1) % 100 == 0: print(f"Processed {idx + 1}/{len(self.nn.reference_labels)} images")

        accuracy = correct / total if total > 0 else 0
        results['correct'] = correct
        results['total'] = total
        results['accuracy'] = accuracy
        print(f"Accuracy on original images: {accuracy:.4f} ({correct}/{total})")
        return results

    def test_with_augmentation(self, num_samples: int = 100, augmentation_type: str = 'all', level: str = 'medium') -> Dict:
        print(f"\n=== Testing with Augmentation ({augmentation_type}, level={level}) ===")

        indices = np.random.choice(len(self.nn.reference_labels), size=min(num_samples, len(self.nn.reference_labels)), replace=False)
        aug = self._create_augmentation(augmentation_type, level)
        correct = 0
        total = 0
        errors = []
        
        i = 0

        for idx in indices:
            label = self.nn.reference_labels[idx]
            class_name = self.label_to_name[label]
            pil_image = self.database.get_image(class_name)
            
            if pil_image is None:
                continue
            
            image = pil_to_tensor(pil_image, size=(16, 16))
            augmented_image = apply_augmentation(aug, image, image_idx=int(label)).numpy()
            
            if i < 5:
                # Plot the first few augmented images for visual inspection
                import matplotlib.pyplot as plt
                plt.figure(figsize=(4, 4))
                plt.imshow(tensor_to_image(augmented_image))
                plt.title(f"Augmented Image {i+1}")
            
                plt.axis('off')
                plt.show()
                i += 1
                
            predicted_label, predicted_name, distance = self.nn.predict(augmented_image)

            if predicted_label == label:
                correct += 1
            else:
                errors.append({
                    'true_label': int(label),
                    'true_name': class_name,
                    'predicted_label': int(predicted_label),
                    'predicted_name': predicted_name,
                    'distance': float(distance)
                })
            total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy with {augmentation_type} (level={level}): {accuracy:.4f} ({correct}/{total})")

        return {
            'augmentation_type': augmentation_type,
            'level': level,
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'errors': errors
        }

    def test_with_augmentation_all_levels(self, num_samples: int = 100, augmentation_type: str = 'all') -> Dict:
        return {level: self.test_with_augmentation(num_samples=num_samples, augmentation_type=augmentation_type, level=level) for level in LEVELS}

    def test_metrics(self, metrics: list = ['l1', 'l2', 'cosine']) -> Dict:
        results = {}
        for metric in metrics:
            print(f"\n--- Testing with {metric.upper()} distance ---")
            self.nn = MaskedNearestNeighbor(database=self.database, distance_metric=metric, black_threshold=0.05)
            self.nn.add_reference_images_from_database()
            results[metric] = self.test_original_images()
        return results

    def save_results(self, results: Dict, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=convert_to_serializable)

        print(f"Results saved to {output_path}")


def run_comprehensive_test(database: BlockDatabase):
    print("=" * 60)
    print("Masked Nearest Neighbor - Comprehensive Test Suite")
    print("=" * 60)
    metrics = ['cosine']
    augmentation_types = ['all', 'edge_bleeding'] # , 'adjacent_noise', 'irregular_holes']
    tester = NNTester(database)
    all_results = {}
    
    print("\n" + "=" * 60)
    print("TEST 1: Original Images")
    print("=" * 60)
    all_results['original_images'] = tester.test_metrics(metrics=metrics)
    print("\n" + "=" * 60)
    
    
    print("TEST 2: Augmented Images with Level Variations")
    print("=" * 60)
    all_results['augmented_images_by_level'] = {}
    
    for metric in metrics:
        print(f"\n--- Testing augmented images with {metric.upper()} distance ---")
        tester.nn = MaskedNearestNeighbor(database=database, distance_metric=metric)
        tester.nn.add_reference_images_from_database()
        all_results['augmented_images_by_level'][metric] = {
            aug_type: tester.test_with_augmentation_all_levels(num_samples=500, augmentation_type=aug_type)
            for aug_type in augmentation_types
        }
    
    output_file = Path("output/nearest_neighbor_test_results.json")
    tester.save_results(all_results, str(output_file))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nOriginal Images Performance:")
    for metric, result in all_results['original_images'].items():
        print(f"  {metric.upper()}: {result['accuracy']:.4f}")
    
    print("\nAugmented Images Performance by Level:")
    for metric, metric_results in all_results['augmented_images_by_level'].items():
        print(f"\n  {metric.upper()}: ")
        for aug_type, level_results in metric_results.items():
            print(f"    {aug_type}:")
            for level, result in level_results.items():
                print(f"      {level}: {result['accuracy']:.4f}")


if __name__ == "__main__":
    database = get_database()
    run_comprehensive_test(database)
