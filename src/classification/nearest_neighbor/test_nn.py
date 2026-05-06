import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import json
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from minecraft.block_database_query import get_database, BlockDatabase
from .nearest_neighbor import MaskedNearestNeighbor
from src.classification.augmentation import AugmentationPipeline, EdgeBleeding, AdjacentNoise, IrregularHoles


class NNTester:
    def __init__(self, database: BlockDatabase, distance_metric: str = 'l2', black_threshold: float = 0.05):
        self.database = database
        self.nn = MaskedNearestNeighbor(database=database, distance_metric=distance_metric, black_threshold=black_threshold)
        self.augmentation_pipeline = AugmentationPipeline(database=database)

        self.nn.add_reference_images_from_database()

        self.label_to_name = self.nn.label_to_name
        self.name_to_label = self.nn.name_to_label

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
            pil_image = self.database.get_PNG_from_filename(texture_name)
            
            if pil_image is None:
                continue
            
            from src.util.image_transforms import pil_to_tensor
            image = pil_to_tensor(pil_image).numpy()

            predicted_label, predicted_name, distance = self.nn.predict(image)

            if predicted_label == label:
                correct += 1
            else:
                results['errors'].append({
                    'true_label': int(label),
                    'true_name': self.label_to_name[label],
                    'predicted_label': int(predicted_label),
                    'predicted_name': predicted_name,
                    'distance': float(distance)
                })

            total += 1

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(self.nn.reference_labels)} images")

        accuracy = correct / total if total > 0 else 0
        results['correct'] = correct
        results['total'] = total
        results['accuracy'] = accuracy

        print(f"Accuracy on original images: {accuracy:.4f} ({correct}/{total})")

        return results

    def test_with_augmentation(self, num_samples: int = 100, augmentation_type: str = 'all') -> Dict:
        print(f"\n=== Testing with Augmentation ({augmentation_type}) ===")

        indices = np.random.choice(len(self.nn.reference_labels), size=min(num_samples, len(self.nn.reference_labels)), 
                                  replace=False)

        correct = 0
        total = 0
        results = {
            'augmentation_type': augmentation_type,
            'correct': 0,
            'total': 0,
            'accuracy': 0.0,
            'errors': []
        }

        for idx in indices:
            label = self.nn.reference_labels[idx]
            class_name = self.label_to_name[label]

            pil_image = self.database.get_PNG_from_filename(class_name)
            if pil_image is None:
                continue
            
            from src.util.image_transforms import pil_to_tensor
            image = pil_to_tensor(pil_image, size=(16, 16))

            if augmentation_type == 'all':
                augmented_image = self.augmentation_pipeline(image, int(label))
            elif augmentation_type == 'edge_bleeding':
                aug = EdgeBleeding(image_database=self.database)
                augmented_image = aug(image, int(label))
            elif augmentation_type == 'adjacent_noise':
                aug = AdjacentNoise()
                augmented_image = aug(image)
            elif augmentation_type == 'irregular_holes':
                aug = IrregularHoles()
                augmented_image = aug(image)
            else:
                raise ValueError(f"Unknown augmentation type: {augmentation_type}")

            augmented_image = augmented_image.numpy()

            predicted_label, predicted_name, distance = self.nn.predict(augmented_image)

            if predicted_label == label:
                correct += 1
            else:
                results['errors'].append({
                    'true_label': int(label),
                    'true_name': class_name,
                    'predicted_label': int(predicted_label),
                    'predicted_name': predicted_name,
                    'distance': float(distance)
                })

            total += 1

        accuracy = correct / total if total > 0 else 0
        results['correct'] = correct
        results['total'] = total
        results['accuracy'] = accuracy

        print(f"Accuracy with {augmentation_type}: {accuracy:.4f} ({correct}/{total})")

        return results

    def test_all_metrics(self, test_type: str = 'original', num_samples: int = 100) -> Dict:
        results = {}

        for metric in ['l1', 'l2', 'cosine']:
            print(f"\n--- Testing with {metric.upper()} distance ---")
            self.nn = MaskedNearestNeighbor(database=self.database, distance_metric=metric, black_threshold=0.05)
            self.nn.add_reference_images_from_database()

            if test_type == 'original':
                result = self.test_original_images()
            else:
                result = self.test_with_augmentation(num_samples=num_samples, augmentation_type='all')

            results[metric] = result

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
    
    tester = NNTester(database)
    all_results = {}
    
    print("\n" + "=" * 60)
    print("TEST 1: Original Images")
    print("=" * 60)
    all_results['original_images'] = tester.test_all_metrics(test_type='original')
    
    print("\n" + "=" * 60)
    print("TEST 2: Augmented Images (All Metrics)")
    print("=" * 60)
    augmentation_types = ['all', 'edge_bleeding', 'adjacent_noise', 'irregular_holes']
    all_results['augmented_images'] = {}
    for metric in ['l1', 'l2', 'cosine']:
        print(f"\n--- Testing augmented images with {metric.upper()} distance ---")
        tester.nn = MaskedNearestNeighbor(database=database, distance_metric=metric)
        tester.nn.add_reference_images_from_database()

        metric_results = {}
        for aug_type in augmentation_types:
            result = tester.test_with_augmentation(num_samples=1000, augmentation_type=aug_type)
            metric_results[aug_type] = result

        all_results['augmented_images'][metric] = metric_results
    
    output_file = Path("output/nearest_neighbor_test_results.json")
    tester.save_results(all_results, str(output_file))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nOriginal Images Performance:")
    for metric, result in all_results['original_images'].items():
        print(f"  {metric.upper()}: {result['accuracy']:.4f}")
    
    print("\nAugmented Images Performance:")
    for metric, metric_results in all_results['augmented_images'].items():
        print(f"\n  {metric.upper()}: ")
        for aug_type, result in metric_results.items():
            print(f"    {aug_type}: {result['accuracy']:.4f}")


if __name__ == "__main__":
    database = get_database()
    run_comprehensive_test(database)
