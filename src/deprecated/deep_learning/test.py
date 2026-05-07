"""
Testing and evaluation for ResNet-18-lite classifier.

Provides inference, accuracy metrics, per-class performance analysis,
and confusion matrix visualization.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from .model import create_model
from .dataset import MinecraftTextureDataset


class Evaluator:
    """Evaluator class for ResNet-18-lite model."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, loader: DataLoader) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            loader (DataLoader): Data loader for evaluation
        
        Returns:
            Dict: Evaluation metrics including accuracy, precision, recall, F1
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get predictions
                _, predicted = outputs.max(1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Classification report
        report = classification_report(all_labels, all_predictions, zero_division=0, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        metrics = {
            'accuracy': accuracy,
            'top_5_accuracy': self._top_k_accuracy(all_probabilities, all_labels, k=5),
            'top_10_accuracy': self._top_k_accuracy(all_probabilities, all_labels, k=10),
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'per_class_metrics': report
        }
        
        return metrics
    
    def _top_k_accuracy(self, probabilities: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
        """Calculate top-k accuracy."""
        if k > probabilities.shape[1]:
            k = probabilities.shape[1]
        
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0
        for i, label in enumerate(labels):
            if label in top_k_preds[i]:
                correct += 1
        
        return correct / len(labels)
    
    def infer_single_image(self, image_path: str, dataset: MinecraftTextureDataset,
                          return_top_k: int = 5) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_path (str): Path to image file
            dataset (MinecraftTextureDataset): Dataset for class mapping
            return_top_k (int): Return top K predictions. Default: 5
        
        Returns:
            Dict: Prediction results
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((16, 16), Image.LANCZOS)
        image = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)
        
        probabilities = probabilities.cpu().numpy()[0]
        top_k_indices = np.argsort(probabilities)[-return_top_k:][::-1]
        
        results = {
            'image_path': str(image_path),
            'top_predictions': [
                {
                    'class_name': dataset.get_class_name(idx),
                    'class_index': int(idx),
                    'probability': float(probabilities[idx])
                }
                for idx in top_k_indices
            ]
        }
        
        return results
    
    def plot_confusion_matrix(self, metrics: Dict, save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            metrics (Dict): Metrics dictionary from evaluate()
            save_path (Optional[str]): Path to save figure. If None, displays it.
        """
        cm = metrics['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, cmap='viridis', cbar=True, fmt='.2f')
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            history (Dict): Training history from train.py
            save_path (Optional[str]): Path to save figure. If None, displays it.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(history['train_accuracy'], label='Train Accuracy')
        axes[1].plot(history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def load_trained_model(checkpoint_path: str, num_classes: int = 989,
                       device: torch.device = None) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        num_classes (int): Number of output classes. Default: 989
        device (torch.device): Device to load model on. Default: CPU
    
    Returns:
        nn.Module: Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_model(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_model(checkpoint_path: str, data_dir: str, num_classes: int = 989,
                   batch_size: int = 32) -> Dict:
    """
    Evaluate a trained model on test set.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        data_dir (str): Path to minecraft/blocks/ directory
        num_classes (int): Number of output classes. Default: 989
        batch_size (int): Batch size. Default: 32
    
    Returns:
        Dict: Evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_trained_model(checkpoint_path, num_classes, device)
    
    # Load test data
    test_dataset = MinecraftTextureDataset(data_dir, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                             shuffle=False)
    
    # Evaluate
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(test_loader)
    
    # Print results
    print(f"\n=== Test Set Evaluation ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
    print(f"Top-10 Accuracy: {metrics['top_10_accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    
    return metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Example usage
    blocks_dir = Path(__file__).parent.parent.parent / "minecraft" / "blocks"
    checkpoint_path = Path("checkpoints/classification/best_model.pt")
    
    if blocks_dir.exists() and checkpoint_path.exists():
        metrics = evaluate_model(str(checkpoint_path), str(blocks_dir))
    else:
        print(f"Either blocks directory or checkpoint not found")
