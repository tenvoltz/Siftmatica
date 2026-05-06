"""
Training pipeline for ResNet-18-lite classifier.

Trains the model on Minecraft block textures with optional augmentation,
tracks metrics, and saves checkpoints on validation improvement.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

from .model import create_model
from .dataset import create_dataloaders
from ..augmentation import AugmentationPipeline


class TrainingConfig:
    """Configuration for training."""
    
    def __init__(self):
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.patience = 10  # For LR scheduler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0
        self.checkpoint_dir = Path("checkpoints/classification")
        self.log_dir = Path("logs/classification")
        self.split_ratios = (0.7, 0.15, 0.15)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'patience': self.patience,
            'device': str(self.device),
            'split_ratios': self.split_ratios,
        }


class Trainer:
    """Trainer class for ResNet-18-lite model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = config.device
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def train(self, data_dir: str, num_classes: int = 989) -> Tuple[nn.Module, Dict]:
        """
        Train the model.
        
        Args:
            data_dir (str): Path to minecraft/blocks/ directory
            num_classes (int): Number of output classes. Default: 989
        
        Returns:
            Tuple[nn.Module, Dict]: (trained model, training history)
        """
        print(f"Device: {self.device}")
        print(f"Config:\n{json.dumps(self.config.to_dict(), indent=2)}")
        
        # Save config
        config_path = self.config.checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Create model
        model = create_model(num_classes=num_classes)
        model = model.to(self.device)
        
        # Create dataloaders
        print("\nLoading data...")
        augmentation = AugmentationPipeline()
        train_loader, val_loader, test_loader, actual_num_classes = create_dataloaders(
            data_dir,
            batch_size=self.config.batch_size,
            split_ratios=self.config.split_ratios,
            augmentation=augmentation,
            num_workers=self.config.num_workers
        )
        print(f"Actual number of classes: {actual_num_classes}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), 
                              lr=self.config.learning_rate,
                              weight_decay=self.config.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                     patience=self.config.patience//2, verbose=True)
        
        print("\nStarting training...")
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # Train phase
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Print progress
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{self.config.epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                  f"Elapsed: {elapsed_time/60:.1f}m")
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(model, epoch, val_loss, val_acc)
                print(f"✓ Checkpoint saved (best val loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
        
        # Save final history
        history_path = self.config.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining completed in {(time.time() - start_time)/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Load best model
        best_checkpoint = self._get_best_checkpoint()
        if best_checkpoint:
            model.load_state_dict(torch.load(best_checkpoint, map_location=self.device))
            print(f"Loaded best model from {best_checkpoint}")
        
        return model, self.history
    
    def _train_epoch(self, model: nn.Module, loader: DataLoader, 
                     criterion: nn.Module, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': total_loss / (len(pbar))})
        
        avg_loss = total_loss / len(loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def _validate_epoch(self, model: nn.Module, loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def _save_checkpoint(self, model: nn.Module, epoch: int, 
                        val_loss: float, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
        }
        checkpoint_path = self.config.checkpoint_dir / f"best_model.pt"
        torch.save(checkpoint, checkpoint_path)
    
    def _get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / "best_model.pt"
        if checkpoint_path.exists():
            return checkpoint_path
        return None


def train_classification_model(data_dir: str, config: Optional[TrainingConfig] = None) -> Tuple[nn.Module, Dict]:
    """
    Train a ResNet-18-lite classification model.
    
    Args:
        data_dir (str): Path to minecraft/blocks/ directory
        config (Optional[TrainingConfig]): Training configuration. If None, uses defaults.
    
    Returns:
        Tuple[nn.Module, Dict]: (trained model, history)
    """
    if config is None:
        config = TrainingConfig()
    
    trainer = Trainer(config)
    model, history = trainer.train(data_dir)
    
    return model, history


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Example usage
    blocks_dir = Path(__file__).parent.parent.parent / "minecraft" / "blocks"
    
    if blocks_dir.exists():
        config = TrainingConfig()
        config.epochs = 5  # Quick test with fewer epochs
        config.batch_size = 16
        
        model, history = train_classification_model(str(blocks_dir), config)
        print(f"Training completed. Final model: {type(model)}")
    else:
        print(f"Blocks directory not found: {blocks_dir}")
