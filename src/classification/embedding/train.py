import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict
import time

from .model import TinyEmbeddingCNN, create_embedding_model
from .dataset import TripletTextureDataset, create_validation_datasets
from .inference import EmbeddingMatcher


class TripletLoss(nn.Module):
    
    def __init__(self, margin: float = 0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        pos_dist = 1 - torch.sum(anchor * positive, dim=1)
        neg_dist = 1 - torch.sum(anchor * negative, dim=1)
        
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


class EmbeddingTrainer:

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.history = {
            'train_loss': [], 
            'val_accuracy': [],
            'val_accuracy_by_level': {},
            'epoch_time': []
        }

    def train(self, num_epochs: int = 50, batch_size: int = 32, 
              learning_rate: float = 0.001, margin: float = 0.3):
        self.model = create_embedding_model().to(self.device)
        self.criterion = TripletLoss(margin=margin)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        train_dataset = TripletTextureDataset()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_datasets = create_validation_datasets()

        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training on all {len(train_dataset)} textures")

        for epoch in range(num_epochs):
            is_epoch_val = (epoch % 10 == 0 or epoch == num_epochs - 1)

            start_time = time.time()

            print(f"Training epoch {epoch + 1}/{num_epochs}...")
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            if is_epoch_val:
                print(f"Validating epoch {epoch + 1}/{num_epochs}...")
                val_results = self._validate_on_augmented(val_datasets)
                self.history['val_accuracy'].append(val_results['overall_accuracy'])
                self.history['val_accuracy_by_level'] = val_results['by_level']

            epoch_time = time.time() - start_time
            self.history['epoch_time'].append(epoch_time)

            if is_epoch_val:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Accuracy: {val_results['overall_accuracy']:.4f} | "
                    f"Time: {epoch_time:.1f}s")
                for level, acc in val_results['by_level'].items():
                    print(f"  {level}: {acc:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Time: {epoch_time:.1f}s")

        return self.model, self.history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        for anchor, positive, negative in loader:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)

            loss = self.criterion(anchor_emb, positive_emb, negative_emb)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate_on_augmented(self, val_datasets: Dict) -> Dict:
        self.model.eval()
        matcher = EmbeddingMatcher(self.model)
        matcher.precompute_embeddings()

        results_by_level = {}
        all_correct = 0
        all_total = 0

        for level, dataset in val_datasets.items():
            correct = 0
            total = 0

            for idx in range(len(dataset)):
                query, target_name = dataset[idx]
                predicted, score, pred_idx = matcher.find_best_match(query)

                if predicted == target_name:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0
            results_by_level[level] = accuracy
            all_correct += correct
            all_total += total

        return {
            'overall_accuracy': all_correct / all_total if all_total > 0 else 0,
            'by_level': results_by_level
        }

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("No model to save")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    OUTPUT_DIR = "checkpoints"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer = EmbeddingTrainer()
    model, history = trainer.train(num_epochs=100, batch_size=256, learning_rate=0.001, margin=0.5)
    trainer.save_model("checkpoints/embedding_model.pt")
