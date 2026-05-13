import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .model import TinyEmbeddingCNN
from minecraft.block_database import BlockDatabase, get_database
from src.util.image_transforms import pil_to_tensor


class EmbeddingMatcher:
    
    def __init__(self, model: TinyEmbeddingCNN, 
                 database: Optional[BlockDatabase] = None,
                 device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        
        self.database = database or get_database()
        self.reference_embeddings = None
        self.texture_names = None
    
    def precompute_embeddings(self):
        self.model.eval()
        textures = self.database.get_all_valid_textures()
        embeddings = []
        
        with torch.no_grad():
            for texture_name in textures:
                pil_image = self.database.get_image(texture_name)
                if pil_image is None:
                    continue
                
                tensor = pil_to_tensor(pil_image, size=(16, 16)).float().unsqueeze(0).to(self.device)
                embedding = self.model(tensor).squeeze(0)
                embeddings.append(embedding.cpu())
        
        self.reference_embeddings = torch.stack(embeddings)
        self.texture_names = textures
        print(f"Precomputed {len(self.texture_names)} reference embeddings")
    
    def embed_query(self, image: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device).float()
            embedding = self.model(image)
            return embedding.squeeze(0).cpu()
    
    def find_best_match(self, query_image: torch.Tensor) -> Tuple[str, float, int]:
        if self.reference_embeddings is None:
            raise ValueError("Call precompute_embeddings() first")
        
        query_emb = self.embed_query(query_image)
        similarities = F.cosine_similarity(
            query_emb.unsqueeze(0), 
            self.reference_embeddings, 
            dim=1
        )
        
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        best_texture = self.texture_names[best_idx]
        
        return best_texture, best_score, best_idx
    
    def find_top_k_matches(self, query_image: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        if self.reference_embeddings is None:
            raise ValueError("Call precompute_embeddings() first")
        
        query_emb = self.embed_query(query_image)
        similarities = F.cosine_similarity(
            query_emb.unsqueeze(0), 
            self.reference_embeddings, 
            dim=1
        )
        
        top_k = torch.topk(similarities, min(k, len(similarities)))
        results = []
        for score, idx in zip(top_k.values, top_k.indices):
            results.append((self.texture_names[idx.item()], score.item()))
        
        return results
    
    def load_model(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")


if __name__ == "__main__":
    model = TinyEmbeddingCNN()
    matcher = EmbeddingMatcher(model)
    matcher.precompute_embeddings()
    
    test_texture = matcher.texture_names[0]
    pil_image = matcher.database.get_image(test_texture)
    query = pil_to_tensor(pil_image, size=(16, 16)).float()
    
    best_match, score, idx = matcher.find_best_match(query)
    print(f"Query: {test_texture}")
    print(f"Best match: {best_match} (score: {score:.4f})")
