# Embedding Model for Minecraft Texture Matching

## Overview

Learns a robust embedding space where damaged versions of the same texture are close together and different textures are far apart.

## Architecture

`TinyEmbeddingCNN` in `model.py`:
- 3 conv blocks (Conv2d + ReLU + BatchNorm)
- 1 max-pooling layer (after first block)
- Global average pooling
- Linear projection to 64D embedding
- L2 normalization

Input: (3, 16, 16) RGB normalized to [0, 1]  
Output: 64D L2-normalized embedding vector

## Training

Uses triplet loss with cosine distance:
- Anchor: clean texture
- Positive: augmented version (EdgeBleeding, AdjacentNoise, IrregularHoles)
- Negative: different texture

Run training:
```python
from src.classification.embedding import EmbeddingTrainer

trainer = EmbeddingTrainer()
model, history = trainer.train(num_epochs=50, batch_size=32)
trainer.save_model("checkpoints/embedding_model.pt")
```

## Inference

```python
from src.classification.embedding import EmbeddingMatcher, create_embedding_model

model = create_embedding_model()
matcher = EmbeddingMatcher(model)
matcher.precompute_embeddings()

best_texture, score, idx = matcher.find_best_match(query_image)
top_k = matcher.find_top_k_matches(query_image, k=5)
```

## Files

- `model.py` - TinyEmbeddingCNN architecture
- `dataset.py` - TripletTextureDataset with augmentation
- `train.py` - Training loop with TripletLoss
- `inference.py` - EmbeddingMatcher for texture retrieval
- `validate.py` - Compare against MaskedNearestNeighbor
- `__init__.py` - Module exports
