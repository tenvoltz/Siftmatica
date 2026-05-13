# Siftmatica

Automated Minecraft structure reconstruction from multi-view images via 3D geometry and learned texture classification.

## Core Pipeline

| Stage | Technology | Purpose |
|-------|-----------|---------|
| **Point Cloud Generation (1 Image)** | Depth Anything V2 (Vision Transformer) | Per-pixel depth inference from single images |
| **Point Cloud Generation (2+ Image)** | COLMAP (SfM + MVS) | Multi-view structure-from-motion & dense point cloud generation |
||
| **Point Cloud I/O** | Open3D | Geometry processing, RANSAC plane fitting, normal estimation |
| **Voxel Alignment** | RANSAC Plane Segmentation + Gradient Analysis | Estimate axis orientation, block scale (via color-gradient periodicity), phase offset |
||
| **Texture Classification** | PyTorch CNN + Embedding Learning | Three classifiers: (A) Simple CNN, (B) Masked L₂ nearest-neighbor, (C) TinyEmbeddingCNN |
||
| **Schematic Export** | Litemapy (NBT) | Litematica-compatible `.litematic` file generation |

## Key Algorithms

- **Scale Recovery**: Color-gradient magnitude peaks in CIELAB space -> iterative reweighted least-squares for block periodicity
- **Phase Alignment**: Signed distance of RANSAC-detected planes + normal-guided point snapping
- **Face Partitioning**: Point assignment to voxel faces via surface normal orientation
- **Data Augmentation**: Edge bleeding simulation, adjacent noise corruption, irregular hole occlusion (3 intensity levels)

## Dependencies

**Vision & Geometry**: OpenCV, Open3D, SciPy  
**Deep Learning**: PyTorch 2.11.0, Torchvision 0.26.0  
**Data**: NumPy 2.4.4, Pandas 3.0.1, Pillow 12.1.1  
**Minecraft**: Litemapy 0.11.0b0, NBTlib 2.0.4  
**Utilities**: Matplotlib, Seaborn, TQDM

