# Siftmatica
Multiview Reconstruction of Minecraft Structures from Gameplay Images

# Environment
Minecraft 1.16.1
Fabric
Litematica

# Process/Comment
1. Keypoint detection + matching using SIFT
- Might want to mask away elements like hotbar, arms, etc.
- Ratio test, mutual check, RANSAC Geometric Filtering

2. Structure from Motion

3. Pixel projection to voxel grid

4. Texture to block assignment

5. Litematica export