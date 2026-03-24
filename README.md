# Siftmatica
Multiview Reconstruction of Minecraft Structures from Gameplay Images

# Environment
Minecraft 1.21.1
Fabric
Litematica

# Process/Comment
1. Keypoint detection + matching using SIFT
- Might want to mask away elements like hotbar, arms, etc.
- Ratio test, mutual check, RANSAC Geometric Filtering

2. Structure from Motion

3. Pixel projection to voxel grid
- Might want to extract wall, plane, etc. to compute normals and restrict canonical axes

4. Texture to block assignment
- Might want to look into Truncated Signed Distance Function (TSDF)

5. Litematica export