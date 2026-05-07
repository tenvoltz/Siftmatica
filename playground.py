import numpy as np
import open3d as o3d
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from src.pipeline.alignment import PointCloudAlignment
from minecraft.block_database import get_database, BlockDatabase
from src.util.schematic import create_schematic

FACE_KEYS = ["+x", "-x", "+y", "-y", "+z", "-z"]


class NearestNeighborClassifier:
    def __init__(self, database: BlockDatabase, distance_metric: str = "cosine", black_threshold: float = 0.05):
        from src.classification.nearest_neighbor.nearest_neighbor import MaskedNearestNeighbor
        self.classifier = MaskedNearestNeighbor(
            database=database,
            distance_metric=distance_metric,
            black_threshold=black_threshold
        )
        self.classifier.add_reference_images_from_database()
    
    def classify(self, image_chw: np.ndarray) -> Dict[str, Any]:
        _, texture_name, distance = self.classifier.predict(image_chw)
        blockstate = self.classifier.database.get_blockstate(texture_name)
        return {
            "texture": texture_name,
            "blockstate": blockstate,
            "distance": float(distance),
        }


class EmbeddingClassifier:
    def __init__(self, database: BlockDatabase, model_path: str = "checkpoints/embedding_model.pt"):
        from src.classification.embedding.inference import EmbeddingMatcher
        from src.classification.embedding.model import create_embedding_model
        import torch
        
        self.model = create_embedding_model()
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded embedding model from {model_path}")
        
        self.matcher = EmbeddingMatcher(self.model, database)
        self.matcher.precompute_embeddings()
    
    def classify(self, image_chw: np.ndarray) -> Dict[str, Any]:
        import torch
        from src.util.image_transforms import ensure_tensor
        
        image = ensure_tensor(image_chw).float()
        texture_name, score, idx = self.matcher.find_best_match(image)
        blockstate = self.matcher.database.get_blockstate(texture_name)
        return {
            "texture": texture_name,
            "blockstate": blockstate,
            "distance": float(1.0 - score),  # Convert similarity to distance
        }


def _face_grid_to_chw(face_grid: np.ndarray) -> Tuple[np.ndarray, int]:
    valid_pixels = ~np.isnan(face_grid).any(axis=2)
    valid_count = int(valid_pixels.sum())
    if valid_count == 0: return np.zeros((3, face_grid.shape[0], face_grid.shape[1]), dtype=np.float32), 0
    clean_grid = np.nan_to_num(face_grid, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    chw = clean_grid.transpose(2, 0, 1)
    return chw, valid_count


def _classify_face(face_grid: np.ndarray, classifier, pixel_threshold: int = 10) -> Dict[str, Any]:
    image_chw, valid_pixel_count = _face_grid_to_chw(face_grid)
    if valid_pixel_count < pixel_threshold:
        return {
            "texture": None,
            "blockstate": None,
            "distance": None,
            "valid_pixels": 0,
        }
    
    result = classifier.classify(image_chw)
    result["valid_pixels"] = valid_pixel_count
    return result


def assign_textures_to_voxels(
    voxel_grid: Dict[Tuple[int, int, int], Dict[str, Any]], 
    database: BlockDatabase,
    classifier_type: str = "nearest_neighbor",
    black_threshold: float = 0.05
) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    
    if classifier_type == "embedding":
        classifier = EmbeddingClassifier(database)
    else:
        classifier = NearestNeighborClassifier(database, black_threshold=black_threshold)
    
    assignments = {}
    for voxel_key, voxel_data in voxel_grid.items():
        face_assignments = {}
        agreement_scores: Dict[str, float] = {}
        
        for face in FACE_KEYS:
            face_grid = voxel_data.get(f"{face}_color_grid")
            if face_grid is None:
                face_result = {
                    "texture": None,
                    "blockstate": None,
                    "distance": None,
                    "valid_pixels": 0,
                }
            else:
                face_result = _classify_face(face_grid, classifier)
            
            face_assignments[face] = face_result
            blockstate = face_result["blockstate"]
            pixel_weight = face_result["valid_pixels"]
            if blockstate is not None and pixel_weight > 0:
                agreement_scores[blockstate] = agreement_scores.get(blockstate, 0.0) + float(pixel_weight)
        
        final_blockstate = None
        if agreement_scores:
            final_blockstate = max(agreement_scores.items(), key=lambda item: item[1])[0]
        
        assignments[voxel_key] = {
            "faces": face_assignments,
            "agreement_scores": agreement_scores,
            "voxel_blockstate": final_blockstate,
        }
    
    return assignments


def save_assignments(assignments: Dict[Tuple[int, int, int], Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        f"{x},{y},{z}": value
        for (x, y, z), value in assignments.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved voxel block assignments to: {output_path}")


def main(classifier_type: str = "nearest_neighbor"):
    data_dir = Path("data/house2-dense")
    input_path = data_dir / "workspace" / "dense" / "0" / "fused.ply"
    pcd = o3d.io.read_point_cloud(str(input_path))
    aligner = PointCloudAlignment()
    voxel_grid = aligner.align_point_cloud(pcd)
    
    print("Alignment completed. Voxel grid created.")
    print(f"Number of voxels: {len(voxel_grid)}")
    
    database = get_database()
    assignments = assign_textures_to_voxels(voxel_grid, database, classifier_type=classifier_type)
    
    print("\nSample voxel assignments:")
    for i, (voxel_key, assignment) in enumerate(assignments.items()):
        print(f"Voxel {i}: Key={voxel_key}, Final={assignment['voxel_blockstate']}")
        for face in FACE_KEYS:
            face_data = assignment["faces"][face]
            print(
                f"  {face}: blockstate={face_data['blockstate']}, "
                f"texture={face_data['texture']}, pixels={face_data['valid_pixels']}"
            )
        if i >= 2:
            break
    
    output_file = Path("output/voxel_face_block_assignments.json")
    save_assignments(assignments, output_file)
    create_schematic(assignments_path=output_file, output_path=Path("output/output.litematic"))


if __name__ == "__main__":
    # main(classifier_type="nearest_neighbor")
    main(classifier_type="embedding")
