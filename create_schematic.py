from litemapy import Schematic, Region, BlockState
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple


def _parse_voxel_key(voxel_key: str) -> Tuple[int, int, int]:
    x_str, y_str, z_str = voxel_key.split(",")
    return int(x_str), int(y_str), int(z_str)


def assign_textures(assignments_path: Path = Path("output/voxel_face_block_assignments.json")) -> Dict[Tuple[int, int, int], str]:
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Assignments file not found at {assignments_path}. "
            "Run playground.py first to generate voxel assignments."
        )

    with open(assignments_path, "r", encoding="utf-8") as f:
        assignments = json.load(f)

    key_texture_dict: Dict[Tuple[int, int, int], str] = {}
    for voxel_key, voxel_assignment in assignments.items():
        blockstate = voxel_assignment.get("voxel_blockstate")
        if not blockstate:
            continue
        key_texture_dict[_parse_voxel_key(voxel_key)] = blockstate

    if not key_texture_dict:
        raise ValueError("No valid voxel blockstates found in assignments file.")

    return key_texture_dict


def create_schematic(
    assignments_path: Path = Path("output/voxel_face_block_assignments.json"),
    output_path: Path = Path("output/output.litematic"),
):
    key_texture_dict = assign_textures(assignments_path)

    min_x, min_y, min_z = np.min(list(key_texture_dict.keys()), axis=0)
    max_x, max_y, max_z = np.max(list(key_texture_dict.keys()), axis=0)
    offset_x, offset_y, offset_z = abs(min_x), abs(min_y), abs(min_z)

    reg = Region(
        0,
        0,
        0,
        int(max_x + offset_x + 1),
        int(max_y + offset_y + 1),
        int(max_z + offset_z + 1),
    )
    schematic = reg.as_schematic(name="Output")
    print(f"Min voxel coordinates: ({min_x}, {min_y}, {min_z})")
    print(f"Max voxel coordinates: ({max_x}, {max_y}, {max_z})")
    print(f"Placing {len(key_texture_dict)} blocks into schematic")

    for key, texture in key_texture_dict.items():
        texture_id = "minecraft:" + texture
        block = BlockState(texture_id)
        reg[key[0] + offset_x, key[1] + offset_y, key[2] + offset_z] = block
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schematic.save(str(output_path))
    print(f"Saved schematic to: {output_path}")


if __name__ == "__main__":
    create_schematic()