from litemapy import Schematic, Region, BlockState
from texture_assignment import assign_textures
import numpy as np

def create_schematic():
    key_texture_dict = assign_textures()

    min_x, min_y, min_z = np.min(list(key_texture_dict.keys()), axis=0)
    max_x, max_y, max_z = np.max(list(key_texture_dict.keys()), axis=0)
    offset_x, offset_y, offset_z = abs(min_x), abs(min_y), abs(min_z)
    print(type(min_x), type(max_x), type(offset_x))

    reg = Region(0, 0, 0, int(max_x + offset_x + 1), int(max_y + offset_y + 1), int(max_z + offset_z + 1))
    schematic = reg.as_schematic(name="Output")
    print(f"Min voxel coordinates: ({min_x}, {min_y}, {min_z})")
    print(f"Max voxel coordinates: ({max_x}, {max_y}, {max_z})")
    
    for key, texture in key_texture_dict.items():
        texture_id = "minecraft:" + texture
        block = BlockState(texture_id)
        reg[key[0]+offset_x, key[1]+offset_y, key[2]+offset_z] = block
    schematic.save("./output.litematic")

if __name__ == "__main__":
    create_schematic()