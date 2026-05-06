"""
Generate a database mapping PNG textures to Minecraft block states.

This script:
1. Loads blocks_textures.json which contains block state information
2. Scans the blocks folder for PNG files
3. Extracts the base block name from each PNG by progressively removing suffixes
4. Maps each PNG to its corresponding block state
5. Outputs a database in JSON format
"""

import json
import os
from pathlib import Path


def extract_possible_block_names(png_filename):
    """
    Extract possible block names from a PNG filename.
    Returns a list of candidates in order of likelihood (longest first).
    
    Examples:
        acacia_log_top.png -> ['acacia_log_top', 'acacia_log', 'acacia']
        birch_log.png -> ['birch_log', 'birch']
        campfire_fire.png -> ['campfire_fire', 'campfire']
        fire_0.png -> ['fire_0', 'fire']
        destroy_stage_5.png -> ['destroy_stage_5', 'destroy_stage', 'destroy']
    """
    # Remove .png extension and .mcmeta
    name = png_filename.replace('.png', '').replace('.mcmeta', '')
    
    candidates = [name]
    
    # Generate progressively shorter names by removing the last component
    parts = name.split('_')
    while len(parts) > 1:
        parts.pop()
        candidates.append('_'.join(parts))
    
    return candidates


def load_block_states(json_path):
    """Load block states from blocks_textures.json."""
    with open(json_path, 'r') as f:
        blocks = json.load(f)
    
    # Create a mapping from block name to block state
    name_to_blockstate = {}
    for block in blocks:
        name = block.get('name', '')
        blockstate = block.get('blockState', '')
        if name and blockstate:
            name_to_blockstate[name] = blockstate
    
    return name_to_blockstate


def get_png_files(blocks_dir):
    """Get all PNG files in the blocks directory."""
    png_files = []
    for filename in os.listdir(blocks_dir):
        if filename.endswith('.png') and not filename.endswith('.mcmeta'):
            png_files.append(filename)
    return sorted(png_files)


def create_database(blocks_dir, blocks_textures_json):
    """Create the PNG to block state database."""
    
    # Load block states
    name_to_blockstate = load_block_states(blocks_textures_json)
    
    # Get PNG files
    png_files = get_png_files(blocks_dir)
    
    # Create mapping
    database = {}
    unmapped = []
    
    for png_file in png_files:
        candidates = extract_possible_block_names(png_file)
        
        # Try each candidate in order until we find a match
        found = False
        for block_name in candidates:
            if block_name in name_to_blockstate:
                blockstate = name_to_blockstate[block_name]
                database[png_file] = {
                    'blockState': blockstate,
                    'blockName': block_name,
                }
                found = True
                break
        
        if not found:
            unmapped.append({
                'pngFile': png_file,
                'candidates': candidates,
                'bestGuess': candidates[0],
            })
    
    return database, unmapped


def main():
    script_dir = Path(__file__).parent
    blocks_dir = script_dir / 'blocks'
    blocks_textures_json = script_dir / 'blocks_textures.json'
    output_json = script_dir / 'png_to_blockstate.json'
    unmapped_json = script_dir / 'unmapped_textures.json'
    
    print(f"Loading block states from {blocks_textures_json}...")
    print(f"Scanning PNG files in {blocks_dir}...")
    
    database, unmapped = create_database(str(blocks_dir), str(blocks_textures_json))
    
    # Save database
    with open(output_json, 'w') as f:
        json.dump(database, f, indent=2)
    
    # Save unmapped files
    if unmapped:
        with open(unmapped_json, 'w') as f:
            json.dump(unmapped, f, indent=2)
    
    print(f"\n✓ Database created: {output_json}")
    print(f"  Total PNG files mapped: {len(database)}")
    print(f"  Total PNG files: {len(database) + len(unmapped)}")
    
    if unmapped:
        print(f"\n⚠ Unmapped textures: {len(unmapped)}")
        print(f"  Details saved to: {unmapped_json}")
        
        # Show first few unmapped
        print("\n  First 10 unmapped textures:")
        for item in unmapped[:10]:
            print(f"    {item['pngFile']}")
            print(f"      Tried: {item['candidates']}")
    else:
        print("\n✓ All PNG files have been mapped!")


if __name__ == '__main__':
    main()
