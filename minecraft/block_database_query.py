import json
from pathlib import Path
from typing import Optional, List, Dict, Set

from src.util.image_transforms import load_image_as_tensor, pil_to_tensor


class BlockDatabase:
    def __init__(self, database_dir: Optional[Path] = None):
        if database_dir is None:
            database_dir = Path(__file__).parent
        
        self.database_dir = Path(database_dir)
        self._png_to_blockstate = None
        self._blockstate_to_pngs = None
        self._unmapped = None
    
    def _load_png_to_blockstate(self):
        if self._png_to_blockstate is None:
            path = self.database_dir / 'png_to_blockstate.json'
            with open(path, 'r') as f:
                data = json.load(f)
                self._png_to_blockstate = {
                    k: v['blockState'] for k, v in data.items()
                }
        return self._png_to_blockstate
    
    def _load_blockstate_to_pngs(self):
        if self._blockstate_to_pngs is None:
            path = self.database_dir / 'blockstate_to_pngs.json'
            with open(path, 'r') as f:
                self._blockstate_to_pngs = json.load(f)
        return self._blockstate_to_pngs
    
    def _load_unmapped(self):
        if self._unmapped is None:
            path = self.database_dir / 'unmapped_textures.json'
            with open(path, 'r') as f:
                unmapped = json.load(f)
                self._unmapped = {item['pngFile'] for item in unmapped}
        return self._unmapped
    
    def get_blockstate(self, png_file: str) -> Optional[str]:
        data = self._load_png_to_blockstate()
        return data.get(png_file)
    
    def get_textures(self, blockstate: str) -> Optional[List[str]]:
        data = self._load_blockstate_to_pngs()
        return data.get(blockstate)
    
    def is_mapped(self, png_file: str) -> bool:
        return self.get_blockstate(png_file) is not None
    
    def is_unmapped(self, png_file: str) -> bool:
        unmapped = self._load_unmapped()
        return png_file in unmapped
    
    def get_all_blockstates(self) -> List[str]:
        data = self._load_blockstate_to_pngs()
        return sorted(data.keys())
    
    def get_stats(self) -> Dict:
        png_data = self._load_png_to_blockstate()
        blockstate_data = self._load_blockstate_to_pngs()
        unmapped = self._load_unmapped()
        
        return {
            'total_pngs': len(png_data) + len(unmapped),
            'mapped_pngs': len(png_data),
            'unmapped_pngs': len(unmapped),
            'unique_blockstates': len(blockstate_data),
            'blocks_with_multiple_textures': sum(
                1 for textures in blockstate_data.values() if len(textures) > 1
            ),
        }
    
    def batch_query(self, png_files: List[str]) -> Dict[str, Optional[str]]:
        return {png: self.get_blockstate(png) for png in png_files}
    
    def filter_by_blockstate(self, blockstate_prefix: str) -> Dict[str, List[str]]:
        blockstate_data = self._load_blockstate_to_pngs()
        result = {}
        for blockstate, textures in blockstate_data.items():
            if blockstate.startswith(blockstate_prefix):
                result[blockstate] = textures
        return result
    
    def get_all_valid_textures(self) -> Set[str]:
        png_data = self._load_png_to_blockstate()
        return set(png_data.keys())
    
    def get_valid_texture_by_index(self, index: int) -> Optional[str]:
        textures = self.get_all_valid_textures()
        if 0 <= index < len(textures):
            return list(textures)[index]
        return None
    
    def get_PNG_from_filename(self, filename: str) -> Optional[str]:
        png_data = self._load_png_to_blockstate()
        if filename in png_data:
            img_path = self.database_dir / 'blocks' / filename
            if img_path.exists():
                from PIL import Image
                return Image.open(img_path)
        return None


_default_db = None

def get_database(database_dir: Optional[Path] = None) -> BlockDatabase:
    global _default_db
    if _default_db is None or database_dir is not None:
        _default_db = BlockDatabase(database_dir)
    return _default_db


def get_blockstate(png_file: str) -> Optional[str]:
    return get_database().get_blockstate(png_file)


def get_textures(blockstate: str) -> Optional[List[str]]:
    return get_database().get_textures(blockstate)

def get_all_blockstates() -> List[str]:
    return get_database().get_all_blockstates()

def get_all_valid_textures() -> Set[str]:
    return get_database().get_all_valid_textures()

def get_valid_texture_img_by_img_name(filename: str) -> Optional[str]:
    return get_database().get_PNG_from_filename(filename)

if __name__ == '__main__':
    db = get_database()
    
    print("Database loaded successfully!")
    print(f"Stats: {db.get_stats()}")
    print()
    
    print("Examples:")
    print(f"  acacia_log_top.png -> {db.get_blockstate('acacia_log_top.png')}")
    print(f"  acacia_log_bottom.png -> {db.get_blockstate('acacia_log_bottom.png')}")
    print(f"  Textures for 'acacia_log': {db.get_textures('acacia_log')}")
    print()
    
    print("All acacia blocks:")
    acacia_blocks = db.filter_by_blockstate('acacia')
    for blockstate in sorted(acacia_blocks.keys())[:5]:
        print(f"  {blockstate}: {len(acacia_blocks[blockstate])} textures")
