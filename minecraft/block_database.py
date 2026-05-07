import json
from pathlib import Path
from typing import Optional, List, Dict, Set, Any
from functools import cached_property


class BlockDatabase:
    def __init__(self, database_dir: Optional[Path] = None):
        self.database_dir = Path(database_dir or Path(__file__).parent)
        self.blocks_path = self.database_dir / "blocks"

    def _load_json(self, filename: str, default: Any = None) -> Any:
        path = self.database_dir / filename
        if not path.exists():
            return default or {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @cached_property
    def png_to_blockstate(self) -> Dict[str, str]:
        data = self._load_json("png_to_blockstate.json")
        return {k: v["blockState"] for k, v in data.items()}

    @cached_property
    def blockstate_to_pngs(self) -> Dict[str, List[str]]:
        return self._load_json("blockstate_to_pngs.json")

    @cached_property
    def unmapped_textures(self) -> Set[str]:
        data = self._load_json("unmapped_textures.json", default=[])
        return {item["pngFile"] for item in data}

    @cached_property
    def valid_textures_list(self) -> List[str]:
        """Stable list for index-based access."""
        # Filter to only full block textures that are mapped to blockstates
        valid_textures = [
            png for png in self.png_to_blockstate.keys()
            if self.is_image_a_full_block(png)
        ]
        return sorted(valid_textures)
    
    def is_image_a_full_block(self, png_file: str) -> bool:
        if png_file == "air.png":
            return True
        # Load the image and check if all pixels are non-transparent
        img_path = self.blocks_path / png_file
        if not img_path.exists():
            return False
        from PIL import Image
        img = Image.open(img_path).convert("RGBA")
        return all(pixel[3] > 0 for pixel in img.getdata())
    
    
    def get_blockstate(self, png_file: str) -> Optional[str]:
        return self.png_to_blockstate.get(png_file)

    def get_textures(self, blockstate: str) -> Optional[List[str]]:
        return self.blockstate_to_pngs.get(blockstate)

    def is_mapped(self, png_file: str) -> bool:
        return png_file in self.png_to_blockstate

    def is_unmapped(self, png_file: str) -> bool:
        return png_file in self.unmapped_textures

    def get_all_blockstates(self) -> List[str]:
        return sorted(self.blockstate_to_pngs.keys())
    
    def get_all_valid_textures(self) -> List[str]:
        return self.valid_textures_list

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_pngs": len(self.png_to_blockstate) + len(self.unmapped_textures),
            "mapped_pngs": len(self.png_to_blockstate),
            "unmapped_pngs": len(self.unmapped_textures),
            "unique_blockstates": len(self.blockstate_to_pngs),
            "blocks_with_multiple_textures": sum(
                1 for textures in self.blockstate_to_pngs.values() if len(textures) > 1
            ),
        }

    def filter_by_blockstate(self, prefix: str) -> Dict[str, List[str]]:
        return {
            bs: tex
            for bs, tex in self.blockstate_to_pngs.items()
            if bs.startswith(prefix)
        }

    def get_texture_by_index(self, index: int) -> Optional[str]:
        if 0 <= index < len(self.valid_textures_list):
            return self.valid_textures_list[index]
        return None

    def get_image(self, filename: str):
        """Returns a PIL Image object if the file exists."""
        if filename in self.png_to_blockstate:
            img_path = self.blocks_path / filename
            if img_path.exists():
                from PIL import Image
                return Image.open(img_path)
        return None
    
    def get_texture_index(self, texture_name: str) -> Optional[int]:
        try:
            return self.valid_textures_list.index(texture_name)
        except ValueError:
            return None
    

_db_instance: Optional[BlockDatabase] = None

def get_database(database_dir: Optional[Path] = None) -> BlockDatabase:
    global _db_instance
    if _db_instance is None or database_dir is not None:
        _db_instance = BlockDatabase(database_dir)
    return _db_instance

def get_all_blockstates() -> List[str]:
    return get_database().get_all_blockstates()

def get_all_valid_textures() -> List[str]:
    return get_database().get_all_valid_textures()

def get_blockstate(png_file: str) -> Optional[str]:
    return get_database().get_blockstate(png_file)

def get_textures(blockstate: str) -> Optional[List[str]]:
    return get_database().get_textures(blockstate)

def get_image(filename: str):
    return get_database().get_image(filename)

if __name__ == "__main__":
    db = get_database()
    print(f"Database loaded! Mapped textures: {len(db.png_to_blockstate)}")
    print(f"Stats: {db.get_stats()}")

    test_png = "acacia_log_top.png"
    print(f"{test_png} -> {db.get_blockstate(test_png)}")
