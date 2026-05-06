# Block Texture to Block State Database

This folder contains a complete database that maps Minecraft block PNG textures to their block states.

## Problem Statement

Minecraft block states can have multiple texture variants. For example:
- `acacia_log.png` and `acacia_log_top.png` both represent the **same block state** `acacia_log`
- The `blocks_textures.json` provides a one-to-one mapping, but doesn't capture the full texture variance

This database solves that by creating a many-to-one mapping: **multiple PNG files → one block state**.

## Database Files

### Main Databases

1. **`png_to_blockstate.json`** - Primary database
   - Maps PNG filename → block state information
   - Format: `{ "png_filename": { "blockState": "block_name", "blockName": "block_name" }, ... }`
   - ~989 PNG files mapped to 603 unique block states

2. **`blockstate_to_pngs.json`** - Reverse mapping
   - Maps block state → list of PNG filenames
   - Format: `{ "block_state": ["png1.png", "png2.png", ...], ... }`
   - Useful for finding all textures for a block

3. **`png_to_blockstate.csv`** - Spreadsheet format
   - Tab-separated values: PNG File | Block State | Block Name
   - Easy to import into Excel, pandas, etc.

### Metadata

- **`unmapped_textures.json`** - Special textures that don't map to any block state
  - 23 files total (2.3% of textures)
  - Mostly debug textures, animation frames, and special effects:
    - `debug.png`, `debug2.png` - Debug textures (not real blocks)
    - `destroy_stage_*.png` - Block breaking animation frames (10 files)
    - Various special effect textures (redstone_dust, item_frame, magma particles, etc.)

- **`database_report.txt`** - Human-readable statistics and report

## Statistics

- **Total PNG textures**: 1,012
- **Successfully mapped**: 989 (97.7%)
- **Unmapped/Special**: 23 (2.3%)
- **Unique block states**: 603
- **Blocks with multiple textures**: 192

### Top Blocks by Texture Count

| Block State | Texture Count |
|---|---|
| sniffer_egg | 18 |
| vault | 16 |
| crafter | 14 |
| trial_spawner | 11 |
| pointed_dripstone | 10 |

## Usage Examples

### Using Python (Recommended)

```python
from block_database_query import get_database

# Get the database
db = get_database()

# Query single PNG file
blockstate = db.get_blockstate('acacia_log_top.png')
print(blockstate)  # Output: 'acacia_log'

# Get all textures for a block
textures = db.get_textures('acacia_log')
print(textures)  # Output: ['acacia_log.png', 'acacia_log_top.png']

# Check if mapped
is_mapped = db.is_mapped('acacia_log_top.png')
print(is_mapped)  # Output: True

# Get statistics
stats = db.get_stats()
print(stats)

# Find all blocks starting with a prefix
acacia_blocks = db.filter_by_blockstate('acacia')
for block_state in acacia_blocks:
    print(f"{block_state}: {len(acacia_blocks[block_state])} textures")
```

### Using JSON

```python
import json

# Load the direct mapping
with open('png_to_blockstate.json', 'r') as f:
    database = json.load(f)

# Look up a PNG
info = database['acacia_log_top.png']
print(info['blockState'])  # 'acacia_log'

# Load the reverse mapping
with open('blockstate_to_pngs.json', 'r') as f:
    reverse_db = json.load(f)

# Find all textures for a block
textures = reverse_db['acacia_log']
print(textures)  # ['acacia_log.png', 'acacia_log_top.png']
```

### Using CSV

```python
import pandas as pd

# Load into pandas
df = pd.read_csv('png_to_blockstate.csv', sep='\t')

# Query
print(df[df['PNG File'] == 'acacia_log_top.png'])

# Group by block state
grouped = df.groupby('Block State')['PNG File'].apply(list)
print(grouped['acacia_log'])
```

## How the Database Was Generated

### Scripts in this folder:

1. **`create_png_to_blockstate_database.py`**
   - Scans the `blocks/` folder for PNG files
   - Extracts block names by progressively removing suffix components
   - Matches against block states in `blocks_textures.json`
   - Generates `png_to_blockstate.json` and `unmapped_textures.json`

2. **`generate_database_report.py`**
   - Creates human-readable report
   - Generates CSV and grouped JSON formats
   - Produces statistics

3. **`block_database_query.py`**
   - Provides `BlockDatabase` class with query methods
   - Can be imported and used in other Python scripts
   - Includes convenience functions

### Algorithm

For each PNG file (e.g., `acacia_log_top.png`):
1. Try exact match in block states
2. Remove last suffix component and try again: `acacia_log`
3. Continue removing components: `acacia`
4. Return the first match found, or mark as unmapped

This handles all variants:
- `acacia_log.png` → `acacia_log` (direct match)
- `acacia_log_top.png` → `acacia_log` (remove `_top`)
- `blast_furnace_front_on.png` → `blast_furnace` (remove `_front_on`)
- `destroy_stage_0.png` → unmapped (not a real block)

## Example Queries

### Find all wood log textures
```python
db = get_database()
wood_types = ['acacia', 'birch', 'cherry', 'dark_oak', 'jungle', 'oak', 'spruce', 'warped', 'crimson', 'mangrove', 'bamboo']
for wood in wood_types:
    logs = db.filter_by_blockstate(f'{wood}_log')
    if logs:
        print(f"{wood}: {logs[f'{wood}_log']}")
```

### Find all colored blocks
```python
colors = ['black', 'red', 'green', 'blue', 'yellow', 'white']
for color in colors:
    wool = db.get_textures(f'{color}_wool')
    if wool:
        print(f"{color} wool: {wool}")
```

### Find blocks with the most texture variants
```python
blockstate_data = db._load_blockstate_to_pngs()
top_blocks = sorted(blockstate_data.items(), key=lambda x: len(x[1]), reverse=True)[:10]
for block, textures in top_blocks:
    print(f"{block}: {len(textures)} textures")
```

## Extending the Database

To add new blocks or update the database:

1. **Add new PNG files** to the `blocks/` folder
2. **Update `blocks_textures.json`** if new block states are added
3. **Run `create_png_to_blockstate_database.py`** to regenerate the database
4. **Run `generate_database_report.py`** to update reports

## Notes

- Block names are derived from PNG filenames, not Minecraft's internal identifiers
- The mapping assumes Minecraft naming conventions (underscores as separators)
- Some textures (debug, animation frames) intentionally don't map to block states
- The database reflects the blocks_textures.json version used during generation

## Related Files

- `blocks/` - Folder containing all 1,012 PNG texture files
- `blocks_textures.json` - Minecraft block state to texture mapping (one-to-one)
- `block_database.py` - (May) Existing implementation you want to integrate with

---

Generated: 2026-05-06
