"""
Generate summaries and alternative formats for the PNG to block state database.
"""

import json
from pathlib import Path
from collections import defaultdict


def generate_report(database_json, unmapped_json):
    """Generate a detailed report about the database."""
    
    with open(database_json, 'r') as f:
        database = json.load(f)
    
    with open(unmapped_json, 'r') as f:
        unmapped = json.load(f)
    
    # Statistics
    total_pngs = len(database) + len(unmapped)
    mapped_pngs = len(database)
    unmapped_count = len(unmapped)
    
    # Group by block state
    blockstate_to_textures = defaultdict(list)
    for png_file, info in database.items():
        blockstate = info['blockState']
        blockstate_to_textures[blockstate].append(png_file)
    
    # Find blocks with multiple textures
    blocks_with_multiple_textures = {
        bs: textures for bs, textures in blockstate_to_textures.items()
        if len(textures) > 1
    }
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("PNG to Block State Database - Summary Report")
    report.append("=" * 80)
    report.append("")
    
    report.append("STATISTICS")
    report.append("-" * 80)
    report.append(f"Total PNG files:              {total_pngs:,}")
    report.append(f"Successfully mapped:         {mapped_pngs:,} ({100*mapped_pngs/total_pngs:.1f}%)")
    report.append(f"Unmapped (debug/special):    {unmapped_count:,} ({100*unmapped_count/total_pngs:.1f}%)")
    report.append("")
    report.append(f"Unique block states:         {len(blockstate_to_textures):,}")
    report.append(f"Blocks with multiple textures: {len(blocks_with_multiple_textures):,}")
    report.append("")
    
    report.append("TOP 20 BLOCKS BY TEXTURE COUNT")
    report.append("-" * 80)
    sorted_blocks = sorted(
        blockstate_to_textures.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:20]
    for blockstate, textures in sorted_blocks:
        report.append(f"{blockstate:40} {len(textures):3} textures")
    report.append("")
    
    report.append("UNMAPPED TEXTURES")
    report.append("-" * 80)
    if unmapped:
        report.append("The following textures could not be mapped to any block state:")
        report.append("(These are typically debug textures, animation frames, or special effects)")
        report.append("")
        for item in unmapped:
            report.append(f"  {item['pngFile']}")
    else:
        report.append("All textures have been successfully mapped!")
    
    report.append("")
    report.append("=" * 80)
    
    return '\n'.join(report)


def generate_csv(database_json, output_csv):
    """Generate a CSV version of the database."""
    
    with open(database_json, 'r') as f:
        database = json.load(f)
    
    import csv
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PNG File', 'Block State', 'Block Name'])
        
        for png_file in sorted(database.keys()):
            info = database[png_file]
            writer.writerow([
                png_file,
                info['blockState'],
                info['blockName']
            ])


def generate_grouped_json(database_json, output_json):
    """Generate a JSON file grouped by block state."""
    
    with open(database_json, 'r') as f:
        database = json.load(f)
    
    # Group by block state
    grouped = defaultdict(list)
    for png_file, info in database.items():
        blockstate = info['blockState']
        grouped[blockstate].append(png_file)
    
    # Convert to sorted dict
    grouped_sorted = {
        bs: sorted(textures)
        for bs, textures in sorted(grouped.items())
    }
    
    with open(output_json, 'w') as f:
        json.dump(grouped_sorted, f, indent=2)


def main():
    script_dir = Path(__file__).parent
    database_json = script_dir / 'png_to_blockstate.json'
    unmapped_json = script_dir / 'unmapped_textures.json'
    report_txt = script_dir / 'database_report.txt'
    output_csv = script_dir / 'png_to_blockstate.csv'
    grouped_json = script_dir / 'blockstate_to_pngs.json'
    
    # Generate report
    print("Generating report...")
    report = generate_report(str(database_json), str(unmapped_json))
    with open(report_txt, 'w') as f:
        f.write(report)
    print(report)
    
    # Generate CSV
    print("\nGenerating CSV...")
    generate_csv(str(database_json), str(output_csv))
    print(f"✓ CSV saved to: {output_csv}")
    
    # Generate grouped JSON
    print("Generating grouped JSON...")
    generate_grouped_json(str(database_json), str(grouped_json))
    print(f"✓ Grouped JSON saved to: {grouped_json}")
    
    # Save report to file
    print(f"✓ Report saved to: {report_txt}")


if __name__ == '__main__':
    main()
