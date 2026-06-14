import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap

from .loader import load_mapping, load_world_data

BLOCK_COLORS = {
    0:  "#ffffff", # air
    1:  "#ebd982", # sand
    2:  "#d8c878", # sandstone
    3:  "#7d7d7d", # stone
    4:  "#59c93c", # grass_block
    5:  "#2c42c7", # water
    6:  "#856041", # dirt
    7:  "#9e6b5d", # granite
    8:  "#827e7b", # gravel
    9:  "#cfcfd1", # diorite
    10: "#8a7a6f", # iron_ore
    11: "#3b3a39", # coal_ore
    12: "#d65918", # lava
    13: "#84898a", # andesite
    14: "#9caeb3", # clay
    15: "#754e33", # coarse_dirt
    16: "#ffffff", # snow
    17: "#8cb7fa", # ice
    18: "#6ba2f5", # packed_ice
    19: "#f0fbfb", # snow_block
    20: "#4e8ff0", # blue_ice
    21: "#2c42c7", # bubble_column (water)
    22: "#5c4524", # podzol
    23: "#ede174", # gold_ore
    24: "#b85b38", # red_sand
    25: "#6b4a82", # mycelium
    26: "#7a1e1e", # redstone_ore
}

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def create_surface_colormap():
    max_id = max(BLOCK_COLORS.keys())
    colors = [hex_to_rgb("#000000")] * (max_id + 1)
    for block_id, hex_color in BLOCK_COLORS.items():
        colors[block_id] = hex_to_rgb(hex_color)
    return ListedColormap(colors)

def visualize_surface(surface_map, output_path=None, show=False):
    
    cmap = create_surface_colormap()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(surface_map, cmap=cmap, vmin=0, vmax=cmap.N-1, interpolation='nearest')
    ax.axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        print(f"Saved visualization to {output_path}")
    if show:
        plt.show()
    plt.close(fig)

def visualize_heightmap(heightmap, output_path=None, show=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heightmap, cmap='terrain', vmin=40, vmax=120, interpolation='bilinear')
    ax.axis('off')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Elevation (Y)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        print(f"Saved heightmap visualization to {output_path}")
    if show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a Minecraft world from dataset")
    parser.add_argument("--world", type=int, default=1, help="World ID to visualize (e.g. 1 for world_sample_1_...)")
    parser.add_argument("--output", type=str, default="outputs/real.png", help="Path to save the output PNG")
    args = parser.parse_args()
    
    dataset_dir = Path(__file__).parent.parent.parent / "dataset"
    matches = list(dataset_dir.glob(f"world_sample_{args.world}_seed_*.npz"))
    
    if not matches:
        print(f"World {args.world} not found in {dataset_dir}")
    else:
        file_path = matches[0]
        print(f"Visualizing {file_path.name}...")
        
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = load_world_data(file_path)
        surface = data['surface']
        surface[surface == 6] = 4
        visualize_surface(surface, output_path=out_path)
