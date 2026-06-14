import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from surface.data.loader import get_inverse_mapping, load_mapping, load_world_data
from biomes.utils import (
    dominant_grid,
    expand_grid,
    find_world_file,
    load_observable_groups,
    map_to_observable_groups,
)


BIOME_COLORS = {
    "unknown": "#ffffff",
    "desert": "#f2d16b",
    "grassland": "#7ec850",
    "forest_like": "#2f8f3a",
    "snowy": "#d6edf2",
    "snowy_beach": "#e8f1df",
    "beach": "#efd982",
    "river": "#2f6fd6",
    "frozen_river": "#89c6f0",
    "ocean": "#2f50c8",
    "frozen_ocean": "#8fc9f5",
    "wetland": "#4f7a3b",
    "mountains": "#8a8a8a",
    "stone_shore": "#9a9484",
    "minecraft:air": "#ffffff",
    "minecraft:desert_hills": "#d6b35f",
    "minecraft:desert": "#f2d16b",
    "minecraft:plains": "#7ec850",
    "minecraft:wooded_hills": "#3f7f3a",
    "minecraft:forest": "#2f8f3a",
    "minecraft:beach": "#efd982",
    "minecraft:sunflower_plains": "#9bd84a",
    "minecraft:river": "#2f6fd6",
    "minecraft:cold_ocean": "#315fb5",
    "minecraft:deep_cold_ocean": "#24478f",
    "minecraft:deep_ocean": "#1f3f8f",
    "minecraft:savanna": "#b9b84f",
    "minecraft:savanna_plateau": "#9f9e45",
    "minecraft:ocean": "#2f50c8",
    "minecraft:shattered_savanna": "#c0a84a",
    "minecraft:shattered_savanna_plateau": "#a58f3f",
    "minecraft:swamp": "#4f7a3b",
    "minecraft:birch_forest": "#65a85a",
    "minecraft:dark_forest": "#1f5f2f",
    "minecraft:taiga": "#3f806f",
    "minecraft:taiga_hills": "#326b5e",
    "minecraft:mountains": "#8a8a8a",
    "minecraft:stone_shore": "#9a9484",
    "minecraft:wooded_mountains": "#697d5f",
    "minecraft:birch_forest_hills": "#578f4f",
    "minecraft:flower_forest": "#5faa49",
    "minecraft:lukewarm_ocean": "#2f83d6",
    "minecraft:warm_ocean": "#32a6c8",
    "minecraft:deep_lukewarm_ocean": "#236aa8",
    "minecraft:taiga_mountains": "#466b64",
    "minecraft:jungle_edge": "#2fa144",
    "minecraft:jungle": "#168a2e",
    "minecraft:jungle_hills": "#126f27",
    "minecraft:dark_forest_hills": "#194f27",
    "minecraft:desert_lakes": "#e6ca6b",
    "minecraft:modified_gravelly_mountains": "#777777",
    "minecraft:frozen_ocean": "#8fc9f5",
    "minecraft:deep_frozen_ocean": "#5f9fd6",
    "minecraft:swamp_hills": "#416934",
    "minecraft:tall_birch_forest": "#73b464",
    "minecraft:gravelly_mountains": "#7f7f7f",
    "minecraft:modified_jungle": "#0f7f2a",
    "minecraft:snowy_beach": "#e8f1df",
    "minecraft:snowy_taiga": "#b6d8d2",
    "minecraft:snowy_taiga_hills": "#9cc8c2",
    "minecraft:snowy_tundra": "#f0f5f2",
    "minecraft:snowy_mountains": "#d6dedc",
    "minecraft:frozen_river": "#89c6f0",
    "minecraft:giant_tree_taiga_hills": "#2f695c",
    "minecraft:giant_tree_taiga": "#397b69",
    "minecraft:tall_birch_hills": "#639f58",
    "minecraft:bamboo_jungle": "#1f9f39",
    "minecraft:bamboo_jungle_hills": "#16802f",
}


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def create_biome_colormap(mapping, colors_by_id=None):
    inverse_mapping = get_inverse_mapping(mapping)
    colors = []
    for biome_id in range(len(mapping)):
        if colors_by_id is not None:
            colors.append(hex_to_rgb(colors_by_id.get(biome_id, "#ff00ff")))
        else:
            biome_name = inverse_mapping.get(biome_id)
            colors.append(hex_to_rgb(BIOME_COLORS.get(biome_name, "#ff00ff")))
    return ListedColormap(colors)


def format_biome_name(full_name):
    return full_name.replace("minecraft:", "")


def visualize_biomes(biome_map, mapping, output_path, cell_size=16, title=None, colors_by_id=None):
    inverse_mapping = get_inverse_mapping(mapping)
    coarse = dominant_grid(biome_map, cell_size=cell_size)
    display_map = expand_grid(coarse, cell_size=cell_size)

    cmap = create_biome_colormap(mapping, colors_by_id=colors_by_id)
    present_biomes = np.unique(coarse)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(display_map, cmap=cmap, vmin=0, vmax=len(mapping) - 1, interpolation="nearest")
    ax.axis("off")
    ax.set_title(title or f"Dominant biomes ({cell_size}x{cell_size} cells)")

    handles = [
        mpatches.Patch(
            color=cmap.colors[int(biome_id)],
            label=format_biome_name(inverse_mapping.get(int(biome_id), str(biome_id))),
        )
        for biome_id in present_biomes
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        fontsize="small",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved biome visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize dominant biome regions for a real dataset world")
    parser.add_argument("--world", type=int, default=1, help="World ID to visualize")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--output", type=str, default="outputs/biomes_world.png", help="Path to save PNG")
    parser.add_argument("--cell_size", type=int, default=16, help="Cell size used to pick dominant biomes")
    parser.add_argument("--raw", action="store_true", help="Visualize original 54 biome classes instead of observable groups")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    world_file = find_world_file(dataset_dir, args.world)
    biome_mapping = load_mapping(dataset_dir / "biomes_mapping.json")
    data = load_world_data(world_file)

    print(f"Visualizing biomes from {world_file.name}...")

    if args.raw:
        display_map = data["biomes"]
        display_mapping = biome_mapping
        colors_by_id = None
        title = f"World {args.world}: raw biomes ({args.cell_size}x{args.cell_size})"
    else:
        group_mapping, group_colors, biome_to_group = load_observable_groups(
            dataset_dir / "observable_biomes_mapping.json"
        )
        display_map = map_to_observable_groups(data["biomes"], biome_mapping, group_mapping, biome_to_group)
        display_mapping = group_mapping
        colors_by_id = group_colors
        title = f"World {args.world}: observable biome groups ({args.cell_size}x{args.cell_size})"

    visualize_biomes(
        display_map,
        display_mapping,
        output_path=args.output,
        cell_size=args.cell_size,
        title=title,
        colors_by_id=colors_by_id,
    )


if __name__ == "__main__":
    main()
