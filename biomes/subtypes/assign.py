import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from biomes.utils import load_observable_groups, map_to_observable_groups
from biomes.visualize import BIOME_COLORS, hex_to_rgb
from surface.data.loader import get_inverse_mapping, load_mapping, load_world_data


def component_cells(grid, start_y, start_x, visited):
    target_class = grid[start_y, start_x]
    stack = [(start_y, start_x)]
    visited[start_y, start_x] = True
    cells = []

    while stack:
        y, x = stack.pop()
        cells.append((y, x))
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or nx < 0 or ny >= grid.shape[0] or nx >= grid.shape[1]:
                continue
            if visited[ny, nx] or grid[ny, nx] != target_class:
                continue
            visited[ny, nx] = True
            stack.append((ny, nx))

    return cells


def estimate_subtype_probabilities(dataset_dir, raw_biome_mapping, group_mapping, biome_to_group, max_worlds=None):
    files = sorted(Path(dataset_dir).glob("world_sample_*.npz"), key=lambda path: path.name)
    if max_worlds is not None:
        files = files[:max_worlds]

    num_biomes = max(raw_biome_mapping.values()) + 1
    counts = {group_id: np.zeros(num_biomes, dtype=np.float64) for group_id in group_mapping.values()}
    known_ids = set(raw_biome_mapping.values())

    for world_file in tqdm(files, desc="Estimating subtype frequencies"):
        data = load_world_data(world_file)
        grouped = map_to_observable_groups(data["biomes"], raw_biome_mapping, group_mapping, biome_to_group)
        for group_id in group_mapping.values():
            mask = grouped == group_id
            if not np.any(mask):
                continue
            values, value_counts = np.unique(data["biomes"][mask], return_counts=True)
            valid = np.isin(values, list(known_ids))
            values = values[valid]
            value_counts = value_counts[valid]
            if values.size == 0:
                continue
            counts[group_id][values] += value_counts

    probabilities = {}
    for group_id, group_counts in counts.items():
        valid = group_counts > 0
        if not np.any(valid):
            probabilities[group_id] = None
            continue
        ids = np.where(valid)[0]
        probs = group_counts[ids] / group_counts[ids].sum()
        probabilities[group_id] = (ids, probs)

    return probabilities


def assign_component_subtypes(observable_grid, subtype_probs, seed=None):
    rng = np.random.default_rng(seed)
    subtype_grid = np.zeros_like(observable_grid)
    visited = np.zeros(observable_grid.shape, dtype=bool)

    for y in range(observable_grid.shape[0]):
        for x in range(observable_grid.shape[1]):
            if visited[y, x]:
                continue
            cells = component_cells(observable_grid, y, x, visited)
            group_id = int(observable_grid[y, x])
            options = subtype_probs.get(group_id)
            if options is None:
                subtype_id = group_id
            else:
                ids, probs = options
                subtype_id = int(rng.choice(ids, p=probs))
            for cy, cx in cells:
                subtype_grid[cy, cx] = subtype_id

    return subtype_grid


def create_raw_biome_colormap(raw_biome_mapping):
    inverse_mapping = get_inverse_mapping(raw_biome_mapping)
    colors = []
    for biome_id in range(len(raw_biome_mapping)):
        biome_name = inverse_mapping.get(biome_id)
        colors.append(hex_to_rgb(BIOME_COLORS.get(biome_name, "#ff00ff")))
    return ListedColormap(colors)


def visualize_subtypes(subtype_grid, raw_biome_mapping, output_path, title):
    inverse_mapping = get_inverse_mapping(raw_biome_mapping)
    cmap = create_raw_biome_colormap(raw_biome_mapping)
    present = np.unique(subtype_grid)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(subtype_grid, cmap=cmap, vmin=0, vmax=len(raw_biome_mapping) - 1, interpolation="nearest")
    ax.axis("off")
    ax.set_title(title)

    handles = [
        mpatches.Patch(
            color=cmap.colors[int(biome_id)],
            label=inverse_mapping.get(int(biome_id), str(biome_id)).replace("minecraft:", ""),
        )
        for biome_id in present
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
    print(f"Saved subtype visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Assign raw biome subtypes to observable biome components")
    parser.add_argument("--input", required=True, help="Input observable group grid .npy")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--output", type=str, default=None, help="Output subtype PNG")
    parser.add_argument("--npy_output", type=str, default=None, help="Output subtype grid .npy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subtype sampling")
    parser.add_argument("--max_worlds", type=int, default=None, help="Optional subset for subtype frequencies")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    raw_biome_mapping = load_mapping(dataset_dir / "biomes_mapping.json")
    group_mapping, _, biome_to_group = load_observable_groups(dataset_dir / "observable_biomes_mapping.json")

    observable_grid = np.load(args.input)
    subtype_probs = estimate_subtype_probabilities(
        dataset_dir,
        raw_biome_mapping,
        group_mapping,
        biome_to_group,
        max_worlds=args.max_worlds,
    )
    subtype_grid = assign_component_subtypes(observable_grid, subtype_probs, seed=args.seed)

    output_path = args.output or str(Path(args.input).with_name(Path(args.input).stem + "_subtypes.png"))
    npy_output_path = args.npy_output or str(Path(output_path).with_suffix(".npy"))

    Path(npy_output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_output_path, subtype_grid)
    print(f"Saved subtype grid to {npy_output_path}")

    visualize_subtypes(
        subtype_grid,
        raw_biome_mapping,
        output_path=output_path,
        title=f"Subtype biomes from {Path(args.input).name}",
    )


if __name__ == "__main__":
    main()
