from pathlib import Path

import numpy as np

from surface.data.loader import get_inverse_mapping, load_mapping


def find_world_file(dataset_dir, world_id):
    matches = sorted(Path(dataset_dir).glob(f"world_sample_{world_id}_seed_*.npz"))
    if not matches:
        raise FileNotFoundError(f"World {world_id} not found in {dataset_dir}")
    return matches[0]


def load_observable_groups(filepath):
    data = load_mapping(filepath)
    groups = data["observable_biomes"]
    group_mapping = {name: info["id"] for name, info in groups.items()}
    group_colors = {info["id"]: info["color"] for info in groups.values()}
    biome_to_group = {}

    for group_name, info in groups.items():
        for biome_name in info["biomes"]:
            biome_to_group[biome_name] = group_name

    return group_mapping, group_colors, biome_to_group


def map_to_observable_groups(biome_map, biome_mapping, group_mapping, biome_to_group):
    inverse_biome_mapping = get_inverse_mapping(biome_mapping)
    unknown_id = group_mapping["unknown"]
    grouped = np.full_like(biome_map, unknown_id)

    for biome_id, biome_name in inverse_biome_mapping.items():
        group_name = biome_to_group.get(biome_name, "unknown")
        grouped[biome_map == biome_id] = group_mapping[group_name]

    return grouped


def dominant_grid(label_map, cell_size=16):
    height, width = label_map.shape
    grid_height = height // cell_size
    grid_width = width // cell_size
    result = np.zeros((grid_height, grid_width), dtype=label_map.dtype)

    for y in range(grid_height):
        for x in range(grid_width):
            cell = label_map[
                y * cell_size:(y + 1) * cell_size,
                x * cell_size:(x + 1) * cell_size,
            ]
            values, counts = np.unique(cell, return_counts=True)
            result[y, x] = values[np.argmax(counts)]

    return result


def expand_grid(grid, cell_size=16):
    return np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)


def extract_crops(map_data, crop_size=512):
    if crop_size is None or crop_size <= 0:
        return [map_data]

    height, width = map_data.shape
    crops = []
    for y in range(0, height - crop_size + 1, crop_size):
        for x in range(0, width - crop_size + 1, crop_size):
            crops.append(map_data[y:y + crop_size, x:x + crop_size])
    return crops


def crop_map(map_data, crop_x=0, crop_y=0, crop_size=512):
    if crop_size is None or crop_size <= 0:
        return map_data
    return map_data[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]


def surface_cell_histograms(surface, num_blocks, cell_size=16):
    height, width = surface.shape
    grid_height = height // cell_size
    grid_width = width // cell_size
    features = np.zeros((grid_height * grid_width, num_blocks), dtype=np.float32)

    row = 0
    for y in range(grid_height):
        for x in range(grid_width):
            cell = surface[
                y * cell_size:(y + 1) * cell_size,
                x * cell_size:(x + 1) * cell_size,
            ]
            counts = np.bincount(cell.ravel(), minlength=num_blocks)
            features[row] = counts / counts.sum()
            row += 1

    return features, (grid_height, grid_width)


def _cell_histogram(cell, num_blocks):
    counts = np.bincount(cell.ravel(), minlength=num_blocks).astype(np.float32)
    return counts / counts.sum()


def _histogram_entropy(histogram):
    nonzero = histogram[histogram > 0]
    if nonzero.size == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log(nonzero)))


def _contact_ratio(cell, block_a, block_b):
    if block_a is None or block_b is None:
        return 0.0

    horizontal_a = cell[:, :-1] == block_a
    horizontal_b = cell[:, 1:] == block_b
    vertical_a = cell[:-1, :] == block_a
    vertical_b = cell[1:, :] == block_b

    horizontal_contacts = np.count_nonzero(horizontal_a & horizontal_b)
    horizontal_contacts += np.count_nonzero((cell[:, :-1] == block_b) & (cell[:, 1:] == block_a))
    vertical_contacts = np.count_nonzero(vertical_a & vertical_b)
    vertical_contacts += np.count_nonzero((cell[:-1, :] == block_b) & (cell[1:, :] == block_a))

    total_edges = cell.shape[0] * (cell.shape[1] - 1) + (cell.shape[0] - 1) * cell.shape[1]
    return (horizontal_contacts + vertical_contacts) / max(1, total_edges)


def _mapping_id(block_mapping, block_name):
    return block_mapping.get(block_name)


def surface_cell_features(surface, block_mapping, cell_size=16):
    num_blocks = len(block_mapping)
    histograms, grid_shape = surface_cell_histograms(surface, num_blocks, cell_size=cell_size)
    grid_height, grid_width = grid_shape
    histogram_grid = histograms.reshape(grid_height, grid_width, num_blocks)

    context_histograms = np.zeros_like(histogram_grid)
    for y in range(grid_height):
        for x in range(grid_width):
            y0 = max(0, y - 1)
            y1 = min(grid_height, y + 2)
            x0 = max(0, x - 1)
            x1 = min(grid_width, x + 2)
            context_histograms[y, x] = histogram_grid[y0:y1, x0:x1].mean(axis=(0, 1))

    key_blocks = [
        "minecraft:water",
        "minecraft:sand",
        "minecraft:grass_block",
        "minecraft:snow",
        "minecraft:ice",
        "minecraft:packed_ice",
        "minecraft:blue_ice",
        "minecraft:stone",
        "minecraft:dirt",
        "minecraft:gravel",
        "minecraft:red_sand",
        "minecraft:mycelium",
    ]
    key_ids = [_mapping_id(block_mapping, block_name) for block_name in key_blocks]
    contact_pairs = [
        ("minecraft:water", "minecraft:sand"),
        ("minecraft:water", "minecraft:grass_block"),
        ("minecraft:sand", "minecraft:grass_block"),
        ("minecraft:grass_block", "minecraft:dirt"),
        ("minecraft:grass_block", "minecraft:stone"),
        ("minecraft:water", "minecraft:stone"),
    ]
    contact_ids = [
        (_mapping_id(block_mapping, block_a), _mapping_id(block_mapping, block_b))
        for block_a, block_b in contact_pairs
    ]

    extra_features = []
    for y in range(grid_height):
        for x in range(grid_width):
            cell = surface[
                y * cell_size:(y + 1) * cell_size,
                x * cell_size:(x + 1) * cell_size,
            ]
            hist = histogram_grid[y, x]
            dominant_ratio = float(np.max(hist))
            unique_ratio = np.count_nonzero(hist) / num_blocks
            entropy = _histogram_entropy(hist)
            key_ratios = [hist[block_id] if block_id is not None else 0.0 for block_id in key_ids]
            contacts = [_contact_ratio(cell, block_a, block_b) for block_a, block_b in contact_ids]
            extra_features.append([dominant_ratio, unique_ratio, entropy, *key_ratios, *contacts])

    extra_features = np.asarray(extra_features, dtype=np.float32)
    features = np.concatenate(
        [
            histograms,
            context_histograms.reshape(-1, num_blocks),
            extra_features,
        ],
        axis=1,
    )
    return features, grid_shape
