import argparse
from pathlib import Path

import numpy as np

from biomes.metrics.evaluate import build_cell_dataset, limit_files, split_world_files
from biomes.models.mrf_smoothing import MRFSmoother
from biomes.models.random_forest import RandomForestBiomeModel
from biomes.utils import (
    crop_map,
    expand_grid,
    find_world_file,
    load_observable_groups,
    surface_cell_features,
)
from biomes.visualize import visualize_biomes
from surface.data.loader import get_inverse_mapping, load_mapping, load_world_data
from surface.data.patches import clean_surface_blocks


def class_ids(group_mapping, names):
    return [group_mapping[name] for name in names if name in group_mapping]


def predict_probability_grid(model, surface, block_mapping, cell_size):
    X, grid_shape = surface_cell_features(surface, block_mapping, cell_size=cell_size)
    raw_probs = model.predict_proba(X)
    probs = np.zeros((X.shape[0], model.num_biomes), dtype=np.float64)

    for output_index, class_id in enumerate(model.model.classes_):
        probs[:, int(class_id)] = raw_probs[:, output_index]

    return probs.reshape((*grid_shape, model.num_biomes))


def main():
    parser = argparse.ArgumentParser(description="Smooth Random Forest biome predictions with an MRF")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--world", type=int, default=None, help="World ID to predict and smooth")
    parser.add_argument("--input_surface", type=str, default=None, help="Path to custom surface block grid (.npy)")
    parser.add_argument("--cell_size", type=int, default=8, help="Cell size for biome grid")
    parser.add_argument("--crop_size", type=int, default=512, help="Crop size used for training and prediction")
    parser.add_argument("--crop_x", type=int, default=0, help="Prediction crop left coordinate")
    parser.add_argument("--crop_y", type=int, default=0, help="Prediction crop top coordinate")
    parser.add_argument("--clean_dirt", action="store_true", help="Replace dirt with grass_block in surface features")
    parser.add_argument("--max_train_worlds", type=int, default=None, help="Optional train subset for quick runs")
    parser.add_argument("--n_estimators", type=int, default=200, help="Number of Random Forest trees")
    parser.add_argument("--max_depth", type=int, default=None, help="Max tree depth for Random Forest")
    parser.add_argument("--min_samples_leaf", type=int, default=2, help="Min samples per leaf for Random Forest")
    parser.add_argument("--smoothness", type=float, default=1.2, help="MRF neighbor agreement strength")
    parser.add_argument("--iterations", type=int, default=10, help="MRF smoothing iterations")
    parser.add_argument("--min_region_size", type=int, default=6, help="Remove non-preserved regions smaller than this many cells")
    parser.add_argument("--output", type=str, default=None, help="Path to save smoothed PNG")
    parser.add_argument("--npy_output", type=str, default=None, help="Optional path to save smoothed observable group grid")
    parser.add_argument("--raw_output", type=str, default=None, help="Optional path for unsmoothed RF PNG")
    parser.add_argument("--save_model", type=str, default=None, help="Path to save fitted Random Forest")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load fitted Random Forest")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    block_mapping = load_mapping(dataset_dir / "blocks_mapping.json")
    biome_mapping = load_mapping(dataset_dir / "biomes_mapping.json")
    group_mapping, group_colors, biome_to_group = load_observable_groups(
        dataset_dir / "observable_biomes_mapping.json"
    )

    if args.load_model:
        print(f"Loading Random Forest from {args.load_model}...")
        model = RandomForestBiomeModel.load(args.load_model)
    else:
        splits = split_world_files(dataset_dir)
        train_files = limit_files(splits["train"], args.max_train_worlds)
        print(f"Training worlds: {len(train_files)}")

        X_train, y_train = build_cell_dataset(
            train_files,
            dataset_dir,
            block_mapping,
            biome_mapping,
            group_mapping,
            biome_to_group,
            cell_size=args.cell_size,
            clean_dirt=args.clean_dirt,
            feature_mode="full",
            crop_size=args.crop_size,
        )

        model = RandomForestBiomeModel(
            block_mapping=block_mapping,
            num_biomes=len(group_mapping),
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
        )
        model.fit(X_train, y_train)
        if args.save_model:
            print(f"Saving Random Forest to {args.save_model}...")
            model.save(args.save_model)

    if args.input_surface:
        print(f"Loading custom surface map from {args.input_surface}...")
        surface = np.load(args.input_surface)
        world_label = Path(args.input_surface).stem
    else:
        if args.world is None:
            parser.error("Either --world or --input_surface must be specified.")
        world_file = find_world_file(dataset_dir, args.world)
        print(f"Predicting observable biomes for {Path(world_file).name}...")
        data = load_world_data(world_file)
        surface = data["surface"]
        world_label = f"world_{args.world}"

    if args.clean_dirt:
        surface = clean_surface_blocks(surface, block_mapping)
    surface = crop_map(surface, crop_x=args.crop_x, crop_y=args.crop_y, crop_size=args.crop_size)

    unary_probs = predict_probability_grid(model, surface, block_mapping, cell_size=args.cell_size)
    raw_grid = np.argmax(unary_probs, axis=-1)

    if args.raw_output:
        visualize_biomes(
            expand_grid(raw_grid, cell_size=args.cell_size),
            group_mapping,
            output_path=args.raw_output,
            cell_size=args.cell_size,
            title=(
                f"{world_label}: RF raw observable biomes "
                f"crop {args.crop_x},{args.crop_y} ({args.cell_size}x{args.cell_size})"
            ),
            colors_by_id=group_colors,
        )

    inverse_mapping = get_inverse_mapping(group_mapping)
    edge_classes = class_ids(group_mapping, ["river", "frozen_river", "beach", "snowy_beach"])
    locked_classes = class_ids(group_mapping, ["river", "frozen_river"])
    preserve_small_classes = class_ids(
        group_mapping,
        ["river", "frozen_river", "beach", "snowy_beach", "stone_shore"],
    )
    smoother = MRFSmoother(
        num_classes=len(group_mapping),
        smoothness=args.smoothness,
        iterations=args.iterations,
        edge_preserve_classes=edge_classes,
        min_region_size=args.min_region_size,
        preserve_small_classes=preserve_small_classes,
        locked_classes=locked_classes,
    )
    smoothed_grid = smoother.smooth(unary_probs)
    output_path = args.output or (
        f"outputs/biomes_rf_mrf_{world_label}_crop_{args.crop_x}_{args.crop_y}.png"
    )
    npy_output_path = args.npy_output or Path(output_path).with_suffix(".npy")

    present = sorted(np.unique(smoothed_grid))
    present_names = ", ".join(inverse_mapping[int(class_id)] for class_id in present)
    print(f"Smoothed classes: {present_names}")
    Path(npy_output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_output_path, smoothed_grid)
    print(f"Saved observable biome grid to {npy_output_path}")

    visualize_biomes(
        expand_grid(smoothed_grid, cell_size=args.cell_size),
        group_mapping,
        output_path=output_path,
        cell_size=args.cell_size,
        title=(
            f"World {args.world}: RF + MRF observable biomes "
            f"crop {args.crop_x},{args.crop_y} ({args.cell_size}x{args.cell_size})"
        ),
        colors_by_id=group_colors,
    )


if __name__ == "__main__":
    main()
