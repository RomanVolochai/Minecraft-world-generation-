import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from biomes.models.naive_bayes import NaiveBayesBiomeModel
from biomes.models.random_forest import RandomForestBiomeModel
from biomes.utils import (
    dominant_grid,
    extract_crops,
    find_world_file,
    load_observable_groups,
    map_to_observable_groups,
    surface_cell_features,
    surface_cell_histograms,
)
from biomes.visualize import visualize_biomes
from surface.data.loader import get_inverse_mapping, load_mapping, load_world_data
from surface.data.patches import clean_surface_blocks


def split_world_files(dataset_dir, splits=(0.8, 0.1, 0.1), seed=42):
    files = sorted(Path(dataset_dir).glob("world_sample_*.npz"), key=lambda path: path.name)
    rng = np.random.default_rng(seed)
    files = np.array(files)
    rng.shuffle(files)

    total = len(files)
    train_count = int(total * splits[0] / sum(splits))
    val_count = int(total * splits[1] / sum(splits))

    return {
        "train": files[:train_count],
        "val": files[train_count:train_count + val_count],
        "test": files[train_count + val_count:],
    }


def build_cell_dataset(
    files,
    dataset_dir,
    block_mapping,
    biome_mapping,
    group_mapping,
    biome_to_group,
    cell_size=16,
    clean_dirt=False,
    feature_mode="histogram",
    crop_size=None,
):
    X_parts = []
    y_parts = []
    clean_mapping = block_mapping if clean_dirt else None

    for world_file in tqdm(files, desc="Loading biome cells"):
        data = load_world_data(world_file)
        surface = data["surface"]
        if clean_dirt:
            surface = clean_surface_blocks(surface, clean_mapping)

        grouped_biomes = map_to_observable_groups(
            data["biomes"],
            biome_mapping,
            group_mapping,
            biome_to_group,
        )
        surface_crops = extract_crops(surface, crop_size=crop_size)
        biome_crops = extract_crops(grouped_biomes, crop_size=crop_size)

        for surface_crop, biome_crop in zip(surface_crops, biome_crops):
            if feature_mode == "full":
                X, _ = surface_cell_features(surface_crop, block_mapping, cell_size=cell_size)
            else:
                X, _ = surface_cell_histograms(surface_crop, len(block_mapping), cell_size=cell_size)
            y = dominant_grid(biome_crop, cell_size=cell_size).ravel()

            X_parts.append(X)
            y_parts.append(y)

    if not X_parts:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=int)

    return np.vstack(X_parts), np.concatenate(y_parts).astype(int)


def limit_files(files, max_worlds):
    if max_worlds is None:
        return files
    return files[:max_worlds]


def evaluate_predictions(y_true, y_pred, group_mapping):
    inverse_mapping = get_inverse_mapping(group_mapping)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    target_names = [inverse_mapping.get(int(label), str(label)) for label in labels]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Cell Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate biome prediction models")
    parser.add_argument("--model", choices=["naive_bayes", "random_forest"], default="naive_bayes")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--cell_size", type=int, default=16, help="Cell size for surface histograms")
    parser.add_argument("--crop_size", type=int, default=None, help="Optional crop size for train/test maps")
    parser.add_argument("--clean_dirt", action="store_true", help="Replace dirt with grass_block in surface features")
    parser.add_argument("--max_train_worlds", type=int, default=None, help="Optional train subset for quick runs")
    parser.add_argument("--max_test_worlds", type=int, default=None, help="Optional test subset for quick runs")
    parser.add_argument("--predict_world", type=int, default=None, help="World ID to visualize with the fitted model")
    parser.add_argument("--output", type=str, default=None, help="Optional PNG path for first test prediction")
    parser.add_argument("--n_estimators", type=int, default=200, help="Number of trees for Random Forest")
    parser.add_argument("--max_depth", type=int, default=None, help="Max tree depth for Random Forest")
    parser.add_argument("--min_samples_leaf", type=int, default=2, help="Min samples per leaf for Random Forest")
    parser.add_argument("--save_model", type=str, default=None, help="Path to save fitted Random Forest")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load fitted Random Forest")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    block_mapping = load_mapping(dataset_dir / "blocks_mapping.json")
    biome_mapping = load_mapping(dataset_dir / "biomes_mapping.json")
    group_mapping, group_colors, biome_to_group = load_observable_groups(
        dataset_dir / "observable_biomes_mapping.json"
    )

    splits = split_world_files(dataset_dir)
    train_files = limit_files(splits["train"], args.max_train_worlds)
    test_files = limit_files(splits["test"], args.max_test_worlds)

    print(f"Training worlds: {len(train_files)}")
    print(f"Test worlds: {len(test_files)}")
    feature_mode = "full" if args.model == "random_forest" else "histogram"

    if args.load_model and args.model == "random_forest":
        X_train = None
        y_train = None
    else:
        X_train, y_train = build_cell_dataset(
            train_files,
            dataset_dir,
            block_mapping,
            biome_mapping,
            group_mapping,
            biome_to_group,
            cell_size=args.cell_size,
            clean_dirt=args.clean_dirt,
            feature_mode=feature_mode,
            crop_size=args.crop_size,
        )
    X_test, y_test = build_cell_dataset(
        test_files,
        dataset_dir,
        block_mapping,
        biome_mapping,
        group_mapping,
        biome_to_group,
        cell_size=args.cell_size,
        clean_dirt=args.clean_dirt,
        feature_mode=feature_mode,
        crop_size=args.crop_size,
    )

    if args.load_model and args.model == "random_forest":
        print(f"Loading Random Forest from {args.load_model}...")
        model = RandomForestBiomeModel.load(args.load_model)
    elif args.model == "random_forest":
        model = RandomForestBiomeModel(
            block_mapping=block_mapping,
            num_biomes=len(group_mapping),
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
        )
    else:
        model = NaiveBayesBiomeModel(
            num_blocks=len(block_mapping),
            num_biomes=len(group_mapping),
        )
    if not (args.load_model and args.model == "random_forest"):
        model.fit(X_train, y_train)
    if args.save_model and args.model == "random_forest":
        print(f"Saving Random Forest to {args.save_model}...")
        model.save(args.save_model)

    print(f"Evaluating {args.model} biome model...")
    y_pred = model.predict(X_test)
    evaluate_predictions(y_test, y_pred, group_mapping)

    if args.predict_world is not None:
        prediction_file = find_world_file(dataset_dir, args.predict_world)
        model_prefix = "rf" if args.model == "random_forest" else "nb"
        output_path = args.output or f"outputs/biomes_{model_prefix}_world_{args.predict_world}.png"
        title = f"World {args.predict_world}: {args.model} predicted observable biomes"
    elif args.output and len(test_files) > 0:
        prediction_file = test_files[0]
        output_path = args.output
        title = f"First test world: {args.model} predicted observable biomes"
    else:
        prediction_file = None
        output_path = None
        title = None

    if prediction_file is not None:
        print(f"Predicting observable biomes for {Path(prediction_file).name}...")
        data = load_world_data(prediction_file)
        surface = data["surface"]
        if args.clean_dirt:
            surface = clean_surface_blocks(surface, block_mapping)
        prediction_map = model.predict_surface_map(surface, cell_size=args.cell_size)
        visualize_biomes(
            prediction_map,
            group_mapping,
            output_path=output_path,
            cell_size=args.cell_size,
            title=f"{title} ({args.cell_size}x{args.cell_size})",
            colors_by_id=group_colors,
        )


if __name__ == "__main__":
    main()
