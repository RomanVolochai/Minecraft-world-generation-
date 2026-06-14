import argparse
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))

from data.patches import create_dataset_splits
from data.visualize import visualize_surface, visualize_heightmap
from data.loader import load_mapping
from models.baseline import BaselineModel
from models.naive_bayes import NaiveBayesModel
from models.mrf import MRFModel
from models.gmm_model import GMMModel
from models.gan import GANModel
from models.gan512 import GAN512Model
from models.unet_height import UNetHeightModel

def main():
    parser = argparse.ArgumentParser(description="Generate Minecraft worlds using ML models")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["baseline", "naive_bayes", "mrf", "gmm", "ensemble", "gan", "gan512", "unet_height"],
                        help="Which model to use for generation.")
    parser.add_argument("--size", type=int, default=128, help="Size of the generated map (size x size)")
    parser.add_argument("--output", type=str, default="outputs/generated.png", help="Path to save the generated PNG")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--sweeps", type=int, default=50, help="Number of Gibbs sampling sweeps (only for MRF)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for MRF generation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for GAN training")
    parser.add_argument("--save_model", type=str, default=None, help="Path to save the trained model (e.g. weights/gan.pth)")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load pre-trained model weights")
    parser.add_argument("--input_map", type=str, default=None, help="Path to a specific .npz map to analyze (for U-Net)")
    parser.add_argument("--input_surface", type=str, default=None, help="Path to custom surface block grid (.npy)")
    parser.add_argument("--input_biomes", type=str, default=None, help="Path to custom biome grid (.npy)")
    parser.add_argument("--continue_training", action="store_true", help="Continue training a loaded GAN model")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to use for training GMM (to avoid OOM)")
    parser.add_argument("--no_smooth", dest="smooth_heightmap", action="store_false", help="Disable smoothing of the predicted heightmap")
    parser.add_argument("--smooth_sigma", type=float, default=1.0, help="Sigma for gaussian filter smoothing")
    parser.set_defaults(smooth_heightmap=True)
    args = parser.parse_args()
    
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        np.random.seed(args.seed)
        import torch
        torch.manual_seed(args.seed)
    else:
        np.random.seed(None)
    
    dataset_dir = Path(__file__).parent.parent / "dataset"
    block_mapping = load_mapping(dataset_dir / "blocks_mapping.json")
    num_classes = len(block_mapping)
    
    need_splits = True
    if args.load_model and not args.continue_training:
        if (args.input_surface and args.input_biomes) or args.input_map or args.model in ["ensemble", "gmm", "mrf", "naive_bayes"]:
            need_splits = False
            
    train_patches = None
    if need_splits:
        print("Loading data for training...")
        if args.model == "unet_height":
            splits = create_dataset_splits(dataset_dir, patch_size=128, return_all=True)
        elif args.model == "gan512":
            splits = create_dataset_splits(dataset_dir, patch_size=512)
            train_patches = splits['train']
        else:
            splits = create_dataset_splits(dataset_dir, patch_size=args.size)
            train_patches = splits['train']
    else:
        print("Skipping dataset loading (inference only)...")

    
    if args.model == "baseline":
        model = BaselineModel()
        print("Training Baseline Model (calculating marginal frequencies)...")
        model.fit(train_patches)
    elif args.model == "naive_bayes":
        model = NaiveBayesModel(num_classes=num_classes)
        if args.load_model:
            model.load(args.load_model)
        else:
            model.fit(train_patches)
            if args.save_model:
                model.save(args.save_model)
    elif args.model == "mrf":
        model = MRFModel(num_classes=num_classes)
        if args.load_model:
            model.load(args.load_model)
        else:
            model.fit(train_patches)
            if args.save_model:
                model.save(args.save_model)
    elif args.model == "gmm":
        model = GMMModel(num_classes=num_classes, max_samples=args.max_samples)
        if args.load_model:
            model.load(args.load_model)
        else:
            model.fit(train_patches)
            if args.save_model:
                model.save(args.save_model)
    elif args.model == "ensemble":
        gmm = GMMModel(num_classes=num_classes, max_samples=args.max_samples)
        mrf = MRFModel(num_classes=num_classes)
        
        if args.load_model:
            print(f"Loading ensemble models using base path {args.load_model}...")
            gmm.load(args.load_model + "_gmm.joblib")
            mrf.load(args.load_model + "_mrf.joblib")
        else:
            print("Training GMM...")
            gmm.fit(train_patches)
            print("Training MRF...")
            mrf.fit(train_patches)
            if args.save_model:
                print(f"Saving ensemble models to {args.save_model}...")
                gmm.save(args.save_model + "_gmm.joblib")
                mrf.save(args.save_model + "_mrf.joblib")
    elif args.model == "gan":
        model = GANModel(num_epochs=args.epochs, num_classes=num_classes)
        if args.load_model:
            print(f"Loading GAN from {args.load_model}...")
            model.load(args.load_model)
            if args.continue_training:
                print(f"Continuing training for {args.epochs} epochs...")
                model.fit(train_patches)
                if args.save_model:
                    print(f"Saving updated GAN to {args.save_model}...")
                    model.save(args.save_model)
        else:
            print("Training GAN...")
            model.fit(train_patches)
            if args.save_model:
                print(f"Saving GAN to {args.save_model}...")
                model.save(args.save_model)
    elif args.model == "gan512":
        model = GAN512Model(num_epochs=args.epochs)
        if args.load_model:
            print(f"Loading GAN512 from {args.load_model}...")
            model.load(args.load_model)
            if args.continue_training:
                print(f"Continuing training for {args.epochs} epochs...")
                model.fit(train_patches)
                if args.save_model:
                    print(f"Saving updated GAN512 to {args.save_model}...")
                    model.save(args.save_model)
        else:
            print("Training GAN512...")
            model.fit(train_patches)
            if args.save_model:
                print(f"Saving GAN512 to {args.save_model}...")
                model.save(args.save_model)
                
    elif args.model == "unet_height":
        model = UNetHeightModel(num_epochs=args.epochs)
        if args.load_model:
            print(f"Loading U-Net from {args.load_model}...")
            model.load(args.load_model)
            if args.continue_training:
                train_data = splits['train']
                print(f"Continuing training for {args.epochs} epochs...")
                model.fit(train_data['surface'], train_data['biomes'], train_data['heightmap'])
                if args.save_model:
                    print(f"Saving updated U-Net to {args.save_model}...")
                    model.save(args.save_model)
        else:
            train_data = splits['train']
            print("Training U-Net...")
            model.fit(train_data['surface'], train_data['biomes'], train_data['heightmap'])
            if args.save_model:
                print(f"Saving U-Net to {args.save_model}...")
                model.save(args.save_model)
                
        if args.input_surface and args.input_biomes:
            print(f"Loading custom surface: {args.input_surface}")
            print(f"Loading custom biomes: {args.input_biomes}")
            surface_patch = np.load(args.input_surface)[np.newaxis, ...]
            biome_raw = np.load(args.input_biomes)
            
            # If biomes grid is smaller (cell resolution), upscale it to match surface
            if biome_raw.shape != surface_patch.shape[1:]:
                h_scale = surface_patch.shape[1] // biome_raw.shape[0]
                w_scale = surface_patch.shape[2] // biome_raw.shape[1]
                print(f"Upscaling biome grid by factor of {h_scale}x{w_scale} to match surface resolution...")
                biome_raw = np.repeat(np.repeat(biome_raw, h_scale, axis=0), w_scale, axis=1)
                
            biome_patch = biome_raw[np.newaxis, ...]
            has_true = False
        elif args.input_map:
            print(f"Loading specific map: {args.input_map}")
            from data.loader import load_world_data
            data = load_world_data(args.input_map)
            sz = args.size
            surface_patch = data['surface'][np.newaxis, :sz, :sz]
            biome_patch = data['biomes'][np.newaxis, :sz, :sz]
            has_true = 'heightmap' in data
            if has_true:
                true_heightmap = data['heightmap'][:sz, :sz]
        else:
            test_data = splits['test']
            idx = np.random.randint(0, len(test_data['surface']))
            surface_patch = test_data['surface'][idx:idx+1]
            biome_patch = test_data['biomes'][idx:idx+1]
            true_heightmap = test_data['heightmap'][idx]
            has_true = True
        
        print("Generating heightmap...")
        pred_heightmap = model.generate(surface_patch, biome_patch)[0]
        
        if args.smooth_heightmap:
            print("Smoothing predicted heightmap to remove checkerboard artifacts and crevices...")
            import scipy.ndimage as ndimage
            pred_heightmap = ndimage.median_filter(pred_heightmap, size=3)
            pred_heightmap = ndimage.gaussian_filter(pred_heightmap, sigma=args.smooth_sigma)
        out_base = str(args.output).replace('.png', '')
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        visualize_surface(surface_patch[0], output_path=f"{out_base}_surface.png")
        visualize_heightmap(pred_heightmap, output_path=f"{out_base}_heightmap_pred.png")
        np.save(f"{out_base}_heightmap_pred.npy", pred_heightmap)
        
        if has_true:
            visualize_heightmap(true_heightmap, output_path=f"{out_base}_heightmap_true.png")
            
        print(f"Saved surface, heightmaps (.png) and raw matrix (.npy) to {out_base}_*")
        return
    

        
    print(f"Generating a {args.size}x{args.size} map...")
    if args.model == "mrf":
        generated_map = model.generate(size=(args.size, args.size), sweeps=args.sweeps, temperature=args.temperature)
    elif args.model == "gmm":
        generated_map = model.generate(size=(args.size, args.size))
    elif args.model == "ensemble":
        print("1. Generating global structure with GMM...")
        gmm_probs = gmm.generate(size=(args.size, args.size), return_probs=True)
        print("2. Refining local details with MRF...")
        generated_map = mrf.generate(size=(args.size, args.size), sweeps=args.sweeps, temperature=args.temperature, unary_probs=gmm_probs)
    else:
        generated_map = model.generate(size=(args.size, args.size))
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualize_surface(generated_map, output_path=out_path)
    
    npy_path = out_path.with_suffix(".npy")
    np.save(npy_path, generated_map)
    print(f"Saved raw map matrix to {npy_path}")
    print("Done!")

if __name__ == "__main__":
    main()
