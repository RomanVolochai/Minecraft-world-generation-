import argparse
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))

from data.patches import create_dataset_splits
from data.visualize import visualize_surface, visualize_heightmap
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
    parser.add_argument("--continue_training", action="store_true", help="Continue training a loaded GAN model")
    args = parser.parse_args()
    
    dataset_dir = Path(__file__).parent.parent / "dataset"
    print("Loading data for training...")
    
    if args.model == "unet_height":
        splits = create_dataset_splits(dataset_dir, patch_size=128, return_all=True)
    elif args.model == "gan512":
        splits = create_dataset_splits(dataset_dir, patch_size=512)
        train_patches = splits['train']
    else:
        splits = create_dataset_splits(dataset_dir, patch_size=128)
        train_patches = splits['train']
    
    if args.model == "baseline":
        model = BaselineModel()
        print("Training Baseline Model (calculating marginal frequencies)...")
        model.fit(train_patches)
    elif args.model == "naive_bayes":
        model = NaiveBayesModel()
        model.fit(train_patches)
    elif args.model == "mrf":
        model = MRFModel()
        model.fit(train_patches)
    elif args.model == "gmm":
        model = GMMModel()
        model.fit(train_patches)
    elif args.model == "ensemble":
        print("Training GMM...")
        gmm = GMMModel()
        gmm.fit(train_patches)
        print("Training MRF...")
        mrf = MRFModel()
        mrf.fit(train_patches)
    elif args.model == "gan":
        model = GANModel(num_epochs=args.epochs)
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
        train_data = splits['train']
        model = UNetHeightModel(num_epochs=args.epochs)
        if args.load_model:
            print(f"Loading U-Net from {args.load_model}...")
            model.load(args.load_model)
            if args.continue_training:
                print(f"Continuing training for {args.epochs} epochs...")
                model.fit(train_data['surface'], train_data['biomes'], train_data['heightmap'])
                if args.save_model:
                    print(f"Saving updated U-Net to {args.save_model}...")
                    model.save(args.save_model)
        else:
            print("Training U-Net...")
            model.fit(train_data['surface'], train_data['biomes'], train_data['heightmap'])
            if args.save_model:
                print(f"Saving U-Net to {args.save_model}...")
                model.save(args.save_model)
                
        # Generate on a test patch or specific input map
        if args.input_map:
            print(f"Loading specific map: {args.input_map}")
            from data.loader import load_world_data
            data = load_world_data(args.input_map)
            surface_patch = data['surface'][np.newaxis, ...]
            biome_patch = data['biomes'][np.newaxis, ...]
            # If map doesn't have heightmap (e.g. inference only), we can't show true heightmap
            has_true = 'heightmap' in data
            if has_true:
                true_heightmap = data['heightmap']
        else:
            test_data = splits['test']
            idx = np.random.randint(0, len(test_data['surface']))
            surface_patch = test_data['surface'][idx:idx+1]
            biome_patch = test_data['biomes'][idx:idx+1]
            true_heightmap = test_data['heightmap'][idx]
            has_true = True
        
        print("Generating heightmap...")
        pred_heightmap = model.generate(surface_patch, biome_patch)[0]
        
        out_base = str(args.output).replace('.png', '')
        visualize_surface(surface_patch[0], output_path=f"{out_base}_surface.png")
        visualize_heightmap(pred_heightmap, output_path=f"{out_base}_heightmap_pred.png")
        np.save(f"{out_base}_heightmap_pred.npy", pred_heightmap)
        
        if has_true:
            visualize_heightmap(true_heightmap, output_path=f"{out_base}_heightmap_true.png")
            
        print(f"Saved surface, heightmaps (.png) and raw matrix (.npy) to {out_base}_*")
        return
    
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        np.random.seed(args.seed)
    else:
        np.random.seed(None)
        
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
    print("Done!")

if __name__ == "__main__":
    main()
