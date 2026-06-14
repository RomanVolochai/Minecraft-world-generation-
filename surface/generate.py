import argparse
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))

from data.patches import create_dataset_splits
from data.visualize import visualize_surface
from data.loader import load_mapping
from models.baseline import BaselineModel
from models.naive_bayes import NaiveBayesModel
from models.mrf import MRFModel
from models.gmm_model import GMMModel
from models.gan import GANModel

def main():
    parser = argparse.ArgumentParser(description="Generate Minecraft worlds using ML models")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["baseline", "naive_bayes", "mrf", "gmm", "ensemble", "gan"],
                        help="Which model to use for generation.")
    parser.add_argument("--size", type=int, default=128, help="Size of the generated map (size x size)")
    parser.add_argument("--output", type=str, default="outputs/generated.png", help="Path to save the generated PNG")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--sweeps", type=int, default=50, help="Number of Gibbs sampling sweeps (only for MRF)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for MRF generation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for GAN training")
    parser.add_argument("--save_model", type=str, default=None, help="Path to save the trained model (e.g. weights/gan.pth)")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load pre-trained model weights")
    parser.add_argument("--continue_training", action="store_true", help="Continue training a loaded GAN model")
    parser.add_argument("--clean_dirt", action="store_true", help="Replace dirt with grass_block while loading training data")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to use for training GMM (to avoid OOM)")
    args = parser.parse_args()
    
    dataset_dir = Path(__file__).parent.parent / "dataset"
    block_mapping = load_mapping(dataset_dir / "blocks_mapping.json")
    num_classes = len(block_mapping)
    
    train_patches = None
    if args.load_model and args.model in ["ensemble", "gmm", "mrf", "naive_bayes"]:
        print("Skipping data loading since --load_model is provided...")
    else:
        print("Loading data for training...")
        splits = create_dataset_splits(dataset_dir, patch_size=args.size, clean_dirt=args.clean_dirt)
        train_patches = splits['train']

    
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
    
    npy_path = out_path.with_suffix(".npy")
    np.save(npy_path, generated_map)
    print(f"Saved raw map matrix to {npy_path}")
    print("Done!")

if __name__ == "__main__":
    main()
