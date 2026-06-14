import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.loader import load_mapping, get_inverse_mapping
from data.patches import create_dataset_splits
from models.baseline import BaselineModel
from models.naive_bayes import NaiveBayesModel
from models.mrf import MRFModel
from models.gmm_model import GMMModel
from models.gan import GANModel

def evaluate_predictions(y_true, y_pred, target_names=None):
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    acc = accuracy_score(y_true_flat, y_pred_flat)
    macro_f1 = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    
    print(f"Pixel Accuracy: {acc:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    if target_names is not None:
        labels = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
        filtered_names = [target_names[i] if i in target_names else str(i) for i in labels]
        
        print("\nClassification Report:")
        print(classification_report(y_true_flat, y_pred_flat, labels=labels, target_names=filtered_names, zero_division=0))
        
    return acc, macro_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", type=str, required=True, help="Comma separated list of models to evaluate (e.g. baseline,naive_bayes)")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load pre-trained model weights (for GAN)")
    parser.add_argument("--clean_dirt", action="store_true", help="Replace dirt with grass_block while loading data")
    args = parser.parse_args()
    
    models_to_test = [m.strip() for m in args.compare.split(',')]
    
    dataset_dir = Path(__file__).parent.parent.parent / "dataset"
    mapping = load_mapping(dataset_dir / "blocks_mapping.json")
    num_classes = len(mapping)
    
    print("Loading data splits...")
    splits = create_dataset_splits(dataset_dir, patch_size=128, clean_dirt=args.clean_dirt)
    
    inv_mapping = get_inverse_mapping(mapping)
    
    for model_name in models_to_test:
        print(f"\n{'='*40}")
        print(f"Evaluating Model: {model_name.upper()}")
        print(f"{'='*40}")
        
        if model_name == "baseline":
            model = BaselineModel()
            model.fit(splits['train'])
            y_true = splits['test']
            
            print("Evaluating predictions...")
            y_pred = model.predict(y_true)
            evaluate_predictions(y_true, y_pred, target_names=inv_mapping)
            
        elif model_name == "naive_bayes":
            model = NaiveBayesModel(num_classes=num_classes)
            model.fit(splits['train'])
            y_true = splits['test']
            
            print("Evaluating predictions (predictive spatial coherence)...")
            y_pred = model.predict(y_true)
            evaluate_predictions(y_true, y_pred, target_names=inv_mapping)
            
        elif model_name == "mrf":
            model = MRFModel(num_classes=num_classes)
            model.fit(splits['train'])
            y_true = splits['test']
            
            print("Evaluating predictions (MRF single-step)...")
            y_pred = model.predict(y_true)
            evaluate_predictions(y_true, y_pred, target_names=inv_mapping)
            
        elif model_name == "gmm":
            model = GMMModel(num_classes=num_classes)
            model.fit(splits['train'])
            y_true = splits['test']
            
            print("Evaluating predictions (GMM/PCA Reconstruction)...")
            y_pred = model.predict(y_true)
            evaluate_predictions(y_true, y_pred, target_names=inv_mapping)
            
        elif model_name == "gan":
            model = GANModel(num_epochs=1, num_classes=num_classes)
            if args.load_model:
                print(f"Loading GAN from {args.load_model}...")
                model.load(args.load_model)
            else:
                model.fit(splits['train'])
            print("Evaluating GAN Spatial Coherence (Takes time)...")
            gen = model.generate(size=(128, 128))
            print("Note: GAN generates completely novel maps, so pixel-wise accuracy against test patches is meaningless.")
            print("Please visually inspect the generated map using visualize_surface.")
            
        else:
            print(f"Model {model_name} is not implemented yet in evaluate.py.")

if __name__ == "__main__":
    main()
