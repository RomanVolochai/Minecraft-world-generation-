import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import entropy

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.loader import load_mapping, get_inverse_mapping
from data.patches import create_dataset_splits
from models.baseline import BaselineModel
from models.naive_bayes import NaiveBayesModel
from models.mrf import MRFModel
from models.gmm_model import GMMModel
from models.gan import GANModel
from models.gan512 import GAN512Model

def calculate_kl_divergence(y_true, y_pred, num_classes=27):
    hist_true, _ = np.histogram(y_true.flatten(), bins=np.arange(num_classes + 1))
    hist_pred, _ = np.histogram(y_pred.flatten(), bins=np.arange(num_classes + 1))
    
    epsilon = 1e-8
    p_true = (hist_true + epsilon) / np.sum(hist_true + epsilon)
    p_pred = (hist_pred + epsilon) / np.sum(hist_pred + epsilon)
    
    kl_div = entropy(p_pred, p_true)
    return kl_div

def calculate_cooccurrence_matrix(maps, num_classes=27):
    cooc = np.zeros((num_classes, num_classes))
    for m in maps:
        right = m[:, :-1]
        left = m[:, 1:]
        np.add.at(cooc, (right, left), 1)
        np.add.at(cooc, (left, right), 1)
        
        top = m[:-1, :]
        bottom = m[1:, :]
        np.add.at(cooc, (top, bottom), 1)
        np.add.at(cooc, (bottom, top), 1)
        
    cooc_sum = np.sum(cooc)
    if cooc_sum > 0:
        cooc = cooc / cooc_sum
    return cooc

def calculate_cooccurrence_mse(y_true, y_pred, num_classes=27):
    if y_true.ndim == 2:
        y_true = np.expand_dims(y_true, 0)
    if y_pred.ndim == 2:
        y_pred = np.expand_dims(y_pred, 0)
        
    cooc_true = calculate_cooccurrence_matrix(y_true, num_classes)
    cooc_pred = calculate_cooccurrence_matrix(y_pred, num_classes)
    
    mse = np.mean((cooc_true - cooc_pred) ** 2)
    return mse

def evaluate_predictions(y_true, y_pred, target_names=None):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    kl_div = calculate_kl_divergence(y_true, y_pred)
    cooc_mse = calculate_cooccurrence_mse(y_true, y_pred)
    
    print(f"Generative Metrics:")
    print(f"  - KL Divergence (Block Frequencies): {kl_div:.6f}")
    print(f"  - Spatial Co-occurrence MSE:         {cooc_mse:.6e}")
    
    if len(y_true_flat) == len(y_pred_flat):
        acc = accuracy_score(y_true_flat, y_pred_flat)
        macro_f1 = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
        
        print(f"Classification Metrics (Meaningless for GANs):")
        print(f"  - Pixel Accuracy: {acc:.4f}")
        print(f"  - Macro F1 Score: {macro_f1:.4f}")
        
        if target_names is not None:
            labels = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
            filtered_names = [target_names[i] if i in target_names else str(i) for i in labels]
            print("\nClassification Report:")
            print(classification_report(y_true_flat, y_pred_flat, labels=labels, target_names=filtered_names, zero_division=0))
    else:
        print("Classification Metrics skipped (shape mismatch).")
        
    return kl_div, cooc_mse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", type=str, choices=["baseline", "naive_bayes", "mrf", "gmm", "gan", "gan512"], required=True, help="Comma separated list of models to evaluate (e.g. baseline,naive_bayes)")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load pre-trained model weights (for GAN/GAN512)")
    args = parser.parse_args()
    
    models_to_test = [m.strip() for m in args.compare.split(',')]
    
    dataset_dir = Path(__file__).parent.parent.parent / "dataset"
    print("Loading data splits...")
    splits = create_dataset_splits(dataset_dir, patch_size=128)
    
    mapping = load_mapping(dataset_dir / "blocks_mapping.json")
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
            y_pred = model.generate(y_true.shape)
            evaluate_predictions(y_true, y_pred, target_names=inv_mapping)
            
        elif model_name == "naive_bayes":
            model = NaiveBayesModel()
            model.fit(splits['train'])
            y_true = splits['test'][:100] # Subsample to prevent OOM
            print("Evaluating predictions (predictive spatial coherence)...")
            y_pred = model.predict(y_true)
            evaluate_predictions(y_true, y_pred, target_names=inv_mapping)
            
        elif model_name == "mrf":
            model = MRFModel()
            model.fit(splits['train'])
            y_true = splits['test']
            print("Evaluating predictions (MRF single-step)...")
            y_pred = model.predict(y_true)
            evaluate_predictions(y_true, y_pred, target_names=inv_mapping)
            
        elif model_name == "gmm":
            model = GMMModel()
            model.fit(splits['train'])
            y_true = splits['test']
            print("Evaluating predictions (GMM/PCA Reconstruction)...")
            y_pred = model.predict(y_true)
            evaluate_predictions(y_true, y_pred, target_names=inv_mapping)
            
        elif model_name == "gan":
            model = GANModel(num_epochs=1)
            if args.load_model:
                print(f"Loading GAN from {args.load_model}...")
                model.load(args.load_model)
            else:
                model.fit(splits['train'])
            print("Evaluating GAN Generative Metrics...")
            y_pred = np.stack([model.generate(size=(128, 128)) for _ in range(10)])
            evaluate_predictions(splits['test'], y_pred)
            
        elif model_name == "gan512":
            model = GAN512Model(num_epochs=1)
            if args.load_model:
                print(f"Loading GAN512 from {args.load_model}...")
                model.load(args.load_model)
            else:
                model.fit(splits['train'])
            print("Evaluating GAN512 Generative Metrics...")
            y_pred = np.stack([model.generate(size=(512, 512)) for _ in range(5)])
            evaluate_predictions(splits['test'], y_pred)
            
        else:
            print(f"Model {model_name} is not implemented yet in evaluate.py.")

if __name__ == "__main__":
    main()
