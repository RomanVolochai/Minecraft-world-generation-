import numpy as np
from tqdm import tqdm
import joblib

from models.naive_bayes import NaiveBayesModel

class MRFModel:
    
    def __init__(self, num_classes=24, unary_weight=1.0, pairwise_weight=1.0, smoothing=1e-5):
        self.num_classes = num_classes
        self.unary_weight = unary_weight
        self.pairwise_weight = pairwise_weight
        self.smoothing = smoothing
        
        self.nb_model = NaiveBayesModel(num_classes=num_classes)
        
        # shape: (num_classes, num_classes)
        self.pairwise = np.zeros((num_classes, num_classes))
        self.log_pairwise = None

    def fit(self, train_patches):
        
        print("Fitting MRF Model...")
        self.nb_model.fit(train_patches)
        
        print("Extracting pairwise statistics for MRF...")
        for patch in tqdm(train_patches, desc="Processing patches for MRF"):
            H, W = patch.shape
            
            lefts = patch[:, :-1].flatten()
            rights = patch[:, 1:].flatten()
            
            tops = patch[:-1, :].flatten()
            bottoms = patch[1:, :].flatten()
            
            node1 = np.concatenate([lefts, rights, tops, bottoms])
            node2 = np.concatenate([rights, lefts, bottoms, tops])
            
            combined = node1 * self.num_classes + node2
            counts = np.bincount(combined, minlength=self.num_classes**2)
            counts_2d = counts.reshape((self.num_classes, self.num_classes))
            
            self.pairwise += counts_2d
            
        self.pairwise += self.smoothing
        sums = self.pairwise.sum(axis=0, keepdims=True)
        self.pairwise /= sums
        
        self.log_pairwise = np.log(self.pairwise)

    def generate(self, size=(128, 128), sweeps=50, temperature=1.0, unary_probs=None):
        
        if self.log_pairwise is None:
            raise ValueError("Model must be fitted before generation.")
            
        H, W = size
        classes = np.arange(self.num_classes)
        
        if unary_probs is not None:
            print("Initializing grid with provided probabilities...")
            generated = np.argmax(unary_probs, axis=-1)
        else:
            print("Initializing grid with Naive Bayes...")
            generated = self.nb_model.generate(size)
        
        print(f"Running Gibbs sampling for {sweeps} sweeps...")
        
        for sweep in tqdm(range(sweeps), desc="Gibbs Sweeps"):
            for y in range(H):
                for x in range(W):
                    if unary_probs is not None:
                        nb_probs = unary_probs[y, x]
                        log_unary = np.log(nb_probs + 1e-10)
                    else:
                        causal_neighbors = []
                        if y > 0 and x > 0: causal_neighbors.append(generated[y-1, x-1]) # Top-Left
                        else: causal_neighbors.append(-1)
                        if y > 0: causal_neighbors.append(generated[y-1, x])             # Top
                        else: causal_neighbors.append(-1)
                        if y > 0 and x < W - 1: causal_neighbors.append(generated[y-1, x+1]) # Top-Right
                        else: causal_neighbors.append(-1)
                        if x > 0: causal_neighbors.append(generated[y, x-1])             # Left
                        else: causal_neighbors.append(-1)
                        
                        nb_probs = self.nb_model.predict_pixel_proba(causal_neighbors)
                        log_unary = np.log(nb_probs + 1e-10)
                    
                    top_n    = generated[y-1, x] if y > 0 else -1
                    bottom_n = generated[y+1, x] if y < H-1 else -1
                    left_n   = generated[y, x-1] if x > 0 else -1
                    right_n  = generated[y, x+1] if x < W-1 else -1
                    
                    direct_neighbors = [top_n, bottom_n, left_n, right_n]
                    
                    log_pairwise_sum = np.zeros(self.num_classes)
                    for n_val in direct_neighbors:
                        if n_val != -1:
                            log_pairwise_sum += self.log_pairwise[:, n_val]
                            
                    log_p = self.unary_weight * log_unary + self.pairwise_weight * log_pairwise_sum
                    
                    log_p /= temperature
                    
                    log_p -= np.max(log_p)
                    p = np.exp(log_p)
                    p /= p.sum()
                    
                    generated[y, x] = np.random.choice(classes, p=p)
                    
        return generated

    def predict(self, patches):
        
        return self.nb_model.predict(patches)

    def save(self, path):
        if self.log_pairwise is None:
            raise ValueError("Model must be fitted before saving.")
        state = {
            'pairwise': self.pairwise,
            'log_pairwise': self.log_pairwise,
            'nb_model_state': {
                'prior': self.nb_model.prior,
                'cond': self.nb_model.cond,
                'log_prior': self.nb_model.log_prior,
                'log_cond': self.nb_model.log_cond
            }
        }
        joblib.dump(state, path)
        print(f"MRFModel saved to {path}")

    def load(self, path):
        state = joblib.load(path)
        self.pairwise = state['pairwise']
        self.log_pairwise = state['log_pairwise']
        
        nb_state = state['nb_model_state']
        self.nb_model.prior = nb_state['prior']
        self.nb_model.cond = nb_state['cond']
        self.nb_model.log_prior = nb_state['log_prior']
        self.nb_model.log_cond = nb_state['log_cond']
        print(f"MRFModel loaded from {path}")
