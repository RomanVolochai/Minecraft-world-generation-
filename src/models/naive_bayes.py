import numpy as np
from tqdm import tqdm

class NaiveBayesModel:
    
    def __init__(self, num_classes=24, smoothing=1.0):
        self.num_classes = num_classes
        self.smoothing = smoothing
        
        self.prior = np.zeros(num_classes)
        
        # shape: (4, num_classes, num_classes)
        self.cond = np.zeros((4, num_classes, num_classes))
        
        self.log_prior = None
        self.log_cond = None

    def fit(self, train_patches):
        
        print("Extracting statistics for Naive Bayes...")
        
        unique, counts = np.unique(train_patches, return_counts=True)
        for u, c in zip(unique, counts):
            if u < self.num_classes:
                self.prior[u] = c
                
        self.prior += self.smoothing
        self.prior /= self.prior.sum()
        self.log_prior = np.log(self.prior)
        
        
        
        for patch in tqdm(train_patches, desc="Processing patches"):
            H, W = patch.shape
            
            X_val = patch[1:H, 1:W-1]
            
            N_vals = [
                patch[0:H-1, 0:W-2], # N0: top-left
                patch[0:H-1, 1:W-1], # N1: top
                patch[0:H-1, 2:W],   # N2: top-right
                patch[1:H,   0:W-2], # N3: left
            ]
            
            X_flat = X_val.flatten()
            for i in range(4):
                N_flat = N_vals[i].flatten()
                
                combined = X_flat * self.num_classes + N_flat
                counts = np.bincount(combined, minlength=self.num_classes**2)
                counts_2d = counts.reshape((self.num_classes, self.num_classes))
                
                self.cond[i] += counts_2d
                
        self.cond += self.smoothing
        sums = self.cond.sum(axis=2, keepdims=True)
        self.cond /= sums
        
        self.log_cond = np.log(self.cond)

    def predict_pixel_proba(self, neighbors):
        
        log_p = self.log_prior.copy()
        
        for i, n_val in enumerate(neighbors):
            if n_val is not None and n_val >= 0:
                log_p += self.log_cond[i, :, n_val]
                
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        return p / p.sum()

    def generate(self, size=(128, 128)):
        
        if self.log_prior is None:
            raise ValueError("Model must be fitted before generation.")
            
        H, W = size
        generated = np.zeros((H, W), dtype=int) - 1
        
        classes = np.arange(self.num_classes)
        
        print("Generating map using Naive Bayes...")
        for y in tqdm(range(H), desc="Rows generated"):
            for x in range(W):
                n0 = generated[y-1, x-1] if (y > 0 and x > 0) else -1
                n1 = generated[y-1, x]   if (y > 0) else -1
                n2 = generated[y-1, x+1] if (y > 0 and x < W-1) else -1
                n3 = generated[y, x-1]   if (x > 0) else -1
                
                neighbors = [n0, n1, n2, n3]
                
                probs = self.predict_pixel_proba(neighbors)
                generated[y, x] = np.random.choice(classes, p=probs)
                
        return generated

    def predict(self, patches):
        
        N, H, W = patches.shape
        predictions = np.zeros_like(patches)
        
        best_prior_class = np.argmax(self.log_prior)
        predictions[:, :, :] = best_prior_class
        
        N_vals = [
            patches[:, 0:H-1, 0:W-2], # N0: top-left
            patches[:, 0:H-1, 1:W-1], # N1: top
            patches[:, 0:H-1, 2:W],   # N2: top-right
            patches[:, 1:H,   0:W-2], # N3: left
        ]
        
        scores = np.zeros((N, H-1, W-2, self.num_classes))
        scores += self.log_prior[None, None, None, :]
        
        for i in range(4):
            # log_cond[i] has shape (num_classes_c, num_classes_n)
            scores += self.log_cond[i].T[N_vals[i]]
            
        predictions[:, 1:H, 1:W-1] = np.argmax(scores, axis=-1)
        return predictions
