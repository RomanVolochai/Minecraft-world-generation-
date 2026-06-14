import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import joblib

from models.gmm_em import EMGMM

class GMMModel:
    
    def __init__(self, n_components=10, pca_components=50, num_classes=24, max_samples=100):
        self.num_classes = num_classes
        self.pca = PCA(n_components=pca_components, random_state=42)
        self.gmm = EMGMM(n_components=n_components)
        self.fitted = False
        self.max_samples = max_samples

    def _to_one_hot(self, patches):
        
        N, H, W = patches.shape
        one_hot = np.eye(self.num_classes, dtype=np.float32)[patches]  # Shape: (N, H, W, num_classes)
        return one_hot.reshape(N, H * W * self.num_classes)
        
    def fit(self, train_patches):
        
        if train_patches.shape[0] > self.max_samples:
            np.random.seed(42)
            indices = np.random.choice(train_patches.shape[0], self.max_samples, replace=False)
            sub_patches = train_patches[indices]
        else:
            sub_patches = train_patches
            
        print("Preparing data for GMM (One-Hot Encoding)...")
        X_one_hot = self._to_one_hot(sub_patches)
        
        actual_pca_components = min(self.pca.n_components, X_one_hot.shape[0])
        self.pca.n_components = actual_pca_components
        
        print(f"Fitting PCA ({self.pca.n_components} components) on {X_one_hot.shape} matrix...")
        X_pca = self.pca.fit_transform(X_one_hot)
        
        print(f"Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        self.gmm.fit(X_pca)
        self.fitted = True

    def generate(self, size=(128, 128), return_probs=False, **kwargs):
        
        if not self.fitted:
            raise ValueError("Model must be fitted before generation.")
            
        H, W = size
            
        print("Sampling from latent GMM space...")
        z = self.gmm.sample(n_samples=1)  # shape (1, pca_components)
        
        x_recon = self.pca.inverse_transform(z)  # shape (1, 128*128*24)
        
        x_recon_3d = x_recon.reshape((H, W, self.num_classes))
        
        if return_probs:
            p = np.clip(x_recon_3d, 0, None)
            p_sum = np.sum(p, axis=-1, keepdims=True) + 1e-10
            return p / p_sum
            
        generated = np.argmax(x_recon_3d, axis=-1)
        
        return generated

    def predict(self, patches):
        
        if not self.fitted:
            raise ValueError("Model not fitted.")
            
        X_oh = self._to_one_hot(patches)
        X_pca = self.pca.transform(X_oh)
        X_recon = self.pca.inverse_transform(X_pca)
        
        N, H, W = patches.shape
        X_recon_3d = X_recon.reshape((N, H, W, self.num_classes))
        predictions = np.argmax(X_recon_3d, axis=-1)
        
        return predictions

    def save(self, path):
        if not self.fitted:
            raise ValueError("Model must be fitted before saving.")
        state = {
            'pca': self.pca,
            'gmm_weights': self.gmm.weights_,
            'gmm_means': self.gmm.means_,
            'gmm_covariances': self.gmm.covariances_,
            'fitted': self.fitted
        }
        joblib.dump(state, path)
        print(f"GMMModel saved to {path}")

    def load(self, path):
        state = joblib.load(path)
        self.pca = state['pca']
        self.gmm.weights_ = state['gmm_weights']
        self.gmm.means_ = state['gmm_means']
        self.gmm.covariances_ = state['gmm_covariances']
        self.fitted = state['fitted']
        print(f"GMMModel loaded from {path}")
