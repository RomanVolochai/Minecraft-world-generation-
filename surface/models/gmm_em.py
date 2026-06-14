import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
import warnings

class EMGMM:
    
    def __init__(self, n_components=10, max_iter=100, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.weights_ = None       # shape: (K,)
        self.means_ = None         # shape: (K, D)
        self.covariances_ = None   # shape: (K, D) - diagonal covariances only
        
    def _compute_log_prob(self, X):
        
        N, D = X.shape
        log_prob = np.zeros((N, self.n_components))
        
        for k in range(self.n_components):
            diff = X - self.means_[k]
            inv_cov = 1.0 / self.covariances_[k]
            log_det = np.sum(np.log(self.covariances_[k]))
            
            mahala = np.sum(diff**2 * inv_cov, axis=1)
            
            log_prob[:, k] = -0.5 * (D * np.log(2 * np.pi) + log_det + mahala)
            
        return log_prob

    def _e_step(self, X):
        
        log_prob = self._compute_log_prob(X)
        log_weights = np.log(self.weights_ + 1e-15)
        
        log_rho = log_prob + log_weights
        
        log_prob_norm = logsumexp(log_rho, axis=1, keepdims=True)
        
        log_resp = log_rho - log_prob_norm
        
        return np.sum(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        
        N, D = X.shape
        resp = np.exp(log_resp)
        
        Nk = np.sum(resp, axis=0) + 1e-15
        
        self.weights_ = Nk / N
        
        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]
        
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.sum(resp[:, k:k+1] * diff**2, axis=0) / Nk[k]
            
            self.covariances_[k] += 1e-6

    def fit(self, X):
        
        print(f"Fitting GMM with {self.n_components} components using custom EM...")
        N, D = X.shape
        np.random.seed(self.random_state)
        
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state, n_init=1)
        kmeans.fit(X)
        
        self.means_ = kmeans.cluster_centers_
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        emp_var = np.var(X, axis=0) + 1e-6
        self.covariances_ = np.tile(emp_var, (self.n_components, 1))
        
        log_likelihood = -np.inf
        
        for i in range(self.max_iter):
            prev_ll = log_likelihood
            
            log_likelihood, log_resp = self._e_step(X)
            
            self._m_step(X, log_resp)
            
            if abs(log_likelihood - prev_ll) < self.tol:
                print(f"EM converged at iteration {i+1} with Log-Likelihood: {log_likelihood:.2f}")
                break
        else:
            warnings.warn(f"EM did not converge after {self.max_iter} iterations.")
            print(f"Final Log-Likelihood: {log_likelihood:.2f}")

    def sample(self, n_samples=1):
        
        np.random.seed(None)
        component_indices = np.random.choice(self.n_components, size=n_samples, p=self.weights_)
        
        samples = np.zeros((n_samples, self.means_.shape[1]))
        for i, k in enumerate(component_indices):
            samples[i] = np.random.normal(loc=self.means_[k], scale=np.sqrt(self.covariances_[k]))
            
        return samples
