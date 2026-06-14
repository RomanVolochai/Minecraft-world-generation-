import numpy as np

from biomes.utils import expand_grid, surface_cell_histograms


class NaiveBayesBiomeModel:
    def __init__(self, num_blocks, num_biomes, smoothing=1.0):
        self.num_blocks = num_blocks
        self.num_biomes = num_biomes
        self.smoothing = smoothing
        self.log_prior = None
        self.log_block_probs = None

    def fit(self, X, y):
        class_counts = np.bincount(y, minlength=self.num_biomes).astype(np.float64)
        block_counts = np.full((self.num_biomes, self.num_blocks), self.smoothing, dtype=np.float64)

        for biome_id in range(self.num_biomes):
            mask = y == biome_id
            if np.any(mask):
                block_counts[biome_id] += X[mask].sum(axis=0)

        class_counts += self.smoothing
        self.log_prior = np.log(class_counts / class_counts.sum())
        self.log_block_probs = np.log(block_counts / block_counts.sum(axis=1, keepdims=True))

    def predict_proba(self, X):
        if self.log_prior is None or self.log_block_probs is None:
            raise ValueError("Model must be fitted before prediction.")

        scores = X @ self.log_block_probs.T
        scores += self.log_prior[None, :]
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_surface_grid(self, surface, cell_size=16):
        X, grid_shape = surface_cell_histograms(surface, self.num_blocks, cell_size=cell_size)
        predictions = self.predict(X)
        return predictions.reshape(grid_shape)

    def predict_surface_map(self, surface, cell_size=16):
        grid = self.predict_surface_grid(surface, cell_size=cell_size)
        return expand_grid(grid, cell_size=cell_size)
