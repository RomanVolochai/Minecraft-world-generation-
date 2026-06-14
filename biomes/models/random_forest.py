from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from biomes.utils import expand_grid, surface_cell_features


class RandomForestBiomeModel:
    def __init__(
        self,
        block_mapping,
        num_biomes,
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ):
        self.block_mapping = block_mapping
        self.num_biomes = num_biomes
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_surface_grid(self, surface, cell_size=16):
        X, grid_shape = surface_cell_features(surface, self.block_mapping, cell_size=cell_size)
        predictions = self.predict(X)
        return predictions.reshape(grid_shape)

    def predict_surface_map(self, surface, cell_size=16):
        grid = self.predict_surface_grid(surface, cell_size=cell_size)
        return expand_grid(grid, cell_size=cell_size)

    def save(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "block_mapping": self.block_mapping,
                "num_biomes": self.num_biomes,
                "model": self.model,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath):
        checkpoint = joblib.load(filepath)
        instance = cls(
            block_mapping=checkpoint["block_mapping"],
            num_biomes=checkpoint["num_biomes"],
        )
        instance.model = checkpoint["model"]
        return instance
