import numpy as np

class BaselineModel:
    
    def __init__(self):
        self.frequencies = None
        self.classes = None

    def fit(self, train_patches):
        unique, counts = np.unique(train_patches, return_counts=True)
        self.classes = unique
        self.frequencies = counts / counts.sum()

    def generate(self, size=(128, 128)):
        if self.frequencies is None:
            raise ValueError("Model must be fitted before generation.")
        
        flat_size = np.prod(size)
        samples = np.random.choice(self.classes, size=flat_size, p=self.frequencies)
        return samples.reshape(size)

    def predict(self, patches):
        best_class = self.classes[np.argmax(self.frequencies)]
        return np.full_like(patches, best_class)
