import json
import numpy as np
from pathlib import Path

def load_world_data(filepath: str | Path) -> dict:
    data = np.load(filepath)
    return {
        'surface': data['surface'],
        'biomes': data['biomes'],
        'heightmap': data['heightmap']
    }

def load_mapping(filepath: str | Path) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def get_inverse_mapping(mapping: dict) -> dict:
    return {v: k for k, v in mapping.items()}

if __name__ == "__main__":
    dataset_dir = Path(__file__).parent.parent.parent / "dataset"
    sample_file = list(dataset_dir.glob("world_sample_*.npz"))[0]
    
    data = load_world_data(sample_file)
    print(f"Loaded {sample_file.name}:")
    print(f"  Surface shape: {data['surface'].shape}, dtype: {data['surface'].dtype}")
    print(f"  Biomes shape: {data['biomes'].shape}, dtype: {data['biomes'].dtype}")
    print(f"  Heightmap shape: {data['heightmap'].shape}, dtype: {data['heightmap'].dtype}")
