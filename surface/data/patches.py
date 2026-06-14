import numpy as np
from pathlib import Path
from .loader import load_world_data


def extract_patches(world_map, patch_size=128):
    
    h, w = world_map.shape
    patches = []
    
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = world_map[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            
    return np.stack(patches)

def _resolve_split_counts(total_files, splits):
    
    if all(isinstance(value, float) for value in splits):
        train_ratio, val_ratio, test_ratio = splits
        ratio_sum = train_ratio + val_ratio + test_ratio
        train_count = int(total_files * train_ratio / ratio_sum)
        val_count = int(total_files * val_ratio / ratio_sum)
        test_count = total_files - train_count - val_count
        return train_count, val_count, test_count
        
    train_count, val_count, test_count = splits
    requested = train_count + val_count + test_count
    if total_files < requested:
        print(f"Warning: Only {total_files} files found, requested {requested} for splits.")
        train_ratio = train_count / requested
        val_ratio = val_count / requested
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        test_count = total_files - train_count - val_count
        
    return train_count, val_count, test_count


def create_dataset_splits(dataset_dir, patch_size=128, splits=(0.8, 0.1, 0.1), return_all=False):
    
    np.random.seed(42) # Deterministic split
    
    files = list(Path(dataset_dir).glob("world_sample_*.npz"))
    files.sort(key=lambda p: p.name)
    
    files = np.array(files)
    np.random.shuffle(files)
    
    train_count, val_count, test_count = _resolve_split_counts(len(files), splits)
    
    train_files = files[:train_count]
    val_files = files[train_count:train_count+val_count]
    test_files = files[train_count+val_count:train_count+val_count+test_count]
    
    def process_split(split_files):
        surface_patches = []
        biome_patches = []
        heightmap_patches = []
        for f in split_files:
            data = load_world_data(f)
            surface = data['surface']
            surface[surface == 6] = 4
            surface_patches.append(extract_patches(surface, patch_size))
            if return_all:
                biome_patches.append(extract_patches(data['biomes'], patch_size))
                heightmap_patches.append(extract_patches(data['heightmap'], patch_size))
        
        if not surface_patches:
            if return_all:
                return {'surface': np.array([]), 'biomes': np.array([]), 'heightmap': np.array([])}
            return np.array([])
            
        if return_all:
            return {
                'surface': np.concatenate(surface_patches, axis=0),
                'biomes': np.concatenate(biome_patches, axis=0),
                'heightmap': np.concatenate(heightmap_patches, axis=0)
            }
        return np.concatenate(surface_patches, axis=0)

    print(f"Processing Train split ({len(train_files)} worlds)...")
    train_patches = process_split(train_files)
    print(f"Processing Val split ({len(val_files)} worlds)...")
    val_patches = process_split(val_files)
    print(f"Processing Test split ({len(test_files)} worlds)...")
    test_patches = process_split(test_files)
    
    return {
        'train': train_patches,
        'val': val_patches,
        'test': test_patches
    }

if __name__ == "__main__":
    dataset_dir = Path(__file__).parent.parent.parent / "dataset"
    splits = create_dataset_splits(dataset_dir, patch_size=128)
    
    print(f"Train patches: {splits['train'].shape}")
    print(f"Val patches: {splits['val'].shape}")
    print(f"Test patches: {splits['test'].shape}")
