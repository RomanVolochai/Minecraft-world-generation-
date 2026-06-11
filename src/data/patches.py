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

def create_dataset_splits(dataset_dir, patch_size=128, splits=(48, 6, 6)):
    
    np.random.seed(42) # Deterministic split
    
    files = list(Path(dataset_dir).glob("world_sample_*.npz"))
    files.sort(key=lambda p: p.name)
    
    files = np.array(files)
    np.random.shuffle(files)
    
    train_count, val_count, test_count = splits
    if len(files) < sum(splits):
        print(f"Warning: Only {len(files)} files found, requested {sum(splits)} for splits.")
        total = len(files)
        train_count = int(total * (splits[0] / sum(splits)))
        val_count = int(total * (splits[1] / sum(splits)))
        test_count = total - train_count - val_count
    
    train_files = files[:train_count]
    val_files = files[train_count:train_count+val_count]
    test_files = files[train_count+val_count:train_count+val_count+test_count]
    
    def process_split(split_files):
        surface_patches = []
        for f in split_files:
            data = load_world_data(f)
            patches = extract_patches(data['surface'], patch_size)
            surface_patches.append(patches)
        if not surface_patches:
            return np.array([])
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
