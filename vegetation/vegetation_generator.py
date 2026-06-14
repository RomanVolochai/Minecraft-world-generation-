import argparse
import json
import numpy as np
from pathlib import Path

# Mapping logs to saplings
LOG_TO_SAPLING = {
    "minecraft:oak_log": "minecraft:oak_sapling",
    "minecraft:spruce_log": "minecraft:spruce_sapling",
    "minecraft:birch_log": "minecraft:birch_sapling",
    "minecraft:jungle_log": "minecraft:jungle_sapling",
    "minecraft:acacia_log": "minecraft:acacia_sapling",
    "minecraft:dark_oak_log": "minecraft:dark_oak_sapling"
}

# Average logs per tree (to divide the histogram count)
LOG_DIVISORS = {
    "minecraft:oak_log": 5,
    "minecraft:spruce_log": 7,
    "minecraft:birch_log": 5,
    "minecraft:jungle_log": 5, 
    "minecraft:acacia_log": 5,
    "minecraft:dark_oak_log": 30 # dark oak is very thick and has many logs
}

# Multi-block plants
TALL_PLANTS_3 = ["minecraft:sugar_cane", "minecraft:cactus", "minecraft:kelp"]
TALL_PLANTS_2 = [
    "minecraft:tall_grass", "minecraft:large_fern", "minecraft:tall_seagrass",
    "minecraft:rose_bush", "minecraft:peony", "minecraft:lilac", "minecraft:sunflower"
]

# Plants to exclude entirely (lily pads, seagrass, kelp, vines, etc.)
EXCLUDED_PLANTS = {
    "minecraft:lily_pad",
    "minecraft:seagrass",
    "minecraft:tall_seagrass",
    "minecraft:kelp",
    "minecraft:kelp_plant",
    "minecraft:sea_pickle",
    "minecraft:vine"
}

# Allowed trees (logs and their saplings) per biome to prevent wrong generation
BIOME_VALID_TREES = {
    "minecraft:savanna": {"minecraft:acacia_log", "minecraft:acacia_sapling", "minecraft:oak_log", "minecraft:oak_sapling"},
    "minecraft:savanna_plateau": {"minecraft:acacia_log", "minecraft:acacia_sapling", "minecraft:oak_log", "minecraft:oak_sapling"},
    "minecraft:shattered_savanna": {"minecraft:acacia_log", "minecraft:acacia_sapling", "minecraft:oak_log", "minecraft:oak_sapling"},
    "minecraft:shattered_savanna_plateau": {"minecraft:acacia_log", "minecraft:acacia_sapling", "minecraft:oak_log", "minecraft:oak_sapling"},
    "minecraft:desert": set(),
    "minecraft:desert_hills": set(),
    "minecraft:desert_lakes": set(),
    "minecraft:badlands": set(),
    "minecraft:wooded_badlands_plateau": {"minecraft:oak_log", "minecraft:oak_sapling"},
    "minecraft:plains": {"minecraft:oak_log", "minecraft:oak_sapling", "minecraft:birch_log", "minecraft:birch_sapling"},
    "minecraft:sunflower_plains": {"minecraft:oak_log", "minecraft:oak_sapling", "minecraft:birch_log", "minecraft:birch_sapling"},
    "minecraft:forest": {"minecraft:oak_log", "minecraft:oak_sapling", "minecraft:birch_log", "minecraft:birch_sapling"},
    "minecraft:flower_forest": {"minecraft:oak_log", "minecraft:oak_sapling", "minecraft:birch_log", "minecraft:birch_sapling"},
    "minecraft:birch_forest": {"minecraft:birch_log", "minecraft:birch_sapling"},
    "minecraft:birch_forest_hills": {"minecraft:birch_log", "minecraft:birch_sapling"},
    "minecraft:dark_forest": {"minecraft:dark_oak_log", "minecraft:dark_oak_sapling"},
    "minecraft:dark_forest_hills": {"minecraft:dark_oak_log", "minecraft:dark_oak_sapling"},
    "minecraft:taiga": {"minecraft:spruce_log", "minecraft:spruce_sapling"},
    "minecraft:taiga_hills": {"minecraft:spruce_log", "minecraft:spruce_sapling"},
    "minecraft:taiga_mountains": {"minecraft:spruce_log", "minecraft:spruce_sapling"},
    "minecraft:snowy_taiga": {"minecraft:spruce_log", "minecraft:spruce_sapling"},
    "minecraft:snowy_taiga_hills": {"minecraft:spruce_log", "minecraft:spruce_sapling"},
    "minecraft:giant_tree_taiga": {"minecraft:spruce_log", "minecraft:spruce_sapling"},
    "minecraft:giant_tree_taiga_hills": {"minecraft:spruce_log", "minecraft:spruce_sapling"},
    "minecraft:mountains": {"minecraft:oak_log", "minecraft:oak_sapling", "minecraft:spruce_log", "minecraft:spruce_sapling"},
    "minecraft:wooded_mountains": {"minecraft:oak_log", "minecraft:oak_sapling", "minecraft:spruce_log", "minecraft:spruce_sapling"},
    "minecraft:swamp": {"minecraft:oak_log", "minecraft:oak_sapling"},
    "minecraft:swamp_hills": {"minecraft:oak_log", "minecraft:oak_sapling"},
    "minecraft:jungle": {"minecraft:jungle_log", "minecraft:jungle_sapling", "minecraft:oak_log", "minecraft:oak_sapling"},
    "minecraft:jungle_hills": {"minecraft:jungle_log", "minecraft:jungle_sapling", "minecraft:oak_log", "minecraft:oak_sapling"},
    "minecraft:jungle_edge": {"minecraft:jungle_log", "minecraft:jungle_sapling", "minecraft:oak_log", "minecraft:oak_sapling"},
    "minecraft:bamboo_jungle": {"minecraft:jungle_log", "minecraft:jungle_sapling", "minecraft:oak_log", "minecraft:oak_sapling"},
    "minecraft:bamboo_jungle_hills": {"minecraft:jungle_log", "minecraft:jungle_sapling", "minecraft:oak_log", "minecraft:oak_sapling"}
}

def main():
    parser = argparse.ArgumentParser(description="Generate 3D world matrix with vegetation")
    parser.add_argument("--surface", type=str, required=True, help="Path to surface .npy")
    parser.add_argument("--biomes", type=str, required=True, help="Path to biomes .npy")
    parser.add_argument("--heightmap", type=str, required=True, help="Path to heightmap .npy")
    parser.add_argument("--histogram", type=str, required=True, help="Path to vegetation_histogram.json")
    parser.add_argument("--biomes_mapping", type=str, default="dataset/biomes_mapping.json", help="Path to biomes_mapping.json")
    parser.add_argument("--output", type=str, default="outputs/world_3d.npy", help="Path to save 3D .npy")
    args = parser.parse_args()

    print(f"Loading data from {args.surface}, {args.biomes}, {args.heightmap}...")
    surface = np.load(args.surface)
    biomes = np.load(args.biomes)
    heightmap = np.load(args.heightmap)

    with open(args.biomes_mapping, 'r', encoding='utf-8') as f:
        biomes_mapping = json.load(f)
    inv_biomes_mapping = {v: k for k, v in biomes_mapping.items()}

    with open(args.histogram, 'r', encoding='utf-8') as f:
        histogram = json.load(f)

    # Validate shapes
    if surface.ndim == 3:
        surface = surface[0]
    if biomes.ndim == 3:
        biomes = biomes[0]
    if heightmap.ndim == 3:
        heightmap = heightmap[0]

    H, W = surface.shape
    
    # If biomes grid is smaller, upscale it to match surface
    if biomes.shape != (H, W):
        h_scale = H // biomes.shape[0]
        w_scale = W // biomes.shape[1]
        print(f"Upscaling biome grid by factor of {h_scale}x{w_scale} to match surface resolution...")
        biomes = np.repeat(np.repeat(biomes, h_scale, axis=0), w_scale, axis=1)

    # If heightmap grid is smaller (should not be, but just in case)
    if heightmap.shape != (H, W):
        h_scale = H // heightmap.shape[0]
        w_scale = W // heightmap.shape[1]
        print(f"Upscaling heightmap grid by factor of {h_scale}x{w_scale} to match surface resolution...")
        heightmap = np.repeat(np.repeat(heightmap, h_scale, axis=0), w_scale, axis=1)

    # Initialize 3D matrix (Y, X, Z) -> shape (256, H, W)
    # Using object array to store string block names
    world_3d = np.full((256, H, W), "minecraft:air", dtype=object)

    print("Building base terrain...")
    for x in range(H):
        for z in range(W):
            h = int(heightmap[x, z])
            if h < 0: h = 0
            if h > 255: h = 255
            
            # Fill underground
            if h >= 4:
                world_3d[0:h-4, x, z] = "minecraft:stone"
                world_3d[h-4:h, x, z] = "minecraft:dirt"
            else:
                world_3d[0:h, x, z] = "minecraft:stone"
            
            # Surface block
            world_3d[h, x, z] = surface[x, z]

    print("Processing probabilities...")
    biome_probs = {}
    for biome_name, data in histogram.items():
        if biome_name.startswith("_"):
            continue
        
        total_area = data.get("_total_area", 1)
        if total_area <= 0:
            total_area = 1
            
        probs = {}
        for block, count in data.items():
            if block == "_total_area":
                continue
                
            # Filter out aquatic plants and vines
            if block in EXCLUDED_PLANTS:
                continue
                
            # Filter out trees that do not belong to this biome
            if block.endswith("_log") or block.endswith("_sapling"):
                allowed_trees = BIOME_VALID_TREES.get(biome_name)
                if allowed_trees is not None and block not in allowed_trees:
                    continue
                
            actual_count = count
            if block in LOG_DIVISORS:
                actual_count = count / LOG_DIVISORS[block]
                
            mapped_block = LOG_TO_SAPLING.get(block, block)
            
            # Filter saplings and map check just in case
            if mapped_block in EXCLUDED_PLANTS:
                continue
            if mapped_block.endswith("_sapling"):
                allowed_trees = BIOME_VALID_TREES.get(biome_name)
                if allowed_trees is not None and mapped_block not in allowed_trees:
                    continue
            
            if mapped_block not in probs:
                probs[mapped_block] = 0
            probs[mapped_block] += actual_count / total_area
            
        total_prob = sum(probs.values())
        if total_prob > 1.0:
            for k in probs:
                probs[k] /= total_prob
            probs["minecraft:air"] = 0.0
        else:
            probs["minecraft:air"] = 1.0 - total_prob
            
        # Extract to list for np.random.choice
        blocks = list(probs.keys())
        p_vals = list(probs.values())
        p_vals = np.array(p_vals) / np.sum(p_vals) # ensure sum is exactly 1.0
        
        biome_probs[biome_name] = (blocks, p_vals)

    print("Placing vegetation...")
    for x in range(H):
        for z in range(W):
            h = int(heightmap[x, z])
            if h < 0 or h >= 254:
                continue
                
            biome_id = biomes[x, z]
            biome = inv_biomes_mapping.get(int(biome_id), "unknown")
            if biome not in biome_probs:
                continue
                
            blocks, p_vals = biome_probs[biome]
            
            if world_3d[h+1, x, z] != "minecraft:air":
                continue
                
            # Skip placing vegetation if the surface block is water, stone, or other rocky blocks
            surface_block = world_3d[h, x, z]
            if surface_block in {
                5, "minecraft:water", "water",
                3, "minecraft:stone", "stone",
                7, "minecraft:granite", "granite",
                9, "minecraft:diorite", "diorite",
                13, "minecraft:andesite", "andesite",
                2, "minecraft:sandstone", "sandstone",
                8, "minecraft:gravel", "gravel"
            }:
                continue
                
            chosen_block = np.random.choice(blocks, p=p_vals)
            
            if chosen_block == "minecraft:air":
                continue
                
            # If the surface block is sand, only allow sugar cane, dead bush, or cactus
            if surface_block in {1, "minecraft:sand", "sand"}:
                if chosen_block not in {"minecraft:sugar_cane", "minecraft:dead_bush", "minecraft:cactus"}:
                    continue
                
            if chosen_block == "minecraft:dark_oak_sapling":
                # Dark oak saplings need a 2x2 area to grow
                if x + 1 < H and z + 1 < W:
                    # Check if space is available
                    if (world_3d[h+1, x+1, z] == "minecraft:air" and 
                        world_3d[h+1, x, z+1] == "minecraft:air" and 
                        world_3d[h+1, x+1, z+1] == "minecraft:air"):
                        
                        world_3d[h+1, x, z] = chosen_block
                        world_3d[h+1, x+1, z] = chosen_block
                        world_3d[h+1, x, z+1] = chosen_block
                        world_3d[h+1, x+1, z+1] = chosen_block
            else:
                if chosen_block == "minecraft:sugar_cane":
                    # Sugar cane must have water adjacent to the block it is planted on
                    has_water = False
                    for nx, nz in [(x-1, z), (x+1, z), (x, z-1), (x, z+1)]:
                        if 0 <= nx < H and 0 <= nz < W:
                            adj_surface = world_3d[h, nx, nz]
                            if adj_surface == 5 or adj_surface == "minecraft:water" or adj_surface == "water":
                                has_water = True
                                break
                    if not has_water:
                        continue

                # Height extensions for tall plants
                if chosen_block in TALL_PLANTS_3:
                    world_3d[h+1, x, z] = chosen_block
                    if h + 2 < 256: world_3d[h+2, x, z] = chosen_block
                    if h + 3 < 256: world_3d[h+3, x, z] = chosen_block
                elif chosen_block in TALL_PLANTS_2:
                    world_3d[h+1, x, z] = chosen_block + "[half=lower]"
                    if h + 2 < 256: world_3d[h+2, x, z] = chosen_block + "[half=upper]"
                else:
                    world_3d[h+1, x, z] = chosen_block

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving 3D world matrix to {out_path}...")
    np.save(out_path, world_3d)
    print(f"Done! Matrix shape: {world_3d.shape}")

if __name__ == "__main__":
    main()
