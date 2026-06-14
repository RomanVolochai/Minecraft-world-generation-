import argparse
import numpy as np
from pathlib import Path
import json
import anvil
from nbt import nbt as nbt_lib

OCEAN_BIOMES = {
    "minecraft:ocean", "minecraft:deep_ocean", "minecraft:lukewarm_ocean", 
    "minecraft:deep_lukewarm_ocean", "minecraft:deep_cold_ocean"
}
RIVER_BIOMES = {"minecraft:river"}
DESERT_BIOMES = {
    "minecraft:desert", "minecraft:desert_hills", "minecraft:badlands", 
    "minecraft:wooded_badlands_plateau"
}

BIOME_NAME_TO_ID = {
    "minecraft:ocean": 0,
    "minecraft:plains": 1,
    "minecraft:desert": 2,
    "minecraft:mountains": 3,
    "minecraft:forest": 4,
    "minecraft:taiga": 5,
    "minecraft:swamp": 6,
    "minecraft:river": 7,
    "minecraft:frozen_ocean": 10,
    "minecraft:frozen_river": 11,
    "minecraft:snowy_tundra": 12,
    "minecraft:snowy_mountains": 13,
    "minecraft:mushroom_fields": 14,
    "minecraft:mushroom_field_shore": 15,
    "minecraft:beach": 16,
    "minecraft:desert_hills": 17,
    "minecraft:wooded_hills": 18,
    "minecraft:taiga_hills": 19,
    "minecraft:mountain_edge": 20,
    "minecraft:jungle": 21,
    "minecraft:jungle_hills": 22,
    "minecraft:jungle_edge": 23,
    "minecraft:deep_ocean": 24,
    "minecraft:stone_shore": 25,
    "minecraft:snowy_beach": 26,
    "minecraft:birch_forest": 27,
    "minecraft:birch_forest_hills": 28,
    "minecraft:dark_forest": 29,
    "minecraft:snowy_taiga": 30,
    "minecraft:snowy_taiga_hills": 31,
    "minecraft:giant_spruce_taiga": 32,
    "minecraft:giant_spruce_taiga_hills": 33,
    "minecraft:wooded_badlands_plateau": 34,
    "minecraft:savanna": 35,
    "minecraft:savanna_plateau": 36,
    "minecraft:badlands": 37,
    "minecraft:wooded_badlands_plateau": 38,
    "minecraft:modified_badlands_plateau": 39,
    "minecraft:warm_ocean": 40,
    "minecraft:lukewarm_ocean": 41,
    "minecraft:cold_ocean": 42,
    "minecraft:deep_warm_ocean": 43,
    "minecraft:deep_lukewarm_ocean": 44,
    "minecraft:deep_cold_ocean": 45,
    "minecraft:deep_frozen_ocean": 46,
    "minecraft:the_void": 127,
    "minecraft:sunflower_plains": 129,
    "minecraft:desert_lakes": 130,
    "minecraft:gravelly_mountains": 131,
    "minecraft:flower_forest": 132,
    "minecraft:taiga_mountains": 133,
    "minecraft:swamp_hills": 134,
    "minecraft:ice_spikes": 140,
    "minecraft:modified_jungle": 149,
    "minecraft:modified_jungle_edge": 151,
    "minecraft:tall_birch_forest": 155,
    "minecraft:tall_birch_hills": 156,
    "minecraft:dark_forest_hills": 157,
    "minecraft:snowy_taiga_mountains": 158,
    "minecraft:giant_coarse_taiga": 160,
    "minecraft:giant_coarse_taiga_hills": 161,
    "minecraft:shattered_savanna": 163,
    "minecraft:shattered_savanna_plateau": 164,
    "minecraft:eroded_badlands": 165,
    "minecraft:modified_wooded_badlands_plateau": 166,
    "minecraft:modified_badlands_plateau": 167,
    "minecraft:bamboo_jungle": 168,
    "minecraft:bamboo_jungle_hills": 169
}

class BiomeEmptyChunk(anvil.EmptyChunk):
    def __init__(self, x: int, z: int):
        super().__init__(x, z)
        # 127 is the legacy numeric ID for minecraft:the_void
        self.biomes_data = [127] * 1024

    def set_biome(self, local_x: int, local_y: int, local_z: int, biome_id: int):
        bx = local_x // 4
        bz = local_z // 4
        by = local_y // 4
        idx = (by * 16) + (bz * 4) + bx
        if 0 <= idx < 1024:
            self.biomes_data[idx] = biome_id

    def save(self) -> nbt_lib.NBTFile:
        root = super().save()
        level = root['Level']
        biomes_tag = nbt_lib.TAG_Int_Array(name='Biomes')
        biomes_tag.value = self.biomes_data
        level.tags.append(biomes_tag)
        return root

def main():
    parser = argparse.ArgumentParser(description="Generate .mca regions from 3D matrix")
    parser.add_argument("--world3d", type=str, required=True, help="Path to world_3d.npy")
    parser.add_argument("--biomes", type=str, required=True, help="Path to biomes.npy")
    parser.add_argument("--heightmap", type=str, required=True, help="Path to heightmap.npy")
    parser.add_argument("--biomes_mapping", type=str, default="dataset/biomes_mapping.json", help="Path to biomes_mapping.json")
    parser.add_argument("--output_dir", type=str, default="regions", help="Directory to save .mca files")
    args = parser.parse_args()

    print("Loading data matrices...")
    world_3d = np.load(args.world3d, allow_pickle=True)
    biomes = np.load(args.biomes)
    heightmap = np.load(args.heightmap)

    with open(args.biomes_mapping, 'r', encoding='utf-8') as f:
        biomes_mapping = json.load(f)
    inv_biomes_mapping = {v: k for k, v in biomes_mapping.items()}

    # Clean dimensions
    if biomes.ndim == 3:
        biomes = biomes[0]
    if heightmap.ndim == 3:
        heightmap = heightmap[0]

    # Assume world_3d is (256, H, W)
    _, H, W = world_3d.shape

    # Upscale biomes and heightmap if needed (like in vegetation generator)
    if biomes.shape != (H, W):
        biomes = np.repeat(np.repeat(biomes, H // biomes.shape[0], axis=0), W // biomes.shape[1], axis=1)
    if heightmap.shape != (H, W):
        heightmap = np.repeat(np.repeat(heightmap, H // heightmap.shape[0], axis=0), W // heightmap.shape[1], axis=1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Offset to center around spawn (x=0, z=0)
    # The matrix (0 to 511) maps to (-256 to 255)
    offset_x = - (H // 2)
    offset_z = - (W // 2)

    # We process by region. 
    # A region is 32x32 chunks = 512x512 blocks.
    # Group coordinates by region
    regions_data = {} # (rx, rz) -> dict of chunks

    print("Grouping blocks into regions and chunks...")
    for mx in range(H):
        for mz in range(W):
            global_x = mx + offset_x
            global_z = mz + offset_z

            cx = global_x // 16
            cz = global_z // 16
            rx = cx // 32
            rz = cz // 32

            if (rx, rz) not in regions_data:
                regions_data[(rx, rz)] = {}
            if (cx, cz) not in regions_data[(rx, rz)]:
                regions_data[(rx, rz)][(cx, cz)] = []

            regions_data[(rx, rz)][(cx, cz)].append((mx, mz, global_x, global_z))

    SURFACE_BLOCK_MAPPING = {
        0: "air", 1: "sand", 2: "sandstone", 3: "stone", 4: "grass_block",
        5: "water", 6: "dirt", 7: "granite", 8: "gravel", 9: "diorite",
        10: "iron_ore", 11: "coal_ore", 12: "lava", 13: "andesite",
        14: "clay", 15: "coarse_dirt", 16: "snow", 17: "ice",
        18: "packed_ice", 19: "snow_block", 20: "blue_ice",
        21: "bubble_column", 22: "podzol", 23: "gold_ore",
        24: "diamond_ore", 25: "redstone_ore", 26: "emerald_ore"
    }

    # Helper function to get anvil block
    def get_anvil_block(name_str):
        if isinstance(name_str, (int, np.integer)):
            name_str = SURFACE_BLOCK_MAPPING.get(int(name_str), "dirt")
            
        properties = {}
        if isinstance(name_str, str):
            if '[' in name_str and name_str.endswith(']'):
                base, props_str = name_str[:-1].split('[')
                name_str = base
                for prop in props_str.split(','):
                    k, v = prop.split('=')
                    properties[k] = v
            if name_str.startswith('minecraft:'):
                name_str = name_str.split(':')[1]
                
        return anvil.Block('minecraft', name_str, properties)

    bedrock_block = get_anvil_block('bedrock')
    stone_block = get_anvil_block('stone')
    dirt_block = get_anvil_block('dirt')
    sand_block = get_anvil_block('sand')
    water_block = get_anvil_block('water')
    clay_block = get_anvil_block('clay')

    print("Generating MCA regions...")
    for (rx, rz), chunks_dict in regions_data.items():
        print(f"Creating region {rx}, {rz}...")
        region = anvil.EmptyRegion(rx, rz)
        
        for (cx, cz), columns in chunks_dict.items():
            chunk = BiomeEmptyChunk(cx, cz)
            
            for (mx, mz, gx, gz) in columns:
                local_x = gx % 16
                local_z = gz % 16
                
                h = int(heightmap[mx, mz])
                if h < 0: h = 0
                if h > 255: h = 255
                
                biome_id_raw = biomes[mx, mz]
                biome = inv_biomes_mapping.get(int(biome_id_raw), "minecraft:the_void")
                
                # Set biome in the chunk
                leg_biome_id = BIOME_NAME_TO_ID.get(biome, 127)
                for by in range(64):
                    chunk.set_biome(local_x, by * 4, local_z, leg_biome_id)
                
                # y = 0: Bedrock
                chunk.set_block(bedrock_block, local_x, 0, local_z)
                
                layer_bottom = h
                
                # Sub-surface layers
                if biome in OCEAN_BIOMES:
                    # 10 water, then 1 sand
                    water_bottom = max(1, h - 10)
                    for y in range(water_bottom, h):
                        chunk.set_block(water_block, local_x, y, local_z)
                    if water_bottom - 1 >= 1:
                        chunk.set_block(sand_block, local_x, water_bottom - 1, local_z)
                        layer_bottom = water_bottom - 1
                    else:
                        layer_bottom = water_bottom
                elif biome in RIVER_BIOMES:
                    # Clay only if the surface block is water
                    surface_b = "minecraft:air"
                    if h > 0:
                        surface_b = world_3d[h, mx, mz]
                        if isinstance(surface_b, (int, np.integer)):
                            surface_b = SURFACE_BLOCK_MAPPING.get(int(surface_b), "dirt")
                            
                    if surface_b == "minecraft:water" or surface_b == 5:
                        if h - 1 >= 1:
                            chunk.set_block(clay_block, local_x, h - 1, local_z)
                            layer_bottom = h - 1
                    else:
                        dirt_bottom = max(1, h - 3)
                        for y in range(dirt_bottom, h):
                            chunk.set_block(dirt_block, local_x, y, local_z)
                        layer_bottom = dirt_bottom
                elif biome in DESERT_BIOMES:
                    # 3 sand
                    sand_bottom = max(1, h - 3)
                    for y in range(sand_bottom, h):
                        chunk.set_block(sand_block, local_x, y, local_z)
                    layer_bottom = sand_bottom
                else:
                    # 3 dirt
                    dirt_bottom = max(1, h - 3)
                    for y in range(dirt_bottom, h):
                        chunk.set_block(dirt_block, local_x, y, local_z)
                    layer_bottom = dirt_bottom
                
                # Stone
                for y in range(1, max(1, layer_bottom)):
                    chunk.set_block(stone_block, local_x, y, local_z)
                    
                # Surface block (can be air or proper block)
                if h > 0:
                    surface_b = world_3d[h, mx, mz]
                    if surface_b != "minecraft:air":
                        chunk.set_block(get_anvil_block(surface_b), local_x, h, local_z)
                
                # Vegetation and top layers
                for y in range(h + 1, 256):
                    veg_b = world_3d[y, mx, mz]
                    if veg_b != "minecraft:air" and isinstance(veg_b, str):
                        chunk.set_block(get_anvil_block(veg_b), local_x, y, local_z)
                        
            region.add_chunk(chunk)
            
        out_file = out_dir / f"r.{rx}.{rz}.mca"
        print(f"Saving {out_file}...")
        region.save(str(out_file))

    print("All regions saved successfully!")

if __name__ == "__main__":
    main()
