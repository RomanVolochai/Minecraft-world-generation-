import json
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

import anvil
import numpy as np


NUM_WORLDS = 1000
RADIUS = 512

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
WORLDS_DATA_DIR = DATASET_DIR / "real_worlds_data"
SERVER_DIR = PROJECT_ROOT / "mc_server"

DATASET_DIR.mkdir(exist_ok=True)
WORLDS_DATA_DIR.mkdir(exist_ok=True, parents=True)

BIOMES_JSON_PATH = DATASET_DIR / "biomes_mapping.json"
BLOCKS_JSON_PATH = DATASET_DIR / "blocks_mapping.json"
BIOME_ID_CONFIG_PATH = SCRIPT_DIR / "biomes_1_16.json"
SURFACE_BLACKLIST_CONFIG_PATH = SCRIPT_DIR / "surface_block_blacklist.json"


def load_json(path, default):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_biome_id_mapping():
    raw_mapping = load_json(BIOME_ID_CONFIG_PATH, {})
    return {int(key): value for key, value in raw_mapping.items()}


def load_surface_block_blacklist():
    raw_blacklist = load_json(SURFACE_BLACKLIST_CONFIG_PATH, {})

    if isinstance(raw_blacklist, list):
        return set(raw_blacklist)

    blacklist = set()
    for blocks in raw_blacklist.values():
        blacklist.update(blocks)

    return blacklist


def save_json_atomic(path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    tmp_path.replace(path)


def save_mappings():
    save_json_atomic(BIOMES_JSON_PATH, biome_to_id)
    save_json_atomic(BLOCKS_JSON_PATH, block_to_id)


MC_1_16_BIOME_ID_TO_NAME = load_biome_id_mapping()
SURFACE_BLOCK_BLACKLIST = load_surface_block_blacklist()

biome_to_id = load_json(BIOMES_JSON_PATH, {"minecraft:air": 0})
block_to_id = load_json(BLOCKS_JSON_PATH, {"minecraft:air": 0})

AIR_BLOCK_IDS = {"air", "cave_air", "void_air"}
AIR_BLOCK_NAMES = {f"minecraft:{block_id}" for block_id in AIR_BLOCK_IDS}


def get_block_name(block):
    if hasattr(block, "name"):
        return block.name()

    block_id = str(block.id)
    return block_id if ":" in block_id else f"minecraft:{block_id}"


def is_air_block(block):
    return str(block.id) in AIR_BLOCK_IDS or get_block_name(block) in AIR_BLOCK_NAMES


def is_blacklisted_surface_block(block):
    return get_block_name(block) in SURFACE_BLOCK_BLACKLIST


def normalize_biome_name(biome_val):
    if hasattr(biome_val, "value"):
        biome_val = biome_val.value

    if isinstance(biome_val, str):
        if ":" in biome_val:
            return biome_val
        if biome_val.isdigit():
            return MC_1_16_BIOME_ID_TO_NAME.get(int(biome_val), biome_val)
        return biome_val

    return MC_1_16_BIOME_ID_TO_NAME.get(int(biome_val), str(biome_val))


def get_biome_name(chunk, x, y, z):
    if "Biomes" not in chunk.data:
        return "minecraft:plains"

    biomes_tag = chunk.data["Biomes"]
    if len(biomes_tag) == 1024:
        biome_val = biomes_tag[(y // 4) * 16 + (z // 4) * 4 + (x // 4)]
    else:
        biome_val = biomes_tag[z * 16 + x]

    return normalize_biome_name(biome_val)


def unpack_heightmap(heightmap_values):
    heights = np.zeros(256, dtype=np.int16)
    bits_per_value = 9
    values_per_long = 64 // bits_per_value
    mask = (1 << bits_per_value) - 1

    for index in range(256):
        long_index = index // values_per_long
        bit_offset = (index % values_per_long) * bits_per_value
        value = int(heightmap_values[long_index])

        if value < 0:
            value += 1 << 64

        heights[index] = (value >> bit_offset) & mask

    return heights.reshape((16, 16)).T


def get_chunk_surface_heightmap(chunk):
    try:
        heightmaps = chunk.data["Heightmaps"]
    except KeyError:
        return None

    for key in ("WORLD_SURFACE", "MOTION_BLOCKING", "OCEAN_FLOOR"):
        if key in heightmaps:
            return unpack_heightmap(heightmaps[key].value)

    return None


def normalize_biome_mapping(mapping):
    normalized = {}

    for biome_name in mapping:
        normalized_name = normalize_biome_name(biome_name)
        if normalized_name not in normalized:
            normalized[normalized_name] = len(normalized)

    return normalized


biome_to_id = normalize_biome_mapping(biome_to_id)


def get_last_world_index():
    last_index = 0

    for path in WORLDS_DATA_DIR.glob("world_sample_*_seed_*.npz"):
        match = re.match(r"world_sample_(\d+)_seed_\d+\.npz$", path.name)
        if match:
            last_index = max(last_index, int(match.group(1)))

    return last_index


def change_seed_in_properties(new_seed):
    properties_path = SERVER_DIR / "server.properties"

    with properties_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    saw_generate_structures = False

    with properties_path.open("w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("level-seed="):
                f.write(f"level-seed={new_seed}\n")
            elif line.startswith("generate-structures="):
                f.write("generate-structures=false\n")
                saw_generate_structures = True
            else:
                f.write(line)

        if not saw_generate_structures:
            f.write("generate-structures=false\n")


def run_server_and_generate():
    cmd = [
        "java",
        "-Xms2G",
        "-Xmx4G",
        "-DPaper.IgnoreJavaVersion=true",
        "-jar",
        "server.jar",
        "nogui",
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        cwd=SERVER_DIR,
    )

    chunky_started = False
    chunky_confirmed = False

    for line in iter(process.stdout.readline, ""):
        print(line.strip())

        if "Done (" in line and not chunky_started:
            print("\n[Python] Chunky settings configuration")
            process.stdin.write("chunky center 0 0\n")
            process.stdin.write(f"chunky radius {RADIUS}\n")
            process.stdin.write("chunky start\n")
            process.stdin.flush()
            chunky_started = True

        if "[Chunky] A task was already started for this world." in line and not chunky_confirmed:
            print("\n[Python] Confirming Chunky task restart.")
            process.stdin.write("chunky confirm\n")
            process.stdin.flush()
            chunky_confirmed = True

        if "[Chunky] Task finished for world" in line:
            print("\n[Python] World generation finished, stopping the server.")
            process.stdin.write("stop\n")
            process.stdin.flush()
            break

    process.wait()


def parse_world_to_2d(world_idx, seed, save_npz=True):
    global biome_to_id, block_to_id

    region_folder = SERVER_DIR / "world" / "region"

    print(f"\n[Parser] Starting extracting data for world number {world_idx}.")
    start_parse_time = time.time()

    heightmap = np.zeros((1024, 1024), dtype=np.int16)
    biome_map = np.zeros((1024, 1024), dtype=np.int16)
    surface_map = np.zeros((1024, 1024), dtype=np.int16)

    regions_cache = {}
    parsed_columns = 0

    # Chunk coordinates -32..31 cover 64x64 chunks, i.e. 1024x1024 blocks.
    for cx in range(-32, 32):
        for cz in range(-32, 32):
            rx = cx // 32
            rz = cz // 32
            region_file = f"r.{rx}.{rz}.mca"
            region_path = region_folder / region_file

            if not region_path.exists():
                continue

            if region_file not in regions_cache:
                regions_cache[region_file] = anvil.Region.from_file(str(region_path))

            region = regions_cache[region_file]

            rcx = cx % 32
            rcz = cz % 32
            try:
                chunk = region.get_chunk(rcx, rcz)
            except Exception:
                continue

            chunk_surface_heights = get_chunk_surface_heightmap(chunk)

            for x in range(16):
                for z in range(16):
                    ax = (cx + 32) * 16 + x
                    az = (cz + 32) * 16 + z

                    if chunk_surface_heights is None:
                        y_candidates = range(255, -1, -1)
                    else:
                        surface_y = int(chunk_surface_heights[x, z]) - 1
                        min_y = max(surface_y - 48, 0)
                        y_candidates = range(min(surface_y, 255), min_y - 1, -1)

                    for y in y_candidates:
                        try:
                            block = chunk.get_block(x, y, z)
                        except Exception:
                            continue

                        if is_blacklisted_surface_block(block):
                            continue

                        block_name = get_block_name(block)
                        biome_str = get_biome_name(chunk, x, y, z)

                        if biome_str not in biome_to_id:
                            biome_to_id[biome_str] = len(biome_to_id)
                        if block_name not in block_to_id:
                            block_to_id[block_name] = len(block_to_id)

                        heightmap[ax, az] = y
                        biome_map[ax, az] = biome_to_id[biome_str]
                        surface_map[ax, az] = block_to_id[block_name]
                        parsed_columns += 1
                        break

    if parsed_columns == 0:
        print("[Parser] WARNING: no surface columns were parsed. Check region files and chunk format.")

    duration = int(time.time() - start_parse_time)
    if save_npz:
        out_path = WORLDS_DATA_DIR / f"world_sample_{world_idx}_seed_{seed}.npz"
        np.savez_compressed(
            out_path,
            heightmap=heightmap,
            biomes=biome_map,
            surface=surface_map,
        )
        print(f"[Parser] Matrixes saved into {out_path} ({out_path.stat().st_size // 1024} KB). Time: {duration}s")
    else:
        print(f"[Parser] Mappings updated without saving matrixes. Time: {duration}s")


def process_world(world_idx, seed, total_label, save_npz):
    world_path = SERVER_DIR / "world"

    print(f"\n=== World number {world_idx}{total_label} (SEED: {seed}) ===")

    change_seed_in_properties(seed)

    if world_path.exists():
        shutil.rmtree(world_path)

    gen_start = time.time()
    run_server_and_generate()
    gen_duration = int(time.time() - gen_start)

    parse_world_to_2d(world_idx, seed, save_npz=save_npz)
    save_mappings()

    shutil.rmtree(world_path, ignore_errors=True)
    print(f"[Python] World number {world_idx} processed. Generation duration: {gen_duration}s")


def main():
    global_start_time = time.time()
    start_world_idx = get_last_world_index() + 1
    end_world_idx = start_world_idx + NUM_WORLDS - 1

    print(f"Dataset generation started (adding {NUM_WORLDS} worlds: {start_world_idx}..{end_world_idx}).")
    save_mappings()

    for offset, world_idx in enumerate(range(start_world_idx, end_world_idx + 1), start=1):
        seed = random.randint(1000000000, 9999999999)
        process_world(
            world_idx,
            seed,
            total_label=f" [new {offset}/{NUM_WORLDS}]",
            save_npz=True,
        )

    total_duration = int(time.time() - global_start_time)
    print(f"\nDataset generation ended. Total time: {total_duration // 60} minutes.")


if __name__ == "__main__":
    main()
