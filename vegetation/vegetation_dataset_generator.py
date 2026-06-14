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


NUM_WORLDS = 20
RADIUS = 512

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SERVER_DIR = PROJECT_ROOT / "mc_server"

VEGETATION_MAPPING_PATH = SCRIPT_DIR / "vegetation_mapping.json"
SURFACE_BLACKLIST_CONFIG_PATH = PROJECT_ROOT / "dataset_generator" / "surface_block_blacklist.json"
BIOME_ID_CONFIG_PATH = PROJECT_ROOT / "dataset_generator" / "biomes_1_16.json"
HISTOGRAM_OUT_PATH = SCRIPT_DIR / "vegetation_histogram.json"


def load_json(path, default):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_biome_id_mapping():
    raw_mapping = load_json(BIOME_ID_CONFIG_PATH, {})
    return {int(key): value for key, value in raw_mapping.items()}


MC_1_16_BIOME_ID_TO_NAME = load_biome_id_mapping()


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


def get_block_name(block):
    if hasattr(block, "name"):
        return block.name()

    block_id = str(block.id)
    return block_id if ":" in block_id else f"minecraft:{block_id}"


def load_modified_blacklist():
    raw_blacklist = load_json(SURFACE_BLACKLIST_CONFIG_PATH, {})
    blacklist = set()

    if isinstance(raw_blacklist, list):
        blacklist.update(raw_blacklist)
    else:
        for blocks in raw_blacklist.values():
            blacklist.update(blocks)

    veg_mapping = load_json(VEGETATION_MAPPING_PATH, {})
    generatable = set(veg_mapping.get("generatable_blocks", {}).keys())

    to_remove = set()
    for block in blacklist:
        if block in generatable:
            to_remove.add(block)
        elif "log" in block or "wood" in block:
            to_remove.add(block)

    return blacklist - to_remove


SURFACE_BLOCK_BLACKLIST = load_modified_blacklist()


def is_blacklisted_surface_block(block):
    return get_block_name(block) in SURFACE_BLOCK_BLACKLIST


def load_tracked_blocks():
    veg_mapping = load_json(VEGETATION_MAPPING_PATH, {})
    generatable = set(veg_mapping.get("generatable_blocks", {}).keys())
    logs = set(veg_mapping.get("tree_log_to_sapling", {}).keys())
    return generatable | logs


TRACKED_BLOCKS = load_tracked_blocks()
histogram = load_json(HISTOGRAM_OUT_PATH, {})


def save_data(hist):
    tmp_path = HISTOGRAM_OUT_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(hist, f, indent=4, ensure_ascii=False)
    tmp_path.replace(HISTOGRAM_OUT_PATH)


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


def parse_world_and_update_histogram(world_idx, seed):
    global histogram

    region_folder = SERVER_DIR / "world" / "region"

    print(f"\n[Parser] Starting extracting data for world number {world_idx}.")
    start_parse_time = time.time()

    regions_cache = {}
    parsed_columns = 0

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

                        if biome_str not in histogram:
                            histogram[biome_str] = {}
                        histogram[biome_str]["_total_area"] = histogram[biome_str].get("_total_area", 0) + 1

                        if block_name in TRACKED_BLOCKS:
                            if block_name not in histogram[biome_str]:
                                histogram[biome_str][block_name] = 0
                            
                            histogram[biome_str][block_name] += 1
                        
                        parsed_columns += 1
                        break

    if parsed_columns == 0:
        print("[Parser] WARNING: no surface columns were parsed. Check region files and chunk format.")

    duration = int(time.time() - start_parse_time)
    save_data(histogram)
    print(f"[Parser] Histogram updated. Time: {duration}s")


def get_last_world_index():
    return histogram.get("_worlds_processed", 0)


def process_world(world_idx, seed, total_label):
    world_path = SERVER_DIR / "world"

    print(f"\n=== World number {world_idx}{total_label} (SEED: {seed}) ===")

    change_seed_in_properties(seed)

    if world_path.exists():
        shutil.rmtree(world_path)

    gen_start = time.time()
    run_server_and_generate()
    gen_duration = int(time.time() - gen_start)

    parse_world_and_update_histogram(world_idx, seed)

    shutil.rmtree(world_path, ignore_errors=True)

    histogram["_worlds_processed"] = world_idx
    save_data(histogram)
    print(f"[Python] World number {world_idx} processed. Generation duration: {gen_duration}s")


def main():
    global_start_time = time.time()
    start_world_idx = get_last_world_index() + 1
    end_world_idx = start_world_idx + NUM_WORLDS - 1

    print(f"Vegetation histogram generation started (adding {NUM_WORLDS} worlds: {start_world_idx}..{end_world_idx}).")

    for offset, world_idx in enumerate(range(start_world_idx, end_world_idx + 1), start=1):
        seed = random.randint(1000000000, 9999999999)
        process_world(
            world_idx,
            seed,
            total_label=f" [{offset}/{NUM_WORLDS}]"
        )

    total_duration = int(time.time() - global_start_time)
    print(f"\nGeneration ended. Total time: {total_duration // 60} minutes.")


if __name__ == "__main__":
    main()
