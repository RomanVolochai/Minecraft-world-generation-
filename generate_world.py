import argparse
import subprocess
import sys
import os
from pathlib import Path
import shutil

def run_command(cmd_args):
    cmd_str = " ".join(cmd_args)
    print(f"\n>>> Running: {cmd_str}")
    
    env = os.environ.copy()
    current_dir = str(Path.cwd().resolve())
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = current_dir + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = current_dir
        
    result = subprocess.run(cmd_args, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Automated Minecraft World Generation Pipeline")
    parser.add_argument("name", type=str, nargs="?", default="generated", help="Base name for the generated files (default: 'generated')")
    args = parser.parse_args()

    name = args.name
    print(f"=== Starting generation pipeline for world name: {name} ===")

    # Ensure output directories exist
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # 1. Surface Generation: GMM + MRF (ensemble)
    print("\n--- STEP 1: Surface Block Generation (GMM + MRF) ---")
    run_command([
        "venv/bin/python", "surface/generate.py",
        "--model", "ensemble",
        "--size", "512",
        "--load_model", "checkpoints/surface_ensemble",
        "--output", f"outputs/{name}.png"
    ])

    # 2. Biomes Prediction & Smoothing
    # 2a. Random Forest Predictor + MRF Smoothing
    print("\n--- STEP 2a: Biomes RF Prediction & MRF Smoothing ---")
    run_command([
        "venv/bin/python", "biomes/metrics/smooth_prediction.py",
        "--load_model", "checkpoints/biomes_rf_512_cell8_50w.joblib",
        "--input_surface", f"outputs/{name}.npy",
        "--cell_size", "8",
        "--crop_size", "512",
        "--crop_x", "0",
        "--crop_y", "0",
        "--min_region_size", "30",
        "--output", f"outputs/biomes_rf_mrf_{name}_crop_0_0.png",
        "--npy_output", f"outputs/biomes_rf_mrf_{name}_crop_0_0.npy"
    ])

    # 2b. Subtype assignment using frequencies from all 400 worlds
    print("\n--- STEP 2b: Biomes Subtype Assignment ---")
    run_command([
        "venv/bin/python", "biomes/subtypes/assign.py",
        "--input", f"outputs/biomes_rf_mrf_{name}_crop_0_0.npy",
        "--dataset", "dataset",
        "--output", f"outputs/biomes_rf_mrf_{name}_crop_0_0_subtypes.png",
        "--npy_output", f"outputs/biomes_rf_mrf_{name}_crop_0_0_subtypes.npy"
    ])

    # 3. Heightmap Prediction: UNet with checkpoints/unet_height10.pth
    print("\n--- STEP 3: Heightmap Prediction (U-Net) ---")
    run_command([
        "venv/bin/python", "src/generate.py",
        "--model", "unet_height",
        "--load_model", "checkpoints/unet_height10.pth",
        "--input_surface", f"outputs/{name}.npy",
        "--input_biomes", f"outputs/biomes_rf_mrf_{name}_crop_0_0_subtypes.npy",
        "--output", f"outputs/unet_{name}.png"
    ])

    # 4. Vegetation Placement (saplings, plants, etc.)
    print("\n--- STEP 4: Vegetation Generator ---")
    run_command([
        "venv/bin/python", "vegetation/vegetation_generator.py",
        "--surface", f"outputs/{name}.npy",
        "--biomes", f"outputs/biomes_rf_mrf_{name}_crop_0_0_subtypes.npy",
        "--heightmap", f"outputs/unet_{name}_heightmap_pred.npy",
        "--histogram", "vegetation/vegetation_histogram.json",
        "--biomes_mapping", "dataset/biomes_mapping.json",
        "--output", f"outputs/world_3d_{name}.npy"
    ])

    # 5. MCA Generation (Regions)
    print("\n--- STEP 5: MCA Region File Generation ---")
    mca_output_dir = Path("mca-generate") / f"regions_{name}"
    run_command([
        "venv/bin/python", "mca-generate/mca_generator.py",
        "--world3d", f"outputs/world_3d_{name}.npy",
        "--biomes", f"outputs/biomes_rf_mrf_{name}_crop_0_0_subtypes.npy",
        "--heightmap", f"outputs/unet_{name}_heightmap_pred.npy",
        "--biomes_mapping", "dataset/biomes_mapping.json",
        "--output_dir", str(mca_output_dir)
    ])

    # 6. Organize and collect outputs in the 'worlds' folder
    print("\n--- STEP 6: Collecting files into 'worlds' folder ---")
    world_dir = Path("worlds") / name
    img_dir = world_dir / "visualizations"
    data_dir = world_dir / "matrices"
    reg_dir = world_dir / "regions"

    # Create target directories
    img_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    reg_dir.mkdir(parents=True, exist_ok=True)

    # Move images (PNGs)
    png_files = [
        outputs_dir / f"{name}.png",
        outputs_dir / f"biomes_rf_mrf_{name}_crop_0_0.png",
        outputs_dir / f"biomes_rf_mrf_{name}_crop_0_0_subtypes.png",
        outputs_dir / f"unet_{name}_surface.png",
        outputs_dir / f"unet_{name}_heightmap_pred.png",
    ]
    for pf in png_files:
        if pf.exists():
            shutil.copy(pf, img_dir / pf.name)
            pf.unlink()

    # Move matrices (NPYs)
    npy_files = [
        outputs_dir / f"{name}.npy",
        outputs_dir / f"biomes_rf_mrf_{name}_crop_0_0.npy",
        outputs_dir / f"biomes_rf_mrf_{name}_crop_0_0_subtypes.npy",
        outputs_dir / f"unet_{name}_heightmap_pred.npy",
        outputs_dir / f"world_3d_{name}.npy",
    ]
    for nf in npy_files:
        if nf.exists():
            shutil.copy(nf, data_dir / nf.name)
            nf.unlink()

    # Copy MCA files
    if mca_output_dir.exists():
        for mca_file in mca_output_dir.glob("*.mca"):
            shutil.copy(mca_file, reg_dir / mca_file.name)
        # Clean up temporary regions folder
        shutil.rmtree(mca_output_dir)

    print(f"\n==============================================")
    print(f"World '{name}' successfully generated!")
    print(f"Saved outputs:")
    print(f"  Images:      {img_dir}")
    print(f"  Matrices:    {data_dir}")
    print(f"  MCA Regions: {reg_dir}")
    print(f"==============================================")

if __name__ == "__main__":
    main()
