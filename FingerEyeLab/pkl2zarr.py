import pickle
import argparse  # Added argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm
from zarr_dataset import ZarrDataset

# Define default paths (used if no CLI arguments are provided)
FILE_PATH = Path(__file__).resolve()
DIR_PATH = FILE_PATH.parent
INPUT_PATH = f"{DIR_PATH}/logs"
OUTPUT_PATH = f"{DIR_PATH}/data/0123_coin_render.zarr"

def convert_pickles_to_zarr_simple(
    pickle_paths: List[Path],
    save_path: Path,
):
    """
    Converts a list of pickle files to a Zarr dataset.
    """
    assert len(pickle_paths) > 0, "pickle_paths cannot be empty"
    save_path = Path(save_path)

    all_qpos = []          # (T_i, 8)
    all_all_qpos = []      # (T_i, 47)
    all_coin_pos = []      # (T_i, 7)
    all_actions = []       # (T_i, 8)

    episode_ends = []
    total_steps = 0

    print(f"📦 Converting {len(pickle_paths)} pickle files to Zarr...")
    print(f"📂 Output path: {save_path}")

    for pkl in tqdm(pickle_paths, desc="Processing pickles"):
        with open(pkl, "rb") as f:
            data = pickle.load(f)

        # ---- Check required keys ----
        required_keys = [
            "current_joint_values",
            "current_all_joint_values",
            "coin_pose",
            "target_action",
        ]
        for k in required_keys:
            if k not in data:
                raise KeyError(f"Missing key '{k}' in {pkl}")

        # Note: Slicing logic [:-1, 0, :] is preserved from original code
        qpos = np.asarray(data["current_joint_values"], dtype=np.float32)[:-1, 0, :]
        all_q = np.asarray(data["current_all_joint_values"], dtype=np.float32)[:-1, 0, :]
        coin = np.asarray(data["coin_pose"], dtype=np.float32)[:-1, 0, :]
        act = np.asarray(data["target_action"], dtype=np.float32)[:-1]

        # ---- Shape Validation ----
        if qpos.ndim != 2 or act.ndim != 2:
            raise ValueError(f"{pkl}: qpos/action expect shape (T, J), got qpos={qpos.shape}, act={act.shape}")
        if all_q.ndim != 2:
            raise ValueError(f"{pkl}: current_all_joint_values expect shape (T, 47), got {all_q.shape}")
        if coin.ndim != 2:
            raise ValueError(f"{pkl}: coin_pose expect shape (T, 7), got {coin.shape}")

        T = qpos.shape[0]
        if not (all_q.shape[0] == T and coin.shape[0] == T and act.shape[0] == T):
            raise ValueError(
                f"{pkl}: Time step mismatch: qpos T={T}, all_q T={all_q.shape[0]}, "
                f"coin T={coin.shape[0]}, act T={act.shape[0]}"
            )
        if qpos.shape[1] != act.shape[1]:
            raise ValueError(f"{pkl}: action dim {act.shape[1]} != joint dim {qpos.shape[1]}")

        # ---- Collect ----
        all_qpos.append(qpos)
        all_all_qpos.append(all_q)
        all_coin_pos.append(coin)
        all_actions.append(act)

        total_steps += T
        episode_ends.append(total_steps)

    # ---- Concatenate ----
    print("🔗 Concatenating arrays...")
    qpos_np = np.concatenate(all_qpos, axis=0)          # (N, 8)
    all_qpos_np = np.concatenate(all_all_qpos, axis=0)  # (N, 47)
    coin_pos_np = np.concatenate(all_coin_pos, axis=0)  # (N, 7)
    actions_np = np.concatenate(all_actions, axis=0)    # (N, 8)

    # delta = target - current
    delta_actions_np = actions_np - qpos_np            # (N, 8)
    episode_ends_np = np.asarray(episode_ends, dtype=np.int64)

    # ---- Save ----
    print("💾 Saving to Zarr...")
    zarr_dataset = ZarrDataset(str(save_path))

    arrays_to_save = {
        "data/obs/qpos": qpos_np,
        "data/obs/all_qpos": all_qpos_np,
        "data/obs/coin_pose": coin_pos_np,
        "data/actions/original_actions": actions_np,
        "data/actions/delta_actions": delta_actions_np,
        "meta/episode_ends": episode_ends_np,
    }

    zarr_dataset.save_data(arrays_to_save)
    zarr_dataset.print_structure()
    print(
        f"✅ Saved Zarr dataset with {qpos_np.shape[0]} steps, "
        f"{len(episode_ends)} episodes at {save_path}"
    )


def get_args():
    parser = argparse.ArgumentParser(description="Convert Pickle logs to Zarr dataset.")
    
    parser.add_argument(
        "--input", 
        "-i", 
        type=str, 
        default=str(INPUT_PATH),
        help=f"Path to input directory containing .pkl files (Default: {INPUT_PATH})"
    )
    
    parser.add_argument(
        "--output", 
        "-o", 
        type=str, 
        default=str(OUTPUT_PATH),
        help=f"Path to output .zarr directory (Default: {OUTPUT_PATH})"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    input_dir_path = Path(args.input)
    output_zarr_path = Path(args.output)

    # Ensure output parent directory exists
    if not output_zarr_path.parent.exists():
        print(f"Creating parent directory: {output_zarr_path.parent}")
        output_zarr_path.parent.mkdir(parents=True, exist_ok=True)

    # Find pickle files
    logs_files = sorted(input_dir_path.glob("*.pkl"))
    
    if len(logs_files) == 0:
        raise RuntimeError(f"❌ No .pkl files found in {input_dir_path}")

    # Run conversion
    convert_pickles_to_zarr_simple(logs_files, output_zarr_path)