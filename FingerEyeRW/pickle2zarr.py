import pickle
from pathlib import Path
from typing import List

import numpy as np
import zarr
from tqdm.auto import tqdm
import argparse

def split_tag_pose_sequence(all_tag_pose):
    """
    all_tag_pose: list of dict, each dict maps tag name -> 4x4 matrix
    return: dict mapping tag name -> (n,4,4) numpy array
    """
    n = len(all_tag_pose)

    keys = all_tag_pose[0].keys()
    out = {k: np.zeros((n, 4, 4), dtype=float) for k in keys}

    for i, pose_dict in enumerate(all_tag_pose):
        for k in keys:
            out[k][i] = pose_dict[k]

    return out

def relative_transform_sequence_np(T):
    """
    T: (n, 4, 4)
    return: (n, 3, 4)
    """
    R1 = T[0, :3, :3]   # (3,3)
    t1 = T[0, :3, 3]    # (3,)

    R_all = T[:, :3, :3]
    t_all = T[:, :3, 3]

    # R_rel = R1^T @ Ri
    R_rel = R1.T @ R_all

    # t_rel = R1^T @ (ti - t1)
    t_rel = (R1.T @ (t_all - t1).T).T  # shape (n,3)

    out = np.concatenate([R_rel, t_rel[..., None]], axis=-1)
    return out

def stack_history_custom_padding(rel_poses, window=4):
    """
    rel_poses: (n, 3, 4)
    window: produce window+1 frames.
    Padding rule: for target_idx < 0, always use index 0.
    """
    n = rel_poses.shape[0]
    pose_dim = rel_poses.shape[1:]

    out = np.zeros((n, window + 1, *pose_dim), dtype=rel_poses.dtype)

    for t in range(n):
        indices = []
        for k in range(window + 1):
            target_idx = t - (window - k)   # t-window+k
            if target_idx < 0:
                target_idx = 0
            indices.append(target_idx)

        out[t] = rel_poses[indices]

    return out.reshape(n, (window + 1) * pose_dim[0] * pose_dim[1])

def zarr_append(z: zarr.Array, start: int, data: np.ndarray):
    """Append data along axis-0 (zarr v2 compatible)."""
    end = start + data.shape[0]
    z.resize((end, *z.shape[1:]))
    z[start:end] = data

def ensure_dataset(
    root: zarr.Group,
    name: str,
    sample: np.ndarray,
    dtype,
    chunk0: int = 1,
):
    """
    Create a zarr dataset lazily using sample shape.
    Axis-0 is time dimension.
    """
    return root.create_dataset(
        name,
        shape=(0, *sample.shape[1:]),
        chunks=(chunk0, *sample.shape[1:]),
        dtype=dtype,
        compressor=None
    )

def convert_pickles_to_zarr_streaming(
    pickle_paths: List[Path],
    save_path: Path,
):
    """
    Streaming conversion: many pickle episodes -> one zarr.

    This version:
      - reads ONE pkl at a time
      - computes tag features per pkl
      - appends into zarr (no global concatenate)
    """

    save_path = Path(save_path)
    if save_path.exists():
        raise RuntimeError(f"Zarr path already exists: {save_path}")

    root = zarr.open(str(save_path), mode="w")

    # ------------------------------------------------------------
    # Zarr arrays (lazy init)
    # ------------------------------------------------------------
    z_rgb = None
    z_rs_rgb = None
    z_current_T = None
    z_qpos = None
    z_abs_tag_T = None
    z_stacked_tag = None
    z_actions = None
    z_target_T = None

    # meta
    rgb_camera_name_list = None
    rs_camera_name_list = None
    link_name_list = None

    episode_ends = []
    total_steps = 0

    print(f"📦 Streaming convert {len(pickle_paths)} pickles → {save_path}")

    # ============================================================
    # Loop over pickles (CRITICAL: one-by-one)
    # ============================================================
    for pkl in tqdm(pickle_paths, desc="Processing pickles"):
        with open(pkl, "rb") as f:
            data = pickle.load(f)

        # --------------------------------------------------------
        # Required fields
        # --------------------------------------------------------
        rgb = data["current_rgb_images"]            # (T, nv, 3, H, W)
        current_T = data["current_transforms"]      # (T, L, 4, 4)
        qpos = data["current_joint_values"]         # (T, J)
        T = rgb.shape[0]

        # --------------------------------------------------------
        # Meta (only once)
        # --------------------------------------------------------
        if rgb_camera_name_list is None:
            rgb_camera_name_list = np.asarray(data["camera_names"], dtype=str)
            root.create_dataset("meta/rgb_camera_name_list", data=rgb_camera_name_list)

        if "current_rs_images" in data and data["current_rs_images"] is not None:
            if rs_camera_name_list is None:
                rs_camera_name_list = np.asarray(data["realsense_names"], dtype=str)
                root.create_dataset("meta/rs_camera_name_list", data=rs_camera_name_list)

        if link_name_list is None:
            link_name_list = np.asarray(data["link_names"], dtype=str)
            root.create_dataset("meta/link_name_list", data=link_name_list)

        # --------------------------------------------------------
        # Tag processing (per pkl)
        # --------------------------------------------------------
        tag_dict = split_tag_pose_sequence(data["current_center_tag_T"])

        abs_tag_T = np.stack(
            [np.asarray(v) for v in tag_dict.values()],
            axis=1,
        )  # (T, n_tag, 4, 4)

        stacked_tag = np.stack(
            [
                stack_history_custom_padding(
                    relative_transform_sequence_np(v),
                    window=4,
                )
                for v in tag_dict.values()
            ],
            axis=1,
        )  # (T, n_tag, D)

        # --------------------------------------------------------
        # Optional fields
        # --------------------------------------------------------
        rs_rgb = data.get("current_rs_images", None)
        actions = data.get("target_action", None)

        # --------------------------------------------------------
        # Lazy create datasets
        # --------------------------------------------------------
        if z_rgb is None:
            z_rgb = ensure_dataset(root, "data/obs/rgb_images", rgb, rgb.dtype, chunk0=1)
            z_current_T = ensure_dataset(root, "data/obs/current_transforms", current_T, np.float32, chunk0=1)
            z_qpos = ensure_dataset(root, "data/obs/all_qpos", qpos, np.float32, chunk0=1)
            z_abs_tag_T = ensure_dataset(root, "data/obs/abs_tag_transforms", abs_tag_T, np.float32, chunk0=1)
            z_stacked_tag = ensure_dataset(root, "data/obs/tag_ori", stacked_tag, np.float32, chunk0=1)

            if rs_rgb is not None:
                z_rs_rgb = ensure_dataset(root, "data/obs/rs_rgb_images", rs_rgb, rs_rgb.dtype, chunk0=1)
            if actions is not None:
                z_actions = ensure_dataset(root, "data/actions/original_actions", actions, np.float32, chunk0=1)
        # --------------------------------------------------------
        # Append
        # --------------------------------------------------------
        start = total_steps

        zarr_append(z_rgb, start, rgb)
        zarr_append(z_current_T, start, current_T.astype(np.float32))
        zarr_append(z_qpos, start, qpos.astype(np.float32))
        zarr_append(z_abs_tag_T, start, abs_tag_T.astype(np.float32))
        zarr_append(z_stacked_tag, start, stacked_tag.astype(np.float32))

        if rs_rgb is not None:
            zarr_append(z_rs_rgb, start, rs_rgb)
        if actions is not None:
            zarr_append(z_actions, start, actions.astype(np.float32))

        total_steps += T
        episode_ends.append(total_steps)

    # ------------------------------------------------------------
    # Episode ends
    # ------------------------------------------------------------
    root.create_dataset(
        "meta/episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
    )

    print(f"✅ Done. {total_steps} steps, {len(episode_ends)} episodes.")
    print(f"📁 Zarr saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert pickle episodes to a streaming zarr dataset."
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Directory containing pickle files (*.pkl)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output zarr directory path"
    )

    args = parser.parse_args()

    pickles_dir = list(Path(args.input).glob("*.pkl"))
    if len(pickles_dir) == 0:
        raise RuntimeError(f"No pickle files found in {args.input}")

    zarr_path = Path(args.output)

    convert_pickles_to_zarr_streaming(
        pickles_dir,
        zarr_path
    )

    print("Done converting pickles to zarr streaming.")
