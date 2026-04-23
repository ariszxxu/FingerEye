import time
import pickle
from pathlib import Path

import numpy as np
import zarr
import viser
import argparse

def visualize_a_pickle_data(pickle_path: Path):
    # ---------------- load data ----------------
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    current_rgb_images = data.get("current_rgb_images", [])   # (T,nv,3,H,W)
    current_rs_images = data.get("current_rs_images", [])   # (T,nrs,3,H,W)
    current_joint_values = data.get("current_joint_values", [])  # (T,J)
    target_action = data.get("target_action", [])          # (T,L,4,4)
    
    camera_names = data.get("camera_names", [])  # (nv,)
    realsense_names = data.get("realsense_names", ['d435i'])  # (nrs,)
    link_names = data.get("link_names", [])  # (L,)

    T = current_rgb_images.shape[0]

    # ---------------- create viser server ----------------
    server = viser.ViserServer()

    # Slider for timesteps
    step_slider = server.gui.add_slider(
        "Step",
        min=0,
        max=T - 1,
        step=1,
        initial_value=0,
    )

    # Setup image GUI placeholders
    img_gui_handles = {}

    with server.gui.add_folder("📷 Cameras"):
        for cam in camera_names:
            init_img = np.zeros_like(current_rgb_images[0, 0].transpose(1, 2, 0))
            img_gui_handles[cam] = server.gui.add_image(
                init_img,
                label=f"{cam} image"
            )

    with server.gui.add_folder("🎥 Realsense RGB"):
        for dev in realsense_names:
            init_img = np.zeros_like(current_rs_images[0, 0].transpose(1, 2, 0))
            img_gui_handles[f"rs_rgb_{dev}"] = server.gui.add_image(
                init_img,
                label=f"{dev} RGB"
            )

    # ---------------- update callback ----------------
    @step_slider.on_update
    def update_step(_):
        idx = int(step_slider.value)

        # Update camera images
        for cam_idx, cam in enumerate(camera_names):
            frame_img = current_rgb_images[idx, cam_idx].transpose(1, 2, 0)
            img_gui_handles[cam].image = frame_img

        # Update realsense RGB
        for cam_idx, dev in enumerate(realsense_names):
            frame_img = current_rs_images[idx, cam_idx].transpose(1, 2, 0)
            img_gui_handles[f"rs_rgb_{dev}"].image = frame_img  

        print(f"✅ Updated step {idx}", "action:", target_action[idx] if len(target_action) > 0 else None, flush=True)

    print(f"✅ Viser launched with {T} frames")
    return server

def visualize_a_zarr_data_streaming(zarr_path: Path):
    # ---------------- load zarr (NO full read) ----------------
    root = zarr.open(str(zarr_path), mode="r")

    rgb_z = root["data/obs/rgb_images"]            # (T, n_rgb, 3, H, W)

    rs_rgb_z = root["data/obs/rs_rgb_images"] if "data/obs/rs_rgb_images" in root else None

    rgb_camera_names = root["meta/rgb_camera_name_list"][:].tolist()
    rs_camera_names = root["meta/rs_camera_name_list"][:].tolist() if "meta/rs_camera_name_list" in root else []
    episode_ends = root["meta/episode_ends"][:]

    T = rgb_z.shape[0]
    print(f"Loaded Zarr with {T} steps, {len(episode_ends)} episodes")

    # ---------------- create viser server ----------------
    server = viser.ViserServer()

    step_slider = server.gui.add_slider(
        "Step",
        min=0,
        max=T - 1,
        step=1,
        initial_value=0,
    )

    img_gui_handles = {}

    # ---------------- GUI setup ----------------
    with server.gui.add_folder("📷 RGB Cameras"):
        for ci, cam in enumerate(rgb_camera_names):
            dummy = np.zeros((rgb_z.shape[-2], rgb_z.shape[-1], 3), dtype=np.uint8)
            img_gui_handles[f"rgb_{cam}"] = server.gui.add_image(dummy, label=cam)

    if rs_rgb_z is not None:
        with server.gui.add_folder("🎥 Realsense RGB"):
            for ci, cam in enumerate(rs_camera_names):
                dummy = np.zeros((rs_rgb_z.shape[-2], rs_rgb_z.shape[-1], 3), dtype=np.uint8)
                img_gui_handles[f"rs_rgb_{cam}"] = server.gui.add_image(dummy, label=cam)


    # ---------------- update callback (STREAMING READ) ----------------
    @step_slider.on_update
    def update_step(_):
        idx = int(step_slider.value)

        # ---------- RGB ----------
        rgb_frame = rgb_z[idx]  # (n_rgb, 3, H, W)
        for ci, cam in enumerate(rgb_camera_names):
            img = rgb_frame[ci].transpose(1, 2, 0)
            img_gui_handles[f"rgb_{cam}"].image = img

        # ---------- RS RGB ----------
        if rs_rgb_z is not None:
            rs_frame = rs_rgb_z[idx]
            for ci, cam in enumerate(rs_camera_names):
                img = rs_frame[ci].transpose(1, 2, 0)
                img_gui_handles[f"rs_rgb_{cam}"].image = img

        print(f"✅ Updated step {idx}")

    print("✅ Viser launched (streaming mode)")
    return server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize pickle or zarr dataset using Viser."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pickle", "zarr"],
        help="Choose visualization mode: pickle or zarr",
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to pickle file or zarr directory",
    )

    args = parser.parse_args()

    data_path = Path(args.path)

    if not data_path.exists():
        raise RuntimeError(f"Path does not exist: {data_path}")

    # ---------------- Mode Selection ----------------
    if args.mode == "pickle":
        if data_path.suffix != ".pkl":
            raise RuntimeError("For pickle mode, path must be a .pkl file")
        server = visualize_a_pickle_data(data_path)

    elif args.mode == "zarr":
        server = visualize_a_zarr_data_streaming(data_path)

    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")

    # Keep server alive
    while True:
        time.sleep(1)