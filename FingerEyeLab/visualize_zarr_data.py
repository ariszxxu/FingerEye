import argparse
import time
from pathlib import Path
import zarr
import viser

def visualize_simple_obs_images(zarr_dir: Path):
    """
    Simplified streaming visualization for image observations.
    """
    zarr_dir = Path(zarr_dir)
    if not zarr_dir.exists():
        raise FileNotFoundError(f"❌ Zarr file not found: {zarr_dir}")

    # ---------------- load zarr (NO full read) ----------------
    print(f"📂 Opening: {zarr_dir}")
    root = zarr.open(str(zarr_dir), mode="r")
    print(root.tree())

    T_max_limit = 100000  # Safety limit
    if "meta/episode_ends" in root:
        episode_ends = root["meta/episode_ends"][:]   
        T = min(int(episode_ends[-1]), T_max_limit)
    else:
        if "data/obs/rgb_images" not in root:
            raise KeyError("❌ zarr does not contain 'data/obs/rgb_images'")
        T = min(root["data/obs/rgb_images"].shape[0], 100)

    # ---------------- dataset handles (zarr arrays) ----------------
    if "data/obs/rgb_images" not in root:
        raise KeyError("❌ zarr does not contain 'data/obs/rgb_images'")
    rgb_z = root["data/obs/rgb_images"]          # (T_total, N_rgb, 3, H, W)
    n_rgb = rgb_z.shape[1]

    rs_z = root["data/obs/rs_rgb_images"] if "data/obs/rs_rgb_images" in root else None
    third_view_z = root["data/obs/third_view"] if "data/obs/third_view" in root else None

    if rs_z is not None:
        n_rs = rs_z.shape[1]
    else:
        n_rs = 0

    sim_rgb_z = root["data/obs/sim_rgb_images"] if "data/obs/sim_rgb_images" in root else None
    sim_rs_z = root["data/obs/sim_rs_rgb_images"] if "data/obs/sim_rs_rgb_images" in root else None

    if sim_rgb_z is not None:
        sim_n_rgb = sim_rgb_z.shape[1]
        T = min(T, sim_rgb_z.shape[0])
    else:
        sim_n_rgb = 0

    if sim_rs_z is not None:
        sim_n_rs = sim_rs_z.shape[1]
        T = min(T, sim_rs_z.shape[0])
    else:
        sim_n_rs = 0
    # -------------------------------------------------------------------

    T = min(T, rgb_z.shape[0])
    if rs_z is not None:
        T = min(T, rs_z.shape[0])
    if third_view_z is not None:
        T = min(T, third_view_z.shape[0])

    # Camera naming
    rgb_camera_names = [f"rgb_cam_{i}" for i in range(n_rgb)]
    rs_camera_names = [f"rs_cam_{i}" for i in range(n_rs)]

    sim_rgb_camera_names = [f"sim_rgb_cam_{i}" for i in range(sim_n_rgb)]
    sim_rs_camera_names = [f"sim_rs_cam_{i}" for i in range(sim_n_rs)]

    # ---------------- create viser server ----------------
    server = viser.ViserServer()

    step_slider = server.gui.add_slider("Step", min=0, max=T - 1, step=1, initial_value=0)

    img_gui_handles = {}

    def add_camera_tab(folder_name, cam_names, data_source, key_prefix):
        if data_source is None or len(cam_names) == 0:
            return
        with server.gui.add_folder(folder_name):
            for ci, cam in enumerate(cam_names):
                init_img = data_source[0, ci].transpose(1, 2, 0)

                img_gui_handles[f"{key_prefix}{cam}"] = server.gui.add_image(
                    init_img,
                    label=cam,
                )

    # 1. Third View
    if third_view_z is not None:
        with server.gui.add_folder("📷 Third View"):
            img_gui_handles["third_view"] = server.gui.add_image(
                third_view_z[0].transpose(1, 2, 0),
                label="Third View",
            )

    add_camera_tab("📷 RGB Cameras", rgb_camera_names, rgb_z, "rgb_")

    add_camera_tab("🎥 Realsense RGB", rs_camera_names, rs_z, "rs_rgb_")

    add_camera_tab("🧪 SIM RGB Cameras", sim_rgb_camera_names, sim_rgb_z, "sim_rgb_")

    add_camera_tab("🧪 SIM Realsense RGB", sim_rs_camera_names, sim_rs_z, "sim_rs_rgb_")

    # ---------------- update callback (STREAMING READ) ----------------
    def _update_frame(idx: int):
        for ci, cam in enumerate(rgb_camera_names):
            img_gui_handles[f"rgb_{cam}"].image = rgb_z[idx, ci].transpose(1, 2, 0)

        # Update Realsense
        if rs_z is not None:
            for ci, cam in enumerate(rs_camera_names):
                img_gui_handles[f"rs_rgb_{cam}"].image = rs_z[idx, ci].transpose(1, 2, 0)

        # Update SIM RGB
        if sim_rgb_z is not None:
            for ci, cam in enumerate(sim_rgb_camera_names):
                img_gui_handles[f"sim_rgb_{cam}"].image = sim_rgb_z[idx, ci].transpose(1, 2, 0)

        # Update SIM Realsense
        if sim_rs_z is not None:
            for ci, cam in enumerate(sim_rs_camera_names):
                img_gui_handles[f"sim_rs_rgb_{cam}"].image = sim_rs_z[idx, ci].transpose(1, 2, 0)

        # Update Third View
        if third_view_z is not None:
            img_gui_handles["third_view"].image = third_view_z[idx].transpose(1, 2, 0)

        if idx % 10 == 0:
            print(f"✅ Updated step {idx}")

    @step_slider.on_update
    def _on_step_update(_):
        idx = int(step_slider.value)
        _update_frame(idx)

    print(f"✅ Viser launched (streaming) with T={T} frames (max limit {T_max_limit})")
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Zarr image observations")
    parser.add_argument(
        "--path", 
        type=str, 
        default="/home/ps/projects/fingereye/fingereye/data/30_coin.zarr",
        help="Path to the .zarr directory"
    )
    args = parser.parse_args()

    server = visualize_simple_obs_images(Path(args.path))
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")