import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf


def _parse_actuated_joints(urdf_path: Path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "")
        if joint_type == "fixed":
            continue

        name = joint.attrib["name"]
        limit = joint.find("limit")

        if joint_type == "continuous":
            lower, upper = -np.pi, np.pi
        elif limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        else:
            lower, upper = -1.0, 1.0

        joints.append((name, lower, upper))

    return joints


def _load_robot_with_fallback(server: viser.ViserServer, urdf_path: Path) -> ViserUrdf:
    # Some environments fail on visual mesh decoding (e.g., GLB/OBJ parser mismatch).
    # Fall back to collision-only meshes so the robot can still be inspected and controlled.
    modes = [
        {"load_meshes": True, "load_collision_meshes": False, "tag": "visual"},
        {"load_meshes": False, "load_collision_meshes": True, "tag": "collision"},
        {"load_meshes": False, "load_collision_meshes": False, "tag": "links-only"},
    ]
    errors = []
    for mode in modes:
        try:
            robot = ViserUrdf(
                server,
                urdf_or_path=urdf_path,
                load_meshes=mode["load_meshes"],
                load_collision_meshes=mode["load_collision_meshes"],
                scale=1.0,
                root_node_name="/robot_model",
            )
            print(f"URDF loaded with mode: {mode['tag']}")
            return robot
        except Exception as exc:  # noqa: BLE001
            errors.append((mode["tag"], repr(exc)))
            print(f"Load mode failed ({mode['tag']}): {exc}")

    msg = "All URDF load modes failed:\n" + "\n".join(
        f"- {tag}: {err}" for tag, err in errors
    )
    raise RuntimeError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Viser URDF viewer with joint sliders")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("/home/ps/projects/consens_release/FingerEyeLab/assets/xarm7_leap_right/xarm7_leap_right.urdf"),
        help="Path to URDF file",
    )
    parser.add_argument("--port", type=int, default=8081, help="Viser server port")
    args = parser.parse_args()

    urdf_path = args.urdf.resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    server = viser.ViserServer(port=args.port)

    robot_base = server.scene.add_frame("/robot", show_axes=True)
    robot_base.position = (0.0, 0.0, 0.0)

    robot = _load_robot_with_fallback(server, urdf_path)

    actuated_joints = _parse_actuated_joints(urdf_path)
    cfg = np.zeros(len(actuated_joints), dtype=np.float64)
    robot.update_cfg(cfg)

    with server.gui.add_folder("URDF Controls"):
        server.gui.add_markdown(f"Loaded: `{urdf_path}`")
        sliders = []

        for i, (joint_name, lower, upper) in enumerate(actuated_joints):
            if upper < lower:
                lower, upper = upper, lower

            init = 0.0
            if init < lower or init > upper:
                init = 0.5 * (lower + upper)

            cfg[i] = init

            slider = server.gui.add_slider(
                label=joint_name,
                min=lower,
                max=upper,
                step=max((upper - lower) / 1000.0, 1e-4),
                initial_value=init,
            )
            sliders.append(slider)

            @slider.on_update
            def _on_slider_update(_evt, idx=i, s=slider):
                cfg[idx] = float(s.value)
                robot.update_cfg(cfg)

        @server.gui.add_button("Reset joints to zero/mid").on_click
        def _(_evt):
            for i, (joint_name, lower, upper) in enumerate(actuated_joints):
                _ = joint_name
                val = 0.0
                if val < lower or val > upper:
                    val = 0.5 * (lower + upper)
                cfg[i] = val
                sliders[i].value = val
            robot.update_cfg(cfg)

    print(f"Viser running on port {args.port}")
    print(f"Loaded URDF: {urdf_path}")
    print(f"Actuated joints: {len(actuated_joints)}")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()