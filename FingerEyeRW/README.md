# FingerEye Real-World System

**Continuous and Unified Vision–Tactile Sensing for Dexterous Manipulation (Real-World Setup)**

## 📚 Repository Structure

This folder is one component of the main **FingerEye** monorepo.

- FingerEye Policy: `../FingerEyePolicy/`
- FingerEye Real-World (current): `./`
- FingerEye Lab: `../FingerEyeLab/`

This repository contains the real-world system for FingerEye, including:

* Sensor setup
* Teleoperation and data collection
* Dataset conversion and visualization
* Policy deployment on hardware
* Delicate grasp execution

> **Scope of this repository**
> - Sensor setup and device configuration
> - Demonstration data collection
> - Dataset format conversion (pickle → zarr)
> - Real-world policy inference and execution
>
> Policy architecture and training are provided in the **FingerEye Policy repository**.

---

# 🧩 Sensor Setup & Device Identification

Before collecting data, you must properly configure all cameras.

---

## 1️⃣ Identify USB Port Address

Unplug all cameras.

Plug in **one camera at a time**, then run:

```bash
python recorder_utils/get_cam_port.py
```

This script prints the USB port address of the connected device.
Record this information for configuration.

Repeat for each camera.

---

## 2️⃣ Identify Camera and Left–Right Order

For new stereo cameras:

1. Modify the `dev_addr` field in:

```
recorder_utils/get_cam_order.py
```

2. Run:

```bash
python recorder_utils/get_cam_order.py
```

An image window showing stereo images will appear.

You can:

* Cover lenses with your finger
* Observe which side changes
* Identify left/right ordering

Then change them in the configs/base_settings.yml.

Ensure camera indices and left–right ordering are correct before proceeding.

---
## 3️⃣ Get Cameras Intrinsics
Please refer to [RobotCamCalib](https://github.com/ariszxxu/RobotCamCalib) for camera calibration.

---

# 🎥 Data Collection

Data collection is performed via teleoperation.
---

##  1️⃣Launch Teleoperation Server

In another terminal:

```bash
python teleop_server.py --config-name {task_name}
```

Example:

```bash
python teleop_server.py --config-name coin_standing
```

## 2️⃣ Start Recording

```bash
python main.py --config-name {task_name} mode=record
```

Example:

```bash
python main.py --config-name coin_standing mode=record
```

---



You can now control the robot and collect demonstrations.

Recorded files will be saved as `.pkl` in logs.

---

# 🔄 Dataset Conversion (Pickle → Zarr)

After data collection, convert the pickle logs to a zarr dataset for training.

```bash
python pickle2zarr.py \
    -I {path_to_your_pickle_dir} \
    -o {path_to_your_zarr}
```

Example:

```bash
python pickle2zarr.py \
    -i /home/ps/ConTacRW/logs \
    -o /home/ps/ConTacRW/data/60_real_0127.zarr
```

The resulting `.zarr` dataset can then be used in the **FingerEye Policy repository** for training.

---

# 👀 Dataset Visualization

You can visualize both raw pickle logs and converted zarr datasets.

---

## Visualize Zarr Dataset

```bash
python data_vis.py \
    --mode zarr \
    --path {path_to_your_zarr}
```

Example:

```bash
python data_vis.py \
    --mode zarr \
    --path /home/ps/ConTacRW/data/60_real_0127.zarr
```

---

## Visualize Pickle Logs

```bash
python visualize.py \
    --mode pickle \
    --path {path_to_your_pickle}
```

Example:

```bash
python visualize.py \
    --mode pickle \
    --path /home/ps/ConTacRW/logs/recording_20260127_160722.pkl
```

Visualization helps verify:

* Camera synchronization
* Action correctness
* Data integrity

---

# 🚀 Policy Deployment

After training in the **FingerEye Policy repository**, deploy the trained checkpoint to the real robot.

```bash
python main.py \
    --config-name {task_name} \
    mode=policy \
    eval_ckpt_path={ckpt_path}
```

The system will:

* Load the trained policy
* Start real-time inference
* Execute control commands on hardware

Make sure:

* Sensor setup is correct
* Device addresses match configuration
* Robot is in a safe initial pose

---

# 🤏 Delicate Grasp
```bash
python delicate_grasp.py task={task_name}
```
