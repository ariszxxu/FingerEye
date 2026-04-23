# FingerEye Lab
**Simulation and Digital Twin for Continuous Vision-Tactile Sensing**

## 📚 Repository Structure

This folder is one component of the main **FingerEye** monorepo.

- FingerEye Policy: `../FingerEyePolicy/`
- FingerEye Real-World: `../FingerEyeRW/`
- FingerEye Lab (current): `./`

This repository contains the **Isaac Lab-based simulation and digital twin** for the FingerEye robot platform.  
It provides physics-based environments, sensor modeling, simulation data collection to support learning and evaluation for contact-rich dexterous manipulation.

> **Scope of this repository**
> - FingerEye sensor and hand digital twin in Isaac Lab  
> - Task environments and asset definitions  
> - Simulation data collection  
> - Simulation policy valuation  
>
> Policy architectures, training pipelines, and real-world deployment are provided in the other subdirectories of this monorepo.

---

## 🛠️ Installation

This repository builds on **Isaac Lab**.

### 1. Install Isaac Lab

Follow the official installation guide:  
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

We recommend using the **Conda** or **uv** installation to simplify running Python scripts from the terminal. Our versions are 
```
isaaclab==2.3.0
isaacsim==5.1.0.0
```

---

### 2. Install `fingereye` in Editable Mode

Using a Python interpreter that has **Isaac Lab properly installed**, install this package in editable mode:

```bash
# Use `PATH_TO_isaaclab.sh -p` or `PATH_TO_isaaclab.bat -p` instead of `python`
# if Isaac Lab is not installed in a standard conda/venv environment
python -m pip install -e source/fingereye
```

After installation, FingerEye simulation modules will be available within Isaac Lab workflows.

---

# 📡 Teleoperation & State-Based Data Collection

To accelerate teleoperation and support dataset augmentation, we first collect **state-based data** using third-person rendering only.

### Step 1 — Start Teleoperation Server

In the **FingerEye Real-World** repository (`FingerEyeRW`), run:

```bash
python teleop_server.py --config-name coin_standing
```

### Step 2 — Start Recording in Simulation

In this repository:

```bash
python scripts/record.py \
    --task="FingerEye-Teleop-Lab-Direct-v1" \
    --num_envs=1 \
    --headless \
    --enable_cameras
```

Recorded demonstrations will be saved as `.pkl` files.

---

# 🔄 Dataset Conversion (Pickle → Zarr)

Convert recorded pickle files into `.zarr` format for training compatibility:

```bash
python pkl2zarr.py \
    -i {path_to_pickle_dir} \
    -o {path_to_output_zarr}
```

The resulting `.zarr` dataset can be directly used in the **FingerEye Policy** repository.

---

# ▶️ Replay in Simulation

### Replay in Default Environment

```bash
python scripts/replay.py \
    --task "FingerEye-Replay-Lab-Direct-v1" \
    --num_envs 64 \
    --enable_cameras \
    --headless \
    --zarr_path {path_to_zarr}
```

---

### Replay in Randomized Environment

```bash
python scripts/replay.py \
    --task "FingerEye-Replay-Random-Lab-Direct-v1" \
    --num_envs 32 \
    --enable_cameras \
    --headless \
    --zarr_path {path_to_zarr}
```

---

# 📊 Dataset Visualization

Visualize stored `.zarr` data:

```bash
python visualize_zarr_data.py \
    --path {path_to_zarr}
```

---

# 🤖 Policy Evaluation in Simulation

Evaluate a trained policy checkpoint inside the simulation:

```bash
python scripts/record.py \
    --task="FingerEye-Lab-Direct-v1" \
    --num_envs=64 \
    --enable_cameras \
    --headless \
    eval_ckpt_path={ckpt_path} \
    mode=policy
```

Rollout videos and logs will be saved under `outputs`.

