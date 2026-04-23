# FingerEye Policy
**Continuous and Unified Vision-Tactile Sensing for Dexterous Manipulation**

## 📚 Repository Structure

This folder is one component of the main **FingerEye** monorepo.

- FingerEye Policy (current): `./`
- FingerEye Real-World: `../FingerEyeRW/`
- FingerEye Lab: `../FingerEyeLab/`

This repository contains the implementation of the **FingerEye policy**, along with the data preprocessing and training pipeline used in our work on continuous and unified vision–tactile sensing for dexterous manipulation.

> **Scope of this repository**
> - Policy architectures and training code  
> - Dataset preprocessing (e.g., visual feature caching)  
> - Offline training
>
> Real-world data collection, hardware deployment, and the simulation digital twin are provided in the other subdirectories of this monorepo.

---

## 🛠️ Installation

We recommend using Conda.

```bash
conda create -n fingereye python=3.11
conda activate fingereye
pip install --upgrade pip
```

Install PyTorch and torchvision according to your CUDA setup, then install remaining dependencies:

```bash
pip install zarr==2.18.3 numcodecs==0.13.1 viser termcolor hydra-core wandb[media] numba dill einops diffusers transformers prettytable opencv-python nvitop accelerate
```

A full list of dependency versions is provided in `requirements.txt`.

Finally, install this repository in editable mode:

```bash
pip install -e .
```

---

## 📦 Data Preprocessing

After collecting demonstrations, preprocess the dataset to compute and cache **RADIO visual summaries**.
This significantly reduces GPU memory usage and accelerates training.

```bash
python -m fingereye.datasets.add_radio_summary_to_zarr \
    --path "your_dataset.zarr" \
    --batch_size 512
```

This step only needs to be run **once per dataset**.

---

## 🗂️ Data & Task Configuration

Example task and dataset configuration files are provided under:

```
fingereye/configs/setting/
```

Before training, update the corresponding `zarr_path` field in the selected setting file to point to your dataset.

---

## 🚀 Training

From the `fingereye/workspaces` directory, launch training with:

```bash
python workspace.py --config-name fingereye setting=<setting-name>
```

Training outputs (logs, checkpoints, and configs) will be saved to:

```
fingereye/workspaces/outputs/
```
