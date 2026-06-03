import copy
import torch
import numpy as np
from typing import Dict
from pathlib import Path
from typing import Union
from functools import cached_property

from fingereye.datasets.base import BaseDataset
from fingereye.utils.dataset.replay_buffer import ReplayBuffer
from fingereye.utils.torch.common import dict_apply, safe_from_numpy
from fingereye.utils.dataset.sampler import SequenceSampler, get_val_mask, downsample_mask
from fingereye.utils.dataset.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

# /
#  ├── data
#  │   ├── actions
#  │   │   ├── original_actions (226, 23) float32
#  │   │   └── target_transforms (226, 39, 4, 4) float32
#  │   └── obs
#  │       ├── all_qpos (226, 23) float32
#  │       ├── current_transforms (226, 39, 4, 4) float32
#  │       ├── rgb_images (226, 4, 3, 480, 640) uint8
#  │       ├── rs_depths (226, 2, 1, 480, 640) uint16
#  │       └── rs_rgb_images (226, 2, 3, 480, 640) uint8
#  └── meta
#      ├── camera_meta
#      ├── episode_ends (1,) int64
#      ├── link_name_list (39,) <U25
#      ├── rgb_camera_name_list (4,) <U8
#      └── rs_camera_name_list (2,) <U5

class FingerEyeDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        replay_buffer_keys=None,
        load_full_chunk=False,
        key_slice: dict = dict(),
        key_sample_t: dict = dict(),
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path,
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            keys=replay_buffer_keys,
            key_first_k={} if load_full_chunk else {key: pad_before + 1 for key in replay_buffer_keys if "obs/" in key or "aug/" in key },  # only load first pad_before + 1 frames for image keys to save memory
            key_slice=key_slice,
            key_sample_t=key_sample_t,
        )
        self.replay_buffer_keys = replay_buffer_keys
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            keys=self.replay_buffer_keys,
        )
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self):
        """
        We don't nomalize image related data, because image will be normalized in the model by their own normalizer.
        """
        normalizer = LinearNormalizer()
        normalizer["obs/rays"] = SingleFieldLinearNormalizer().create_identity()
        for key in self.replay_buffer_keys:
            if (
                (key.endswith("images") and not key.endswith("xyz_images"))
                or key.endswith("point_colors")
                or key.endswith("_quat")
                or key.endswith("_transforms")
                or key.endswith("masks")
                or key.endswith("mask")
                or key.endswith("radio")
                or key.endswith("ee")
                or key.endswith("sparsh")
                or key.endswith("visibility")
                or key.endswith("dino_grid")
                or key.endswith("coin_z_axis")
            ):
                # images / point_colors / masks are already normalize to (0, 1)
                normalizer[key] = SingleFieldLinearNormalizer().create_identity()
            elif key.endswith("depths"):
                # depths / xyz_images : shape (..., 1 or 3, h, w))
                normalizer[key] = SingleFieldLinearNormalizer().create_fit(
                    data=self.replay_buffer[key], mode="limits", keep_dim_indices=[-3]
                )
            elif key.endswith("point_clouds") or key.endswith("xyz_images") or key.endswith("robot_points") or key.endswith("point_tracks") or key.endswith("point_is_visible"):
                normalizer[key] = SingleFieldLinearNormalizer().create_identity()
            elif (
                key.endswith("states")
                or key.endswith("_qpos")
                or key.endswith("_qvel")
                or key.endswith("_pos")
                or key.endswith("pos")
                or key.endswith("vel")
                or key.endswith("_ang_vel")
                or key.endswith("_lin_vel")
                or key.endswith("current_rgb_camera_poses")
                or key.endswith("current_rs_camera_poses")  # create limit(range) normalizer for camera poses
            ):
                # states / original_actions / qpos / qvel / pos / ang_vel / lin_vel : shape (..., d_state or d_action)
                normalizer[key] = SingleFieldLinearNormalizer().create_fit(
                    data=self.replay_buffer[key], mode="limits", keep_dim_indices=[-1]
                )
            elif (
                key.endswith("original_actions")
                or key.endswith("delta_transforms_actions")
                or key.endswith("delta_actions")
                or key.endswith("tag_ori")
                or key.endswith("theta")
                or key.endswith("theta_dot")
                or key.endswith("box_corners")
                or key.endswith("pts")
                or key.endswith("state")
                or key.endswith("current_joint_values")
                or key.endswith("tracks")
                or key.endswith("current_center_tag_T")
                or key.endswith("pos_of_coin")
            ):
                normalizer[key] = SingleFieldLinearNormalizer().create_fit(
                    data=self.replay_buffer[key], mode="gaussian", keep_dim_indices=[-1]
                )
            else:
                raise ValueError(f"key {key} not supported")
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample, idx=None):
        sample_data = {}
        for key in self.replay_buffer_keys:
            arr = sample[key]

            if key.endswith("images") and not key.endswith("xyz_images"):
                if arr.dtype == np.uint8:
                    # uint8 (0–255) → float32 (0–1)
                    sample_data[key] = arr.astype(np.float32) / 255.0
                else:
                    # already float, assume correct range
                    sample_data[key] = arr.astype(np.float32, copy=False)
            else:
                # keep float32 as-is, avoid copy
                sample_data[key] = arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)

        return sample_data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, idx)
        torch_data = dict_apply(data, safe_from_numpy)
        torch_data["dataset_idx"] = torch.tensor(idx, dtype=torch.long)
        return torch_data


if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf

    config = OmegaConf.load("../configs/task/lift_fingereye_uni.yaml")
    dataset = hydra.utils.instantiate(config.dataset)
    dataset_0 = dataset[0]
    for key in dataset_0.keys():
        print(key, dataset_0[key].shape)
