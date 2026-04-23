import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Dict
from itertools import chain
from einops import rearrange
from fingereye.utils.dataset.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from fingereye.policies.base_policy import BasePolicy
from fingereye.policies.fingereye.fingereye_encoder import FingerEyeEncoder, ACTImageEncoder
from fingereye.policies.fingereye.fingereye_decoder import FingerEyeDecoder


class FingerEyePolicy(BasePolicy):
    def __init__(
        self,
        # task params
        horizon=1,
        n_action_steps=1,
        n_obs_steps=1,

        nv=5,  # number of views
        use_rs_indices=[],
        use_camera_indices=[],
        n_tags=0,
        n_tag_steps=0,
        use_sim_pose_decoder=False,

        # other dimensions
        ds=8,
        da=8,

        # transformer encoder & decoder
        n_encoder_layers=4,
        n_decoder_layers=4,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        feedforward_activation="relu",
        dropout=0.0,
        pre_norm=False,

        **kwargs,
    ):
        super().__init__()

        use_resnet_encoder = kwargs.get("use_resnet_encoder", False)
        use_robopan_embedding = kwargs.get("use_robopan_embedding", False)

        if not use_resnet_encoder:
            # use RADIO-based encoder
            self.encoder = FingerEyeEncoder(
                nv=nv,
                n_obs_steps=n_obs_steps,

                n_encoder_layers=n_encoder_layers,
                dim_model=dim_model,
                n_heads=n_heads,   
                dim_feedforward=dim_feedforward,
                feedforward_activation=feedforward_activation,
                dropout=dropout,
                pre_norm=pre_norm,

                use_robopan_embedding=use_robopan_embedding,
            )
        else:
            # use original ACT image encoder with ResNet backbone
            self.encoder = ACTImageEncoder(
                n_encoder_layers=n_encoder_layers,
                dim_model=dim_model,
                n_heads=n_heads,   
                dim_feedforward=dim_feedforward,
                feedforward_activation=feedforward_activation,
                dropout=dropout,
                pre_norm=pre_norm,
            )

        self.decoder = FingerEyeDecoder(
            n_obs_steps=n_obs_steps,
            horizon=horizon,
            ds=ds,
            da=da,
            n_tags=n_tags,
            n_tag_steps=n_tag_steps,
            use_sim_pose_decoder=use_sim_pose_decoder,

            n_decoder_layers=n_decoder_layers,
            dim_model=dim_model,
            n_heads=n_heads,   
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            dropout=dropout,
            pre_norm=pre_norm,
        )

        self.use_rs_indices = list(use_rs_indices)
        self.use_camera_indices = list(use_camera_indices)
        self.ds = ds 
        self.da = da        
        self.dim_model = dim_model
        self.normalizer = LinearNormalizer()
        self.sim_normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_tag_steps = n_tag_steps
        self.kwargs = kwargs
        self.use_robopan_embedding = use_robopan_embedding
        self.use_sim_pose_decoder = use_sim_pose_decoder

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_sim_normalizer(self, normalizer: LinearNormalizer):
        self.sim_normalizer.load_state_dict(normalizer.state_dict())

    def forward_encoder(self, obs_dict):
        image_list = []
        if "obs/rgb_images" in obs_dict and len(self.use_camera_indices) > 0:
            rgb_images = obs_dict["obs/rgb_images"][:, :, self.use_camera_indices]  # (b, To, nfingereye x 2, 3, h, w)
            image_list.append(rgb_images)
        if "obs/rs_rgb_images" in obs_dict and len(self.use_rs_indices) > 0:
            rs_rgb_images = obs_dict["obs/rs_rgb_images"][:, :, self.use_rs_indices]  # (b, To, nrs, 3, h, w)
            image_list.append(rs_rgb_images)
        if len(image_list) == 0:
            images = None 
        else:
            images = torch.cat(image_list, dim=2)  # (b, To*ncam, 3, h, w)

        radio_list = []
        if "obs/rgb_images_radio" in obs_dict and len(self.use_camera_indices) > 0:
            rgb_radio = obs_dict["obs/rgb_images_radio"][:, :, self.use_camera_indices]  # (b, To, nfingereye x 2, d)
            radio_list.append(rgb_radio)
        if "obs/rs_rgb_images_radio" in obs_dict and len(self.use_rs_indices) > 0:
            rs_rgb_radio = obs_dict["obs/rs_rgb_images_radio"][:, :, self.use_rs_indices]  # (b, To, nrs, d)
            radio_list.append(rs_rgb_radio)
        if len(radio_list) == 0:
            radio_summary = None 
        else:
            radio_summary = torch.cat(radio_list, dim=2)  # (b, To*ncam, d)

        # for robopan embedding
        camera_poses = None 
        if self.use_robopan_embedding:
            camera_poses_list = []
            if "obs/current_rgb_camera_poses" in obs_dict and len(self.use_camera_indices) > 0:
                rgb_camera_poses = obs_dict["obs/current_rgb_camera_poses"][:, :, self.use_camera_indices]  # (b, To, nfingereye x 2, 3+6)
                camera_poses_list.append(rgb_camera_poses)
            if "obs/current_rs_camera_poses" in obs_dict and len(self.use_rs_indices) > 0:
                rs_camera_poses = obs_dict["obs/current_rs_camera_poses"][:, :, self.use_rs_indices]  # (b, To, nrs, 3+6)
                camera_poses_list.append(rs_camera_poses)
            camera_poses = torch.cat(camera_poses_list, dim=2)  # (b, To, ncam, 3+9)  
            camera_poses = camera_poses[..., :3+6]  # use the first 6 elements of the rotation matrix as in RoboPan

        encoded_feats = self.encoder(
            images=images,  # (b, To*ncam, 3, h,
            radio_summary=radio_summary,  # (b, To*ncam, d)
            camera_poses=camera_poses  # (b, ncam, 3+6)
        )  # (b, ncam, dim_model)
        return encoded_feats

    def normalize_batch(self, batch, sim_batch=None):
        for k, v in batch.items():
            if k not in self.normalizer.params_dict.keys():
                print(
                    f"Warning: {k} not in normalizer. Creating identity normalizer for {k}"
                )
                self.normalizer[k] = (
                    SingleFieldLinearNormalizer().create_identity().to(self.device)
                )
        nbatch = self.normalizer.normalize(batch)
        nbatch_To1 = {}
        for k, v in batch.items():
            if k.startswith("obs/") and k != "obs/current_center_tag_T":
                nbatch_To1[k] = nbatch[k][:, : self.n_obs_steps]
            else:
                nbatch_To1[k] = nbatch[k]

        nsim_batch = None 
        nsim_batch_To1 = None
        if sim_batch is not None:
            for k, v in sim_batch.items():
                if k not in self.sim_normalizer.params_dict.keys():
                    print(
                        f"Warning: {k} not in sim_normalizer. Creating identity normalizer for {k}"
                    )
                    self.sim_normalizer[k] = (
                        SingleFieldLinearNormalizer().create_identity().to(self.device)
                    )
            nsim_batch = self.sim_normalizer.normalize(sim_batch)
            nsim_batch_To1 = {}
            for k, v in sim_batch.items():
                if k.startswith("obs/") and k != "obs/current_center_tag_T":
                    nsim_batch_To1[k] = nsim_batch[k][:, : self.n_obs_steps]
                else:
                    nsim_batch_To1[k] = nsim_batch[k]

        return nbatch, nbatch_To1, nsim_batch, nsim_batch_To1
    
    def compute_loss(self, batch, sim_batch=None):
        action_key = "actions/original_actions"
        nbatch, nbatch_To1, nsim_batch, nsim_batch_To1 = self.normalize_batch(batch, sim_batch=sim_batch)
        pred = self.forward(nbatch_To1, sim_batch=nsim_batch_To1)

        loss_dict = {}
        loss = 0.0

        actions_hat = pred["actions"]
        actions_gt = nbatch[action_key]
        action_loss = (F.l1_loss(actions_gt, actions_hat, reduction="none")).mean()
        loss_dict["action_loss"] = action_loss
        loss += action_loss

        if self.use_sim_pose_decoder and sim_batch is not None:
            sim_M9_hat = pred["sim_M9"]
            sim_z_axis_gt = nsim_batch["obs/coin_z_axis"][:, : self.n_obs_steps].squeeze()
            sim_pos_gt = nsim_batch["obs/pos_of_coin"][:, : self.n_obs_steps].squeeze()
            sim_M9_gt = torch.cat([sim_z_axis_gt, sim_pos_gt], dim=-1)
            sim_M9_loss = (F.l1_loss(sim_M9_gt, sim_M9_hat, reduction="none")).mean()
            loss_dict["sim_M9_loss"] = sim_M9_loss
            loss += 0.1 * sim_M9_loss 

        loss_dict["loss"] = loss

        return loss_dict

    def forward(self, data_dict, sim_batch = None):
        encoded_feats = self.forward_encoder(data_dict)  # (b, ncam, dim_model)
        cam_pos_embed = None
        if isinstance(encoded_feats, tuple) and len(encoded_feats) == 2:
            encoded_feats, cam_pos_embed = encoded_feats

        state_in = self.get_state_vectors_from_obs(data_dict)  # (b, To*ds)
        tag_in = None
        if self.n_tag_steps > 0:
            tag_in = self.get_tag_vectors_from_obs(data_dict)  # (b, n_tag_step*6)
        pred = self.decoder(
            encoded_feats=encoded_feats,  # (b, ncam, dim_model)
            state_in=state_in,  # (b, To*ds)
            tag_in=tag_in,  # (b, n_tag_step*6)
            encoder_pos_embed=cam_pos_embed
        )

        if sim_batch is not None and self.use_sim_pose_decoder:
            encoded_sim_feats = self.forward_encoder(sim_batch)  # (b, ncam, dim_model)
            sim_pred = self.decoder.sim_forward(encoded_sim_feats)
            pred.update(sim_pred)

        return pred
    
    def get_state_vectors_from_obs(self, obs_dict):
        state_obs = []
        sorted_keys = sorted(obs_dict.keys())
        for k in sorted_keys:
            if "obs/" in k and "images" not in k and "rays" not in k and "transform" not in k and "tag" not in k and "pose" not in k:
                state_obs.append(obs_dict[k])
        state_obs = torch.cat(state_obs, dim=-1)
        state_obs = state_obs.view(state_obs.shape[0], -1)
        return state_obs

    def get_tag_vectors_from_obs(self, obs_dict):
        state_obs = []
        sorted_keys = sorted(obs_dict.keys())
        for k in sorted_keys:
            if "obs/" in k and "tag" in k:
                state_obs.append(obs_dict[k])
        state_obs = torch.cat(state_obs, dim=-1)
        state_obs = state_obs.view(state_obs.shape[0], -1)
        return state_obs

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], action_key=None, is_first_frame=False
    ) -> Dict[str, torch.Tensor]:
        obs_dict_w_prefix = obs_dict
        if len(self.use_rs_indices) == 0:
            obs_dict_w_prefix.pop("obs/rs_rgb_images", None)
            obs_dict_w_prefix.pop("obs/rs_rgb_images_radio", None)
        if len(self.use_camera_indices) == 0:
            obs_dict_w_prefix.pop("obs/rgb_images", None)
            obs_dict_w_prefix.pop("obs/rgb_images_radio", None)
        for k, v in obs_dict_w_prefix.items():
            if k not in self.normalizer.params_dict.keys():
                print(f"Warning: {k} not in normalizer. Creating identity normalizer for {k}")
                self.normalizer[k] = (
                    SingleFieldLinearNormalizer().create_identity().to(self.device)
                )
        nobs = self.normalizer.normalize(obs_dict_w_prefix)

        pred = self.forward(nobs)

        naction_pred = pred["actions"]
        action_key = "actions/original_actions"
        action_pred = self.normalizer[action_key].unnormalize(naction_pred)

        result = {}
        action = action_pred[:, self.n_obs_steps - 1:self.n_obs_steps - 1 + self.n_action_steps]
        result["actions"] = action
        result["actions_pred"] = action_pred
        result["attention_weights"] = pred["attention_weights"]
        return result

