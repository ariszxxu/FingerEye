import torch.nn.functional as F
import torch
from typing import Dict
from fingereye.utils.dataset.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from fingereye.policies.base_policy import BasePolicy
from fingereye.policies.fingereye.fingereye_encoder import FingerEyeEncoder
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

        group_encoding = bool(kwargs.get("group_encoding", False))
        group_decoding = bool(kwargs.get("group_decoding", False))
        n_fe_tokens = kwargs.get("n_fe_tokens", len(use_camera_indices))
        self.state_keys = kwargs.get("state_keys", None)

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
            group_encoding=group_encoding,
            n_fe_tokens=n_fe_tokens,
        )

        self.decoder = FingerEyeDecoder(
            n_obs_steps=n_obs_steps,
            horizon=horizon,
            ds=ds,
            da=da,
            use_sim_pose_decoder=use_sim_pose_decoder,

            n_decoder_layers=n_decoder_layers,
            dim_model=dim_model,
            n_heads=n_heads,   
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            dropout=dropout,
            pre_norm=pre_norm,
            group_decoding=group_decoding,
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
        self.kwargs = kwargs
        self.use_sim_pose_decoder = use_sim_pose_decoder
        self.group_encoding = group_encoding
        self.group_decoding = group_decoding
        self._param_debug_logged = False

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

        encoded_feats = self.encoder(
            images=images,
            radio_summary=radio_summary,
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
            if k.startswith("obs/"):
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
                if k.startswith("obs/"):
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
        self._print_param_debug_once()
        encoded_feats = self.forward_encoder(data_dict)  # (b, ncam, dim_model)
        cam_pos_embed = None
        if isinstance(encoded_feats, tuple) and len(encoded_feats) == 2:
            encoded_feats, cam_pos_embed = encoded_feats

        state_in = self.get_state_vectors_from_obs(data_dict)  # (b, To*ds)
        pred = self.decoder(
            encoded_feats=encoded_feats,  # (b, ncam, dim_model)
            state_in=state_in,  # (b, To*ds)
            encoder_pos_embed=cam_pos_embed
        )

        if sim_batch is not None and self.use_sim_pose_decoder:
            encoded_sim_feats = self.forward_encoder(sim_batch)  # (b, ncam, dim_model)
            sim_pred = self.decoder.sim_forward(encoded_sim_feats)
            pred.update(sim_pred)

        return pred
    
    def get_state_vectors_from_obs(self, obs_dict):
        state_obs = []
        sorted_keys = list(self.state_keys) if self.state_keys is not None else sorted(obs_dict.keys())
        for k in sorted_keys:
            if k not in obs_dict:
                raise KeyError(f"Missing policy state key: {k}")
            if "obs/" in k and "images" not in k and "rays" not in k and "transform" not in k and "tag" not in k and "pose" not in k:
                state_obs.append(obs_dict[k])
        if len(state_obs) == 0:
            raise ValueError("No state observations found for FingerEyePolicy.")
        state_obs = torch.cat(state_obs, dim=-1)
        state_obs = state_obs.view(state_obs.shape[0], -1)
        return state_obs

    def _print_param_debug_once(self):
        if self._param_debug_logged:
            return
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        group_encoder_params = sum(p.numel() for p in self.encoder.group_encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        structured_decoder_params = sum(p.numel() for p in self.decoder.decoder.parameters())
        print("FingerEyePolicy parameter debug:")
        print(f"  encoder_params={encoder_params}")
        print(f"  group_encoder_params={group_encoder_params}")
        print(f"  decoder_params={decoder_params}")
        print(f"  decoder_core_params={structured_decoder_params}")
        print(f"  group_encoding={self.group_encoding}")
        print(f"  group_decoding={self.group_decoding}")
        self._param_debug_logged = True

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
