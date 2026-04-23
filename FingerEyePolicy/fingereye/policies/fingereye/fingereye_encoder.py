import torch
import torchvision
from torch import nn
from itertools import chain
from fingereye.models.vision.radio import RADIO
from torchvision.ops.misc import FrozenBatchNorm2d
from fingereye.policies.fingereye.act_blocks import ACTEncoder, ACTSinusoidalPositionEmbedding2d
from torchvision.models._utils import IntermediateLayerGetter
from einops import rearrange

class FingerEyeEncoder(nn.Module):
    def __init__(
        self,
        # settings parameters
        nv: int = 5,
        n_obs_steps: int = 1,
        # network parameters
        n_encoder_layers=4,
        dim_model=512,
        n_heads=8,   
        dim_feedforward=3200,
        feedforward_activation="relu",
        dropout=0.0,
        pre_norm=False,
        use_robopan_embedding=False,
    ):
        super().__init__()
        self.vit = RADIO() 
        if use_robopan_embedding:
            # RoboPanoptes encoder with 3 Linear projections 
            # ViT summary to half dim_model
            # pos to 1/4 dim_model
            # orientation to 1/4 dim_model
            self.vit_projection_layers = nn.Linear(self.vit.summary_dim * n_obs_steps, dim_model//2)
            self.pos_projection_layers = nn.Linear(3 * n_obs_steps, dim_model // 4)
            self.orientation_projection_layers = nn.Linear(6 * n_obs_steps, dim_model // 4)
        else:
            # FingerEye encoder with nv embeddings
            self.vit_projection_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.vit.summary_dim * n_obs_steps, dim_model * 2),
                        nn.LayerNorm(dim_model * 2),
                        nn.GELU(), 
                        nn.Linear(dim_model * 2, dim_model),
                    ) for _ in range(nv)
                ]
            )
            self.nv_embeddings = nn.Embedding(nv, dim_model)
        self.encoder = ACTEncoder(
            n_encoder_layers=n_encoder_layers,
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            dropout=dropout,
            pre_norm=pre_norm,
        )
        self.use_robopan_embedding = use_robopan_embedding
        self._init_params()
    
    def _init_params(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_radio_summary(self, images=None, radio_summary=None):
        """
        images: (B, nobs, nv, C, H, W) | RGB | [0,1]
        return: (B, nobs, nv, dim_vit)
        """
        if images is None and radio_summary is None:
            raise ValueError("Either images or radio_summary must be provided.")
        elif images is not None and radio_summary is not None:
            raise ValueError("Only one of images or radio_summary can be provided.")
        elif images is not None and radio_summary is None:
            b, nobs, nv, c, h, w = images.shape
            nv = images.shape[2]
            images_stack = images.reshape(-1, c, h, w)
            with torch.no_grad():
                feat_grid, feat_summary = self.vit.get_feature_grid(images_stack, return_processed_img=False)  # (B*nv, Hp, Wp, D_vit)
                feat_summary = feat_summary.view(b, nobs, nv, -1)  # (B, nobs, nv, D_vit)
        elif radio_summary is not None:
            feat_summary = radio_summary # (B, nobs, nv, D_vit)
        return feat_summary  # (B, nobs, nv, D_vit)

    def forward(self, images=None, radio_summary=None, camera_poses=None):
        """
        images: (B, nobs, nv, C, H, W) | RGB | [0,1]
        camera_poses: (B, nobs, nv, 3+6) | optional
        return: (B, nv, dim_model)
        """
        feat_summary = self.get_radio_summary(images=images, radio_summary=radio_summary)  # (B, nobs, nv, D_vit)
        b, nobs, nv, _ = feat_summary.shape

        if self.use_robopan_embedding:
            # Process ViT summary
            feat_summary_vit = self.vit_projection_layers(feat_summary.view(b, nobs, nv, -1).permute(0,2,1,3).reshape(b, nv, -1))  # (B, nv, dim_model//2)
            # Process position
            positions = camera_poses[..., :3]  # (B, nobs, nv, 3)
            positions = positions.permute(0,2,1,3).reshape(b, nv, -1)  # (B, nv, 3*nobs)
            feat_summary_pos = self.pos_projection_layers(positions)  # (B, nv, dim_model//4)
            # Process orientation
            orientations = camera_poses[..., 3:]  # (B, nobs, nv, 6)
            orientations = orientations.permute(0,2,1,3).reshape(b, nv, -1)  # (B, nv, 6*nobs)
            feat_summary_ori = self.orientation_projection_layers(orientations)  # (B, nv, dim_model//4)
            # Combine all
            feat_summary = torch.cat([feat_summary_vit, feat_summary_pos, feat_summary_ori], dim=-1)  # (B, nv, dim_model)
        else:
            feat_summary = torch.stack([self.vit_projection_layers[i](torch.cat([feat_summary[:, j, i, ...] for j in range(nobs)], dim=-1)) for i in range(nv)], dim=1)  # (B, nv, dim_model)
            nv_embeddings = self.nv_embeddings.weight.unsqueeze(0).expand(b, -1, -1)  # (B, nv, dim_model)
            feat_summary = feat_summary + nv_embeddings  # (B, nv, dim_model)

        encoded_feats = self.encoder(feat_summary.permute(1, 0, 2)).permute(1, 0, 2)  # (B, nv, dim_model)
        return encoded_feats  # (B, nv, dim_model)
    

class ACTImageEncoder(nn.Module):
    def __init__(
        self,
        # network parameters
        n_encoder_layers=4,
        dim_model=512,
        n_heads=8,   
        dim_feedforward=3200,
        feedforward_activation="relu",
        dropout=0.0,
        pre_norm=False,
    ):
        super().__init__()
        backbone_model = getattr(torchvision.models, "resnet18")(
            replace_stride_with_dilation=[False, False, False],
            weights="ResNet18_Weights.IMAGENET1K_V1",
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(dim_model // 2)
        self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, dim_model, kernel_size=1)
        self.encoder = ACTEncoder(
            n_encoder_layers=n_encoder_layers,
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            dropout=dropout,
            pre_norm=pre_norm,
        )
        self._init_params()
    
    def _init_params(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_resnet_grid(self, images=None):
        """
        images: (B, nobs, nv, C, H, W) | RGB | [0,1]
        return: (B, nobs, nv, dim_vit)
        """
        b, nobs, nv, c, h, w = images.shape
        assert nobs == 1, "Only support nobs=1 for image encoder."
        # https://github.com/huggingface/lerobot/blob/14a15f90e762170209d283c3545523549841ca3d/src/lerobot/policies/act/modeling_act.py#L472
        images_stack = images.reshape(-1, c, h, w)
        cam_features = self.backbone(images_stack)["feature_map"]
        cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)  # 1, d, h, w
        cam_pos_embed_nv = cam_pos_embed.expand(nv, -1, -1, -1)  # nv, d, h, w
        cam_features = self.encoder_img_feat_input_proj(cam_features)

        # Rearrange features to (sequence, batch, dim).
        cam_features = rearrange(cam_features, "(b nv) c h w -> (nv h w) b c", b=b, nv=nv)
        cam_pos_embed = rearrange(cam_pos_embed_nv, "nv c h w -> (nv h w) c").unsqueeze(1)

        return cam_features, cam_pos_embed  

    def forward(self, images=None, radio_summary=None):
        cam_features, cam_pos_embed = self.get_resnet_grid(images=images)  # (sequence, B*nv*nobs, dim_model)
        encoded_feats = self.encoder(
            cam_features,
            pos_embed=cam_pos_embed,
        ).permute(1, 0, 2)  # (b, sequence, dim_model)
        return encoded_feats, cam_pos_embed  # (b, sequence, dim_model)
    

