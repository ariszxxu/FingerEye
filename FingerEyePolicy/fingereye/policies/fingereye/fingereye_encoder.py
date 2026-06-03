import torch
from torch import nn

from fingereye.models.vision.radio import RADIO
from fingereye.policies.fingereye.act_blocks import ACTEncoder


class FingerEyeEncoder(nn.Module):
    def __init__(
        self,
        nv: int = 5,
        n_obs_steps: int = 1,
        n_encoder_layers=4,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        feedforward_activation="relu",
        dropout=0.0,
        pre_norm=False,
        group_encoding=False,
        n_fe_tokens: int | None = None,
    ):
        super().__init__()
        self.vit = RADIO()
        self.nv = int(nv)
        self.n_obs_steps = int(n_obs_steps)
        self.dim_model = int(dim_model)
        self.group_encoding = bool(group_encoding)
        self.n_fe_tokens = int(n_fe_tokens) if n_fe_tokens is not None else max(self.nv - 1, 0)
        self._token_debug_logged = False

        self.vit_projection_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.vit.summary_dim * self.n_obs_steps, dim_model * 2),
                    nn.LayerNorm(dim_model * 2),
                    nn.GELU(),
                    nn.Linear(dim_model * 2, dim_model),
                )
                for _ in range(self.nv)
            ]
        )
        self.nv_embeddings = nn.Embedding(self.nv, dim_model)
        self.group_encoder = ACTEncoder(
            n_encoder_layers=n_encoder_layers,
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            dropout=dropout,
            pre_norm=pre_norm,
        ) if self.group_encoding else nn.Identity()
        self._init_params()

    def _init_params(self):
        if isinstance(self.group_encoder, ACTEncoder):
            for p in self.group_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def get_radio_summary(self, images=None, radio_summary=None):
        if images is None and radio_summary is None:
            raise ValueError("Either images or radio_summary must be provided.")
        if images is not None and radio_summary is not None:
            raise ValueError("Only one of images or radio_summary can be provided.")
        if radio_summary is not None:
            return radio_summary

        b, nobs, nv, c, h, w = images.shape
        images_stack = images.reshape(-1, c, h, w)
        with torch.no_grad():
            _, feat_summary = self.vit.get_feature_grid(images_stack, return_processed_img=False)
        return feat_summary.view(b, nobs, nv, -1)

    def forward(self, images=None, radio_summary=None):
        feat_summary = self.get_radio_summary(images=images, radio_summary=radio_summary)
        b, nobs, nv, _ = feat_summary.shape
        if nv != self.nv:
            raise ValueError(f"FingerEyeEncoder expected nv={self.nv}, got {nv}.")
        if nobs != self.n_obs_steps:
            raise ValueError(f"FingerEyeEncoder expected n_obs_steps={self.n_obs_steps}, got {nobs}.")

        projected = []
        for i in range(nv):
            token_i = feat_summary[:, :, i].reshape(b, -1)
            projected.append(self.vit_projection_layers[i](token_i))
        tokens = torch.stack(projected, dim=1)
        tokens = tokens + self.nv_embeddings.weight.unsqueeze(0).expand(b, -1, -1)

        if self.group_encoding and self.n_fe_tokens > 1:
            fe_tokens = tokens[:, : self.n_fe_tokens]
            encoded_fe = self.group_encoder(fe_tokens.permute(1, 0, 2)).permute(1, 0, 2)
            tokens = torch.cat([encoded_fe, tokens[:, self.n_fe_tokens :]], dim=1)

        if not self._token_debug_logged:
            print("FingerEyeEncoder token debug:")
            print(f"  visual_tokens={nv}")
            print(f"  fe_tokens={min(self.n_fe_tokens, nv)}")
            print(f"  wrist_tokens={max(nv - self.n_fe_tokens, 0)}")
            print(f"  group_encoding={self.group_encoding}")
            self._token_debug_logged = True

        return tokens
