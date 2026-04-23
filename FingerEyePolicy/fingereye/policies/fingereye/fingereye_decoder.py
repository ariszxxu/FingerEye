import torch
from torch import nn
from fingereye.policies.fingereye.act_blocks import ACTDecoder


class TagFiLMEncoder(nn.Module):
    def __init__(
        self,
        tag_input_dim: int,
        dim_model: int = 512,
        rank_ratio: float = 0.25,   # low-rank FiLM
    ):
        super().__init__()
        self.d_low = int(dim_model * rank_ratio)
        self.encoder = nn.Sequential(
            nn.Linear(tag_input_dim, self.d_low),
            nn.LayerNorm(self.d_low),
            nn.GELU(),
        )
        self.gamma = nn.Linear(self.d_low, self.d_low)
        self.up = nn.Linear(self.d_low, dim_model, bias=False)
        self._init_identity()

    def _init_identity(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)

    def forward(self, tag_in):
        h = self.encoder(tag_in)           # (B, d_low)
        gamma = torch.tanh(self.gamma(h))  # stabilize
        gamma_up = self.up(gamma)           # (B, dim_model)
        return gamma_up


class FingerEyeDecoder(nn.Module):

    def __init__(
        self,
        # settings parameters
        n_obs_steps: int = 1,
        horizon: int = 1,
        ds: int = 8,
        da: int = 8,
        n_tags: int = 0,
        n_tag_steps: int = 0,
        use_sim_pose_decoder: bool = False,
        # network parameters
        n_decoder_layers=4,
        dim_model=512,
        n_heads=8,   
        dim_feedforward=3200,
        feedforward_activation="relu",
        dropout=0.0,
        pre_norm=False,
    ):
        super().__init__()
        self.decoder_action_embeddings = nn.Embedding(horizon, dim_model)  
        self.state_projection = nn.Sequential(
            nn.Linear(ds * n_obs_steps, dim_model),
            nn.LayerNorm(dim_model),
            nn.GELU(), 
            nn.Linear(dim_model, dim_model),
        )
        if n_tag_steps > 0:
            self.tag_projection = TagFiLMEncoder(
                tag_input_dim=n_tags * n_tag_steps * 6,
                dim_model=dim_model,
                rank_ratio=0.25,
                use_beta=False,
            )

        self.decoder = ACTDecoder(
            n_decoder_layers=n_decoder_layers,
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            dropout=dropout,
            pre_norm=pre_norm,
        )
        self.action_head = nn.Sequential(
            nn.Linear(dim_model, dim_model // 2),
            nn.GELU(),
            nn.Linear(dim_model // 2, da),
        )
        if use_sim_pose_decoder:
            self.sim_pose_head = nn.Sequential(
                nn.Linear(dim_model, dim_model // 2),
                nn.GELU(),
                nn.Linear(dim_model // 2, 3+6),
            )

        self.n_tag_steps = n_tag_steps
        self.use_sim_pose_decoder = use_sim_pose_decoder

        self._init_params()

    def _init_params(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, encoded_feats, state_in=None, tag_in=None, encoder_pos_embed=None):
        b, ncam, dim_model = encoded_feats.shape

        decoder_inputs = self.decoder_action_embeddings.weight.unsqueeze(0).expand(b, -1, -1)  # (b, Tp, dim_model)
        if state_in is not None:
            state_feats = self.state_projection(state_in)  # (b, dim_model)
            decoder_inputs = decoder_inputs + state_feats.unsqueeze(1)  # (b, Tp, dim_model)
        if self.n_tag_steps > 0 and tag_in is not None:
            gamma = self.tag_projection(tag_in)     # (B, 512)
            decoder_inputs = decoder_inputs * (1.0 + gamma.unsqueeze(1))
    
        decoder_outputs, attn_weights = self.decoder(
            decoder_inputs.permute(1, 0, 2),
            encoded_feats.permute(1, 0, 2),
            encoder_pos_embed=encoder_pos_embed,
        )
        decoder_outputs = decoder_outputs.permute(1, 0, 2)  # (b, Tp, dim_model)

        outputs = {}
        action_preds = self.action_head(decoder_outputs)  # (b, Tp, da)
        outputs["actions"] = action_preds
        outputs["attention_weights"] = attn_weights
        return outputs  
    
    def sim_forward(self, encoded_feats):
        outputs = {}
        if self.use_sim_pose_decoder:
            b, ncam, dim_model = encoded_feats.shape
            encoded_feats = encoded_feats.reshape(b, -1)  # (b, ncam*dim_model)
            sim_pose_preds = self.sim_pose_head(encoded_feats)  # (b, ncam, 9)
            outputs["sim_M9"] = sim_pose_preds
        return outputs