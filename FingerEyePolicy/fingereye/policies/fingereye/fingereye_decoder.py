import torch
from torch import nn
from fingereye.policies.fingereye.act_blocks import ACTDecoder, StructuredACTDecoder


class FingerEyeDecoder(nn.Module):

    def __init__(
        self,
        # settings parameters
        n_obs_steps: int = 1,
        horizon: int = 1,
        ds: int = 8,
        da: int = 8,
        use_sim_pose_decoder: bool = False,
        # network parameters
        n_decoder_layers=4,
        dim_model=512,
        n_heads=8,   
        dim_feedforward=3200,
        feedforward_activation="relu",
        dropout=0.0,
        pre_norm=False,
        group_decoding=False,
    ):
        super().__init__()
        self.group_decoding = bool(group_decoding)
        self._token_debug_logged = False
        self.decoder_action_embeddings = nn.Embedding(horizon, dim_model)  
        self.state_projection = nn.Sequential(
            nn.Linear(ds * n_obs_steps, dim_model),
            nn.LayerNorm(dim_model),
            nn.GELU(), 
            nn.Linear(dim_model, dim_model),
        )

        decoder_cls = StructuredACTDecoder if self.group_decoding else ACTDecoder
        self.decoder = decoder_cls(
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

        self.use_sim_pose_decoder = use_sim_pose_decoder

        self._init_params()

    def _init_params(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, encoded_feats, state_in=None, encoder_pos_embed=None):
        b, ncam, dim_model = encoded_feats.shape

        decoder_inputs = self.decoder_action_embeddings.weight.unsqueeze(0).expand(b, -1, -1)  # (b, Tp, dim_model)
        if state_in is not None:
            state_feats = self.state_projection(state_in)  # (b, dim_model)
            if self.group_decoding:
                encoded_feats = torch.cat([encoded_feats, state_feats.unsqueeze(1)], dim=1)
            else:
                decoder_inputs = decoder_inputs + state_feats.unsqueeze(1)  # (b, Tp, dim_model)
        elif self.group_decoding:
            raise ValueError("group_decoding=True requires state_in to create the joint token.")

        if self.group_decoding and not self._token_debug_logged:
            print("FingerEyeDecoder token debug:")
            print(f"  decoder_query_tokens={decoder_inputs.shape[1]}")
            print(f"  condition_tokens={encoded_feats.shape[1]}")
            print("  condition_order=[FE tokens..., wrist token, joint token]")
            print("  group_decoding=True")
            self._token_debug_logged = True
    
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
