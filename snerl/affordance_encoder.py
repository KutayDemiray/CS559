import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import make_encoder
from segnet.Handle.SegNet import SegnetEncoder


class MixedEncoder(nn.Module):
    def __init__(
        self,
        # snerl inputs
        encoder_type,
        obs_shape,
        feature_dim,
        num_layers,
        num_filters,
        output_logits=False,
        multiview=3,
        frame_stack=3,
        encoder_name=None,
        finetune_encoder=False,
        log_encoder=False,
        env_name=None,
        exp_type="affordance",  # kutay: "affordance" or "original" (rgb)
        render_mode="rgb_array",  # kutay: "rgb_array" or "rgbd_array"
        # affordance inputs
        in_chn=3,
        out_chn=2,  # binary mask
    ):
        self.snerl_encoder = make_encoder(
            encoder_type=encoder_type,
            obs_shape=obs_shape,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_filters=num_filters,
            output_logits=output_logits,
            multiview=multiview,
            frame_stack=frame_stack,
            encoder_name=encoder_name,
            finetune_encoder=finetune_encoder,
            log_encoder=log_encoder,
            env_name=env_name,
            exp_type=exp_type,
            render_mode=render_mode,
        )

        self.aff_encoder = SegnetEncoder(in_chn=in_chn, out_chn=out_chn)

        if not finetune_encoder:
            self.aff_encoder.eval()

        self.flatten = nn.Flatten()
        self.aff_fc1 = nn.Linear(out_features=feature_dim)

        # self.fc_final = nn.Linear(in_features=2 * feature_dim, out_features=feature_dim)

    def forward(self, obs):
        obs_enc = self.snerl_encoder(obs)

        aff_enc = self.flatten(self.aff_encoder(obs))
        aff_enc = self.aff_fc1(aff_enc)

        mixed_enc = torch.concat([obs_enc, aff_enc])
        return mixed_enc
