import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("./")

from segnet.Handle.SegNet import SegnetEncoder, tie_weights


class MixedEncoder(nn.Module):
    def __init__(
        self,
        snerl_encoder,
        feature_dim,
        finetune_encoder=False,
        # affordance inputs
        in_chn=3,
        out_chn=2,  # binary mask
        frame_stack=2,
        weights_path="segnet/Handle/encoders/aff_encoder_30.pt",
    ):
        super(MixedEncoder, self).__init__()

        self.snerl_encoder = snerl_encoder
        self.snerl_encoder.cuda()

        self.frame_stack = frame_stack

        self.aff_encoder = SegnetEncoder(in_chn=in_chn, out_chn=out_chn)
        self.aff_encoder.cuda()
        self.aff_encoder.load_state_dict(torch.load(weights_path))

        if not finetune_encoder:
            self.aff_encoder.eval()

        self.flatten = nn.Flatten()
        self.aff_fc1 = nn.Linear(in_features=512 * 4 * 4, out_features=feature_dim)
        self.aff_mix_fc = nn.Linear(
            in_features=feature_dim * frame_stack * 3,  # 3: multiview
            out_features=feature_dim,
        )

        self.feature_dim = 2 * feature_dim

        # self.fc_final = nn.Linear(in_features=2 * feature_dim, out_features=feature_dim)

    def forward(self, obs, detach):
        obs_enc = self.snerl_encoder(obs, detach=detach)

        aff_encs = []

        with torch.no_grad():
            # print(obs.shape)
            for frame in range(obs.shape[1] // 3):
                aff_enc, _, _ = self.aff_encoder(
                    obs[:, 3 * frame : 3 * (frame + 1), ...]
                )
                aff_enc = self.flatten(aff_enc)
                aff_enc = self.aff_fc1(aff_enc)
                if detach:
                    aff_enc = aff_enc.detach()
                aff_encs.append(aff_enc.view(obs.shape[0], self.feature_dim // 2))

        aff_encs = torch.concat(aff_encs, dim=1)

        if detach:
            aff_encs = aff_encs.detach()

        aff_mixed_enc = self.aff_mix_fc(aff_encs)

        # print(obs_enc.shape)
        final_enc = torch.concat([obs_enc, aff_mixed_enc], dim=1)
        # print("final", final_enc.shape)
        return final_enc

    def copy_conv_weights_from(self, source):
        self.snerl_encoder.copy_conv_weights_from(source.snerl_encoder)
        self.aff_encoder.copy_conv_weights_from(source.aff_encoder)
