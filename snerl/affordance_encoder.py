import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("./")

from segnet.Handle.SegNet import SegnetEncoder


class MixedEncoder(nn.Module):
    def __init__(
        self,
        snerl_encoder,
        feature_dim,
        finetune_encoder=False,
        # affordance inputs
        in_chn=3,
        out_chn=2,  # binary mask
        weights_path="segnet/Handle/encoders/aff_encoder_30.pt",
    ):
        super(MixedEncoder, self).__init__()

        self.snerl_encoder = snerl_encoder
        self.aff_encoder = SegnetEncoder(in_chn=in_chn, out_chn=out_chn)
        self.aff_encoder.cuda()
        self.aff_encoder.load_state_dict(torch.load(weights_path))

        if not finetune_encoder:
            self.aff_encoder.eval()

        self.flatten = nn.Flatten()
        self.aff_fc1 = nn.Linear(in_features=512 * 4 * 4, out_features=feature_dim)

        # self.fc_final = nn.Linear(in_features=2 * feature_dim, out_features=feature_dim)

    def forward(self, obs):
        obs_enc = self.snerl_encoder(obs)

        aff_enc = self.flatten(self.aff_encoder(obs))
        aff_enc = self.aff_fc1(aff_enc)

        mixed_enc = torch.concat([obs_enc, aff_enc])
        return mixed_enc
