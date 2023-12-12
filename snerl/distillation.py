import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import make_encoder


class Distillation:
    def __init__(self, obs_shape, target_net: nn.Module, args):
        # super(Distillation, self).__init__()

        self.lr = args.distillation_lr
        self.alpha = args.distillation_scale
        self.distillation_type = args.distillation

        if target_net is None:
            self.target_net = make_encoder(
                args.encoder_type,
                obs_shape,
                args.encoder_feature_dim,
                args.num_layers,
                args.num_filters,
                output_logits=False,
                multiview=args.multiview,
                frame_stack=args.frame_stack,
                encoder_name=args.encoder_name,
                finetune_encoder=args.finetune_encoder,
                log_encoder=False,
                env_name=args.env_name,
            )
        else:
            self.target_net = target_net

        self.predictor_net = make_encoder(
            args.encoder_type,
            obs_shape,
            args.encoder_feature_dim,
            args.num_layers,
            args.num_filters,
            output_logits=False,
            multiview=args.multiview,
            frame_stack=args.frame_stack,
            encoder_name=args.encoder_name,
            finetune_encoder=args.finetune_encoder,
            log_encoder=False,
            env_name=args.env_name,
        )

        self.predictor_optim = torch.optim.Adam(
            self.predictor_net.parameters(), lr=args.distillation_lr
        )

    def step(self, obs):
        self.target_net.eval()

        with torch.no_grad():
            target_output = self.target_net(obs)

        self.predictor_net.train()
        predictor_output = self.predictor_net(obs)
        distillation_loss = F.mse_loss(predictor_output, target_output)

        self.predictor_optim.zero_grad()
        distillation_loss.backward()
        self.predictor_optim.step()

        if self.distillation_type == "encoder":
            self.target_net.train()

        return distillation_loss
