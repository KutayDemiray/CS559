import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import make_encoder

import copy


class Distillation:
    def __init__(self, obs_shape, target_net: nn.Module, args):
        # super(Distillation, self).__init__()
        self.lr = args.distillation_lr
        self.alpha = args.distillation_scale
        self.distillation_type = args.distillation

        if target_net is None:
            print("creating random distiller")
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
            print("creating encoder distiller")
            self.target_net = copy.deepcopy(target_net)

        # nn.init.normal_(self.target_net.parameters(), mean=0, std=1)

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
        # self.predictor_net = copy.deepcopy(self.target_net)
        nn.init.normal_(self.predictor_net.mlp1.weight)
        nn.init.normal_(self.predictor_net.mlp2.weight)
        nn.init.normal_(self.predictor_net.fc.weight)
        nn.init.normal_(self.predictor_net.ln.weight)
        nn.init.normal_(self.predictor_net.convs[0].weight)
        nn.init.normal_(self.predictor_net.convs[1].weight)
        nn.init.normal_(self.predictor_net.convs[2].weight)
        self.predictor_optim = torch.optim.Adam(
            self.predictor_net.parameters(), lr=args.distillation_lr
        )

        # nn.init.normal_(self.predictor_net.parameters(), mean=0, std=1)

    def step(self, obs):
        self.target_net.eval()

        with torch.no_grad():
            obs = obs.unsqueeze(0)
            target_output = self.target_net(obs)

        self.predictor_net.train()
        predictor_output = self.predictor_net(obs)
        distillation_loss = F.mse_loss(predictor_output, target_output)

        self.predictor_optim.zero_grad()
        distillation_loss.backward()
        self.predictor_optim.step()

        # if self.distillation_type == "encoder":
        #    self.target_net.train()

        return distillation_loss

    def save(self, model_dir, step):
        torch.save(
            {
                "target_net": self.target_net.state_dict(),
                "predictor_net": self.predictor_net.state_dict(),
                "optim": self.predictor_optim.state_dict(),
            },
            "%s/dist_%s.pt" % (model_dir, step),
        )

    def load(self, model_path):
        ckpt = torch.load(model_path)
        self.target_net.load_state_dict(ckpt["target_net"])
        self.predictor_net.load_state_dict(ckpt["predictor_net"])
        self.predictor_optim.load_state_dict(ckpt["optim"])
