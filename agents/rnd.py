import math
from collections import OrderedDict

import torch
import torch.nn as nn

from agents.nets import init


class PredNet(nn.Module):

    def __init__(self, in_dim, width):
        super(PredNet, self).__init__()
        self.leak = 0.1
        # Create feature extractor
        self.pred_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(in_dim, width)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(width, width)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        # Perform initialization
        self.pred_trunk.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        x = self.pred_trunk(x)
        return x


class TargNet(nn.Module):

    def __init__(self, in_dim, width):
        super(TargNet, self).__init__()
        self.leak = 0.1
        # Create feature extractor
        self.targ_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(in_dim, width)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(width, width)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        # Perform initialization
        self.targ_trunk.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))
        # Make sure the target is never updated
        for param in self.targ_trunk.parameters():
            param.requires_grad = False

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        x = self.targ_trunk(x)
        return x


class RandNetDistill(object):

    def __init__(self, env, hps, device, width, lr):
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.hps = hps
        # Define the input dimension, depending on whether actions are used too.
        in_dim = self.ob_dim if self.hps.state_only else self.ob_dim + self.ac_dim
        self.pred_net = PredNet(in_dim, width).to(device)
        self.targ_net = TargNet(in_dim, width).to(device)
        self.opt = torch.optim.Adam(self.pred_net.parameters(), lr=lr)

    def train(self, ob, ac):
        assert sum([isinstance(x, torch.Tensor) for x in [ob, ac]]) == 2
        x = ob if self.hps.state_only else torch.cat([ob, ac], dim=-1)
        loss = (self.pred_net(x) - self.targ_net(x)).pow(2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def get_novelty(self, ob, ac):
        assert sum([isinstance(x, torch.Tensor) for x in [ob, ac]]) == 2
        x = ob if self.hps.state_only else torch.cat([ob, ac], dim=-1)
        x = x.cpu()
        with torch.no_grad():
            return (self.pred_net(x) - self.targ_net(x)).pow(2).mean(-1)
