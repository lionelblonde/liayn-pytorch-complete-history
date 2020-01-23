from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from agents.nets import init


class PredNet(nn.Module):

    def __init__(self, in_size):
        super(PredNet, self).__init__()
        self.leak = 0.1
        # Create feature extractor
        self.pred_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(in_size, 256)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak, inplace=True)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(256, 256)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak, inplace=True)),
            ]))),
        ]))
        # Perform initialization
        self.pred_trunk.apply(init(nonlin='leaky_relu', param=self.leak))

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        x = self.pred_trunk(x)
        return x


class TargNet(nn.Module):

    def __init__(self, in_size):
        super(TargNet, self).__init__()
        self.leak = 0.1
        # Create feature extractor
        self.targ_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(in_size, 256)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak, inplace=True)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(256, 256)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak, inplace=True)),
            ]))),
        ]))
        # Perform initialization
        self.targ_trunk.apply(init(nonlin='leaky_relu', param=self.leak))
        # Make sure the target is never updated
        for param in self.targ_trunk.parameters():
            param.requires_grad = False

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        x = self.targ_trunk(x)
        return x


class RND(object):

    def __init__(self, in_size, device):
        self.device = device
        self.pred_net = PredNet(in_size).to(self.device)
        self.targ_net = TargNet(in_size).to(self.device)
        self.opt = torch.optim.Adam(self.pred_net.parameters(), lr=3e-4)

    def train(self, x):
        x = torch.FloatTensor(x).to(self.device)
        loss = (self.pred_net(x) -
                self.targ_net(x)).pow(2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def get_novelty(self, x):
        with torch.no_grad():
            x = torch.FloatTensor(x).cpu()
            return np.asscalar((self.pred_net(x) -
                                self.targ_net(x)).pow(2).mean().cpu().numpy().flatten())
