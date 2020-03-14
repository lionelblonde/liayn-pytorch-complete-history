import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from helpers.distributed_util import RunMoms


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.

def init(weight_scale=1., constant_bias=0.):
    """Perform orthogonal initialization"""

    def _init(m):

        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=weight_scale)
            if m.bias is not None:
                nn.init.constant_(m.bias, constant_bias)
        elif (isinstance(m, nn.BatchNorm2d) or
              isinstance(m, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return _init


def snwrap(use_sn=False):
    """Spectral normalization wrapper"""

    def _snwrap(m):

        assert isinstance(m, nn.Linear)
        if use_sn:
            return U.spectral_norm(m)
        else:
            return m

    return _snwrap


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Models.

class Discriminator(nn.Module):

    def __init__(self, env, hps):
        super(Discriminator, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        if hps.wrap_absorb:
            ob_dim += 1
            ac_dim += 1
        self.hps = hps
        self.leak = 0.1
        apply_sn = snwrap(use_sn=self.hps.spectral_norm)
        # Define observation whitening
        self.rms_obs = RunMoms(shape=env.observation_space.shape, use_mpi=True)
        # Define the input dimension, depending on whether actions are used too.
        in_dim = ob_dim if self.hps.state_only else ob_dim + ac_dim
        # Assemble the last layers and output heads
        self.score_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', apply_sn(nn.Linear(in_dim, 100))),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', apply_sn(nn.Linear(100, 100))),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        self.score_head = nn.Linear(100, 1)
        # Perform initialization
        self.score_trunk.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))
        self.score_head.apply(init(weight_scale=0.01))

    def D(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        if self.hps.wrap_absorb:
            ob_ = ob.clone()[:, 0:-1]
            ob_ = torch.clamp(self.rms_obs.standardize(ob_), -5., 5.)
            ob = torch.cat([ob_, ob[:, -1].unsqueeze(-1)], dim=-1)
        else:
            ob = torch.clamp(self.rms_obs.standardize(ob), -5., 5.)
        x = ob if self.hps.state_only else torch.cat([ob, ac], dim=-1)
        x = self.score_trunk(x)
        score = self.score_head(x)
        return score  # no sigmoid here


class KYEDiscriminator(nn.Module):

    def __init__(self, env, hps):
        super(KYEDiscriminator, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        if hps.wrap_absorb:
            ob_dim += 1
            ac_dim += 1
        self.hps = hps
        apply_sn = snwrap(use_sn=self.hps.spectral_norm)
        # Define observation whitening
        self.rms_obs = RunMoms(shape=env.observation_space.shape, use_mpi=True)
        # Assemble the last layers and output heads
        self.ob_encoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', apply_sn(nn.Linear(ob_dim, 64))),
                ('nl', nn.Tanh()),
            ]))),
        ]))
        if not self.hps.state_only:
            self.ac_encoder = nn.Sequential(OrderedDict([
                ('fc_block', nn.Sequential(OrderedDict([
                    ('fc', apply_sn(nn.Linear(ac_dim, 64))),
                    ('nl', nn.Tanh()),
                ]))),
            ]))
        self.score_trunk = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', apply_sn(nn.Linear(128, 64))),
                ('nl', nn.Tanh()),
            ]))),
        ]))
        self.aux_trunk = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', apply_sn(nn.Linear(64, 64))),
                ('nl', nn.Tanh()),
            ]))),
        ]))
        self.score_head = nn.Linear(64, 1)
        self.aux_head = nn.Linear(64, env.action_space.shape[0])  # always original ac_dim
        # Perform initialization
        self.ob_encoder.apply(init(weight_scale=5./3.))
        if not self.hps.state_only:
            self.ac_encoder.apply(init(weight_scale=5./3.))
        self.score_trunk.apply(init(weight_scale=5./3.))
        self.aux_trunk.apply(init(weight_scale=5./3.))
        self.score_head.apply(init(weight_scale=0.01))
        self.aux_head.apply(init(weight_scale=0.01))

    def D(self, ob, ac):
        out = self.forward(ob, ac)
        return out[0]  # score

    def auxo(self, ob, ac):
        out = self.forward(ob, ac)
        return out[1]  # aux

    def forward(self, ob, ac):
        if self.hps.wrap_absorb:
            ob_ = ob.clone()[:, 0:-1]
            ob_ = torch.clamp(self.rms_obs.standardize(ob_), -5., 5.)
            ob = torch.cat([ob_, ob[:, -1].unsqueeze(-1)], dim=-1)
        else:
            ob = torch.clamp(self.rms_obs.standardize(ob), -5., 5.)
        ob_emb = self.ob_encoder(ob)
        if self.hps.state_only:
            score = self.score_head(self.score_trunk(ob_emb))
        else:
            ac_emb = self.ac_encoder(ac)
            score = self.score_head(self.score_trunk(torch.cat([ob_emb, ac_emb], dim=-1)))
        # Careful, no sigmoid here
        aux = self.aux_head(self.aux_trunk(ob_emb))
        return score, aux


class Actor(nn.Module):

    def __init__(self, env, hps):
        super(Actor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.ac_max = env.action_space.high[0]
        self.hps = hps
        # Define observation whitening
        self.rms_obs = RunMoms(shape=env.observation_space.shape, use_mpi=True)
        # Assemble the last layers and output heads
        self.s_encoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, 300)),
                ('ln', nn.LayerNorm(300)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.a_decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(300, 200)),
                ('ln', nn.LayerNorm(200)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.a_head = nn.Linear(200, ac_dim)
        if self.hps.kye_p:
            self.r_decoder = nn.Sequential(OrderedDict([
                ('fc_block_1', nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(300, 200)),
                    ('ln', nn.LayerNorm(200)),
                    ('nl', nn.ReLU()),
                ]))),
            ]))
            self.r_head = nn.Linear(200, 1)
        # Perform initialization
        self.s_encoder.apply(init(weight_scale=math.sqrt(2)))
        self.a_decoder.apply(init(weight_scale=math.sqrt(2)))
        self.a_head.apply(init(weight_scale=0.01))
        if self.hps.kye_p:
            self.r_decoder.apply(init(weight_scale=math.sqrt(2)))
            self.r_head.apply(init(weight_scale=0.01))

    def act(self, ob):
        out = self.forward(ob)
        return out[0]  # ac

    def auxo(self, ob):
        if self.hps.kye_p:
            out = self.forward(ob)
            return out[1]  # aux
        else:
            raise ValueError("should not be called")

    def forward(self, ob):
        ob = torch.clamp(self.rms_obs.standardize(ob), -5., 5.)
        x = self.s_encoder(ob)
        ac = float(self.ac_max) * torch.tanh(self.a_head(self.a_decoder(x)))
        out = [ac]
        if self.hps.kye_p:
            aux = self.r_head(self.r_decoder(x))
            out.append(aux)
        return out

    @property
    def perturbable_params(self):
        return [n for n, _ in self.named_parameters() if 'ln' not in n]

    @property
    def non_perturbable_params(self):
        return [n for n, _ in self.named_parameters() if 'ln' in n]


class Critic(nn.Module):

    def __init__(self, env, hps):
        super(Critic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        if hps.use_c51:
            num_heads = hps.c51_num_atoms
        elif hps.use_qr:
            num_heads = hps.num_tau
        else:
            num_heads = 1
        self.hps = hps
        # Define observation whitening
        self.rms_obs = RunMoms(shape=env.observation_space.shape, use_mpi=True)
        # Assemble the last layers and output heads
        self.c_encoder = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, 400)),
                ('ln', nn.LayerNorm(400)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.c_decoder = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(400, 300)),
                ('ln', nn.LayerNorm(300)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.c_head = nn.Linear(300, num_heads)
        # Perform initialization
        self.c_encoder.apply(init(weight_scale=math.sqrt(2)))
        self.c_decoder.apply(init(weight_scale=math.sqrt(2)))
        self.c_head.apply(init(weight_scale=0.01))

    def QZ(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        ob = torch.clamp(self.rms_obs.standardize(ob), -5., 5.)
        x = torch.cat([ob, ac], dim=-1)
        x = self.c_encoder(x)
        x = self.c_decoder(x)
        x = self.c_head(x)
        if self.hps.use_c51:
            # Return a categorical distribution
            x = F.log_softmax(x, dim=1).exp()
        return x
