import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from torch import autograd
from torch.autograd import Variable


HID_SIZE = 256


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.

def init(nonlin=None, param=None,
         weight_scale=1., constant_bias=0.):
    """Perform orthogonal initialization"""

    def _init(m):

        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
            scale = (nn.init.calculate_gain(nonlin, param)
                     if nonlin is not None
                     else weight_scale)
            nn.init.orthogonal_(m.weight, gain=scale)
            if m.bias is not None:
                nn.init.constant_(m.bias, constant_bias)
        elif (isinstance(m, nn.BatchNorm2d) or
              isinstance(m, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return _init


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Networks.

class Actor(nn.Module):

    def __init__(self, env, hps):
        super(Actor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.ac_max = env.action_space.high[0]
        self.hps = hps
        # Assemble fully-connected encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, HID_SIZE)),
                ('ln', nn.LayerNorm(HID_SIZE)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        # Assemble fully-connected decoder
        self.a_decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(HID_SIZE, HID_SIZE)),
                ('ln', nn.LayerNorm(HID_SIZE)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.a_skip_co = nn.Sequential()
        self.a_head = nn.Linear(HID_SIZE, ac_dim)
        if self.hps.reward_control:
            # > Reward final decoder
            self.r_decoder = nn.Sequential(OrderedDict([
                ('fc_block_1', nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(HID_SIZE, HID_SIZE)),
                    ('ln', nn.LayerNorm(HID_SIZE)),
                    ('nl', nn.ReLU(inplace=True)),
                ]))),
            ]))
            self.r_skip_co = nn.Sequential()
            self.r_head = nn.Linear(HID_SIZE, 1)
        # Perform initialization
        self.encoder.apply(init(nonlin='relu', param=None))
        self.a_decoder.apply(init(nonlin='relu', param=None))
        self.a_head.apply(init(weight_scale=0.01, constant_bias=0.0))
        if self.hps.reward_control:
            self.r_decoder.apply(init(nonlin='relu', param=None))
            self.r_head.apply(init(nonlin='linear', param=None))

    def act(self, ob):
        out = self.forward(ob)
        return out[0]  # ac

    def rc(self, ob):
        if self.hps.reward_control:
            out = self.forward(ob)
            return out[1]  # reward control
        else:
            raise ValueError("should not be called")

    def forward(self, ob):
        x = self.encoder(ob)
        ac = float(self.ac_max) * torch.tanh(self.a_head(self.a_decoder(x) + self.a_skip_co(x)))
        out = [ac]
        if self.hps.reward_control:
            reward = self.r_head(self.r_decoder(x) + self.r_skip_co(x))
            out.append(reward)
        return out

    @property
    def perturbable_params(self):
        return [n for n, _ in self.named_parameters() if 'ln' not in n]

    @property
    def non_perturbable_params(self):
        return [n for n, _ in self.named_parameters() if 'ln' in n]


class VanillaCritic(nn.Module):

    def __init__(self, env, hps):
        super(VanillaCritic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.hps = hps
        # Assemble fully-connected encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, HID_SIZE)),
                ('ln', nn.LayerNorm(HID_SIZE)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        # Assemble fully-connected decoder
        self.q_decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(HID_SIZE, HID_SIZE)),
                ('ln', nn.LayerNorm(HID_SIZE)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.q_skip_co = nn.Sequential()
        self.q_head = nn.Linear(HID_SIZE, 1)
        if self.hps.reward_control:
            # > Reward final decoder
            self.r_decoder = nn.Sequential(OrderedDict([
                ('fc_block_1', nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(HID_SIZE, HID_SIZE)),
                    ('ln', nn.LayerNorm(HID_SIZE)),
                    ('nl', nn.ReLU(inplace=True)),
                ]))),
            ]))
            self.r_skip_co = nn.Sequential()
            self.r_head = nn.Linear(HID_SIZE, 1)
        # Perform initialization
        self.encoder.apply(init(nonlin='relu', param=None))
        self.q_decoder.apply(init(nonlin='relu', param=None))
        self.q_head.apply(init(weight_scale=0.01, constant_bias=0.0))
        if self.hps.reward_control:
            self.r_decoder.apply(init(nonlin='relu', param=None))
            self.r_head.apply(init(nonlin='linear', param=None))

    def Q(self, ob, ac):
        out = self.forward(ob, ac)
        return out[0]  # q

    def rc(self, ob, ac):
        if self.hps.reward_control:
            out = self.forward(ob, ac)
            return out[1]  # reward control
        else:
            raise ValueError("should not be called")

    def forward(self, ob, ac):
        x = self.encoder(torch.cat([ob, ac], dim=-1))
        q = self.q_head(self.q_decoder(x) + self.q_skip_co(x))
        out = [q]
        if self.hps.reward_control:
            reward = self.r_head(self.r_decoder(x) + self.r_skip_co(x))
            out.append(reward)
        return out

    @property
    def out_params(self):
        return [p for p in self.q_head.parameters()]


class C51QRCritic(nn.Module):

    def __init__(self, env, hps):
        """Distributional critic, C51 and QR-DQN version
        based on the value networks used in C51 and QR-DQN
        (C51 paper: http://arxiv.org/abs/1707.06887)
        (QR-DQN paper: http://arxiv.org/abs/1710.10044)
        """
        super(C51QRCritic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        assert sum([hps.use_c51, hps.use_qr]) == 1 and not hps.use_iqn
        num_z_heads = hps.c51_num_atoms if hps.use_c51 else hps.num_tau
        self.hps = hps
        # Assemble fully-connected encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, HID_SIZE)),
                ('ln', nn.LayerNorm(HID_SIZE)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        # Assemble fully-connected decoder
        self.z_decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(HID_SIZE, HID_SIZE)),
                ('ln', nn.LayerNorm(HID_SIZE)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.z_skip_co = nn.Sequential()
        self.z_heads = nn.Linear(HID_SIZE, num_z_heads)
        if self.hps.reward_control:
            # > Reward final decoder
            self.r_decoder = nn.Sequential(OrderedDict([
                ('fc_block_1', nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(HID_SIZE, HID_SIZE)),
                    ('ln', nn.LayerNorm(HID_SIZE)),
                    ('nl', nn.ReLU(inplace=True)),
                ]))),
            ]))
            self.r_skip_co = nn.Sequential()
            self.r_head = nn.Linear(HID_SIZE, 1)
        # Perform initialization
        self.encoder.apply(init(nonlin='relu', param=None))
        self.z_decoder.apply(init(nonlin='relu', param=None))
        self.z_heads.apply(init(weight_scale=0.01, constant_bias=0.0))
        if self.hps.reward_control:
            self.r_decoder.apply(init(nonlin='relu', param=None))
            self.r_head.apply(init(nonlin='linear', param=None))

    def Z(self, ob, ac):
        out = self.forward(ob, ac)
        return out[0]  # z

    def rc(self, ob, ac):
        if self.hps.reward_control:
            out = self.forward(ob, ac)
            return out[1]  # reward control
        else:
            raise ValueError("should not be called")

    def forward(self, ob, ac):
        x = self.encoder(torch.cat([ob, ac], dim=-1))
        z = self.z_heads(self.z_decoder(x) + self.z_skip_co(x))
        if self.hps.use_c51:
            # Return a categorical distribution
            z = F.log_softmax(z, dim=1).exp()
        out = [z]
        if self.hps.reward_control:
            reward = self.r_head(self.r_decoder(x) + self.r_skip_co(x))
            out.append(reward)
        return out


class IQNCritic(nn.Module):

    def __init__(self, env, hps):
        """Distributional critic, IQN version
        (IQN paper: https://arxiv.org/abs/1806.06923)
        """
        super(IQNCritic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.hps = hps
        # > Psi encoder
        # Assemble fully-connected encoder
        self.psi_encoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, HID_SIZE)),
                ('ln', nn.LayerNorm(HID_SIZE)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        # Assemble fully-connected decoder
        self.psi_decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(HID_SIZE, HID_SIZE)),
                ('ln', nn.LayerNorm(HID_SIZE)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        if self.hps.reward_control:
            # > Reward final decoder
            self.r_decoder = nn.Sequential(OrderedDict([
                ('fc_block_1', nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(HID_SIZE, HID_SIZE)),
                    ('ln', nn.LayerNorm(HID_SIZE)),
                    ('nl', nn.ReLU(inplace=True)),
                ]))),
            ]))
            self.r_skip_co = nn.Sequential()
            self.r_head = nn.Linear(HID_SIZE, 1)
        # > Phi encoder
        self.phi_encoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.hps.quantile_emb_dim, HID_SIZE)),
                ('ln', nn.LayerNorm(HID_SIZE)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        # > Hadamard encoder
        self.had_encoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(HID_SIZE, HID_SIZE)),
                ('ln', nn.LayerNorm(HID_SIZE)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.z_head = nn.Linear(HID_SIZE, 1)
        # Perform initialization
        self.psi_encoder.apply(init(nonlin='relu', param=None))
        self.psi_decoder.apply(init(nonlin='relu', param=None))
        self.phi_encoder.apply(init(nonlin='relu', param=None))
        self.had_encoder.apply(init(nonlin='relu', param=None))
        self.z_head.apply(init(weight_scale=0.01, constant_bias=0.0))
        if self.hps.reward_control:
            self.r_decoder.apply(init(nonlin='relu', param=None))
            self.r_head.apply(init(nonlin='linear', param=None))

    def psi_emb(self, ob, ac):
        x = torch.cat([ob, ac], dim=-1)
        x = self.psi_encoder(x)
        x = self.psi_decoder(x)
        return x

    def phi_emb(self, num_quantiles):
        """Equation 4 in IQN paper"""
        tau = torch.FloatTensor(self.hps.batch_size * num_quantiles, 1)
        tau.uniform_(0., 1.)
        # Expand the quantiles, e.g. [tau1, tau2, tau3] tiled with [1, dim]
        # becomes [[tau1, tau2, tau3], [tau1, tau2, tau3], ...]
        x = tau.repeat(1, self.hps.quantile_emb_dim)
        indices = torch.arange(1, self.hps.quantile_emb_dim + 1, dtype=torch.float32).to(tau)
        pi = math.pi * torch.ones(self.hps.quantile_emb_dim, dtype=torch.float32).to(tau)
        assert indices.shape == pi.shape
        x *= torch.mul(indices, pi)
        x = torch.cos(x)
        # Wrap with unique layer
        x = self.phi_encoder(x)
        return x, tau

    def Z(self, ob, ac, num_quantiles):
        out = self.forward(ob, ac, num_quantiles)
        return out[0:2]  # z, tau

    def rc(self, ob, ac, num_quantiles):
        if self.hps.reward_control:
            out = self.forward(ob, ac, num_quantiles)
            return out[2]  # reward control
        else:
            raise ValueError("should not be called")

    def forward(self, ob, ac, num_quantiles):
        # Psi embedding
        psi_emb = self.psi_encoder(torch.cat([ob, ac], dim=-1))
        psi = self.psi_decoder(psi_emb)
        psi = psi.repeat(num_quantiles, 1)
        # Embed tau
        phi, tau = self.phi_emb(num_quantiles)
        assert psi.shape == phi.shape, "{}, {}".format(psi.shape, phi.shape)
        # Multiply the embedding element-wise
        had = self.had_encoder(psi * (1.0 + phi))
        z = F.relu(self.z_head(had))
        z = z.view(-1, num_quantiles, 1)
        out = [z, tau]
        if self.hps.reward_control:
            reward = self.r_head(self.r_decoder(psi_emb) + self.r_skip_co(psi_emb))
            out.append(reward)
        return out


class Discriminator(nn.Module):

    def __init__(self, env, hps):
        super(Discriminator, self).__init__()
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.hps = hps
        # Define the input dimension, depending on whether actions are used too.
        in_dim = self.ob_dim if self.hps.state_only else self.ob_dim + self.ac_dim
        # Assemble the discriminator decoder
        self.decoder = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', U.spectral_norm(nn.Linear(in_dim, HID_SIZE))),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', U.spectral_norm(nn.Linear(HID_SIZE, HID_SIZE))),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.score_head = nn.Linear(HID_SIZE, 1)
        # Perform initialization
        self.decoder.apply(init(nonlin='leaky_relu', param=0.1))
        self.score_head.apply(init(nonlin='linear', param=None))

    def grad_pen(self, p_ob, p_ac, e_ob, e_ac):
        """Gradient penalty regularizer (motivation from Wasserstein GANs (Gulrajani),
        but empirically useful in JS-GANs (Lucic et al. 2017)) and later in (Karol et al. 2018).
        """
        # Retrieve device from either input tensor
        device = p_ob.device
        # Assemble interpolated state-action pair
        ob_eps = torch.rand(self.ob_dim).to(device)
        ac_eps = torch.rand(self.ac_dim).to(device)
        ob_interp = ob_eps * p_ob + ((1. - ob_eps) * e_ob)
        ac_interp = ac_eps * p_ac + ((1. - ac_eps) * e_ac)
        # Set `requires_grad=True` to later have access to
        # gradients w.r.t. the inputs (not populated by default)
        ob_interp = Variable(ob_interp, requires_grad=True)
        ac_interp = Variable(ac_interp, requires_grad=True)
        # Create the operation of interest
        score = self.forward(ob_interp, ac_interp)
        # Get the gradient of this operation with respect to its inputs
        grads = autograd.grad(outputs=score,
                              inputs=[ob_interp, ac_interp],
                              only_inputs=True,
                              grad_outputs=torch.ones(score.size()).to(device),
                              retain_graph=True,
                              create_graph=True,
                              allow_unused=self.hps.state_only)
        assert len(list(grads)) == 2, "length must be exactly 2"
        # Return the gradient penalty (try to induce 1-Lipschitzness)
        if self.hps.state_only:
            grads = grads[0]
        grads_concat = torch.cat(list(grads), dim=-1)
        return (grads_concat.norm(2, dim=-1) - 1.).pow(2).mean()

    def get_reward(self, ob, ac):
        """Craft surrogate reward"""
        assert sum([isinstance(x, torch.Tensor) for x in [ob, ac]]) in [0, 2]
        if not isinstance(ob, torch.Tensor):  # then ac is not neither
            ob = torch.FloatTensor(ob)
            ac = torch.FloatTensor(ac)
        # Transfer to cpu
        ob = ob.cpu()
        ac = ac.cpu()

        # Counterpart of GAN's minimax (also called "saturating") loss
        # Numerics: 0 for non-expert-like states, goes to +inf for expert-like states
        # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
        # e.g. walking simulations that get cut off when the robot falls over
        minimax_reward = -torch.log(1. - torch.sigmoid(self.forward(ob, ac).detach()) + 1e-8)

        if self.hps.minimax_only:
            return minimax_reward
        else:
            # Counterpart of GAN's non-saturating loss
            # Recommended in the original GAN paper and later in (Fedus et al. 2017)
            # Numerics: 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            non_satur_reward = torch.log(torch.sigmoid(self.forward(ob, ac).detach()))
            # Return the sum the two previous reward functions (as in AIRL, Fu et al. 2018)
            # Numerics: might be better might be way worse
            return non_satur_reward + minimax_reward

    def forward(self, ob, ac):
        x = ob if self.hps.state_only else torch.cat([ob, ac], dim=-1)
        return self.score_head(self.decoder(x))
