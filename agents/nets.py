import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from torch import autograd
from torch.autograd import Variable


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

class Discriminator(nn.Module):

    def __init__(self, env, hps):
        super(Discriminator, self).__init__()
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.hps = hps
        self.leak = 0.1
        # Define the input dimension, depending on whether actions are used too.
        in_dim = self.ob_dim if self.hps.state_only else self.ob_dim + self.ac_dim
        self.score_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', U.spectral_norm(nn.Linear(in_dim, 256))),
                ('nl', nn.LeakyReLU(negative_slope=self.leak, inplace=True)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', U.spectral_norm(nn.Linear(256, 256))),
                ('nl', nn.LeakyReLU(negative_slope=self.leak, inplace=True)),
            ]))),
        ]))
        self.score_head = nn.Linear(256, 1)
        # Perform initialization
        self.score_trunk.apply(init(nonlin='leaky_relu', param=self.leak))
        self.score_head.apply(init(weight_scale=0.01, constant_bias=0.0))

    def grad_pen(self, p_ob, p_ac, e_ob, e_ac):
        """Gradient penalty regularizer (motivation from Wasserstein GANs (Gulrajani),
        but empirically useful in JS-GANs (Lucic et al. 2017)) and later in (Karol et al. 2018).
        """
        # Assemble interpolated state-action pair
        ob_eps = torch.rand(self.ob_dim).to(p_ob.device)
        ac_eps = torch.rand(self.ac_dim).to(p_ob.device)
        ob_interp = ob_eps * p_ob + ((1. - ob_eps) * e_ob)
        ac_interp = ac_eps * p_ac + ((1. - ac_eps) * e_ac)
        # Set `requires_grad=True` to later have access to
        # gradients w.r.t. the inputs (not populated by default)
        ob_interp = Variable(ob_interp, requires_grad=True)
        ac_interp = Variable(ac_interp, requires_grad=True)
        # Create the operation of interest
        score = self.D(ob_interp, ac_interp)
        # Get the gradient of this operation with respect to its inputs
        grads = autograd.grad(outputs=score,
                              inputs=[ob_interp, ac_interp],
                              only_inputs=True,
                              grad_outputs=torch.ones(score.size()).to(p_ob.device),
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
        minimax_reward = -torch.log(1. - torch.sigmoid(self.D(ob, ac).detach()) + 1e-8)

        if self.hps.minimax_only:
            return minimax_reward
        else:
            # Counterpart of GAN's non-saturating loss
            # Recommended in the original GAN paper and later in (Fedus et al. 2017)
            # Numerics: 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            non_satur_reward = torch.log(torch.sigmoid(self.D(ob, ac).detach()))
            # Return the sum the two previous reward functions (as in AIRL, Fu et al. 2018)
            # Numerics: might be better might be way worse
            return non_satur_reward + minimax_reward

    def D(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        x = ob if self.hps.state_only else torch.cat([ob, ac], dim=-1)
        x = self.score_trunk(x)
        score = self.score_head(x)
        return score


class MinimalDiscriminator(nn.Module):

    def __init__(self, hps):
        super(MinimalDiscriminator, self).__init__()
        self.hps = hps
        self.leak = 0.1
        self.score_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', U.spectral_norm(nn.Linear(256, 256))),
                ('nl', nn.LeakyReLU(negative_slope=self.leak, inplace=True)),
            ]))),
        ]))
        self.score_head = nn.Linear(256, 1)
        # Perform initialization
        # self.score_trunk.apply(init(nonlin='leaky_relu', param=self.leak))
        self.score_head.apply(init(weight_scale=0.01, constant_bias=0.0))

    def grad_pen(self, p_x, e_x):
        """Gradient penalty regularizer (motivation from Wasserstein GANs (Gulrajani),
        but empirically useful in JS-GANs (Lucic et al. 2017)) and later in (Karol et al. 2018).
        """
        # Assemble interpolated state-action pair
        x_eps = torch.rand(p_x.size(-1)).to(p_x.device)
        x_interp = x_eps * p_x + ((1. - x_eps) * e_x)
        # Set `requires_grad=True` to later have access to
        # gradients w.r.t. the inputs (not populated by default)
        x_interp = Variable(x_interp, requires_grad=True)
        # Create the operation of interest
        score = self.D(x_interp)
        # Get the gradient of this operation with respect to its inputs
        grads = autograd.grad(outputs=score,
                              inputs=[x_interp],
                              only_inputs=True,
                              grad_outputs=torch.ones(score.size()).to(p_x.device),
                              retain_graph=True,
                              create_graph=True,
                              allow_unused=False)
        assert len(list(grads)) == 1, "length must be exactly 1"
        grad = grads[0]  # tuple of size 1, so extract the only element
        # Return the gradient penalty (try to induce 1-Lipschitzness)
        return (grad.norm(2, dim=-1) - 1.).pow(2).mean()

    def get_reward(self, x):
        """Craft surrogate reward"""
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        # Transfer to cpu
        x = x.cpu()

        # Counterpart of GAN's minimax (also called "saturating") loss
        # Numerics: 0 for non-expert-like states, goes to +inf for expert-like states
        # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
        # e.g. walking simulations that get cut off when the robot falls over
        minimax_reward = -torch.log(1. - torch.sigmoid(self.D(x).detach()) + 1e-8)

        if self.hps.minimax_only:
            return minimax_reward
        else:
            # Counterpart of GAN's non-saturating loss
            # Recommended in the original GAN paper and later in (Fedus et al. 2017)
            # Numerics: 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            non_satur_reward = torch.log(torch.sigmoid(self.D(x).detach()))
            # Return the sum the two previous reward functions (as in AIRL, Fu et al. 2018)
            # Numerics: might be better might be way worse
            return non_satur_reward + minimax_reward

    def D(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.score_trunk(x)
        score = self.score_head(x)
        return score


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
                ('fc', nn.Linear(ob_dim, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        # Assemble fully-connected decoder
        self.a_decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(256, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.a_head = nn.Linear(256, ac_dim)
        if self.hps.s2r2:
            # > Reward final decoder
            self.r_decoder = nn.Sequential(OrderedDict([
                ('fc_block_1', nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(256, 256)),
                    ('ln', nn.LayerNorm(256)),
                    ('nl', nn.ReLU(inplace=True)),
                ]))),
            ]))
            self.r_head = nn.Linear(256, 1)
        # Perform initialization
        self.encoder.apply(init(nonlin='relu', param=None))
        self.a_decoder.apply(init(nonlin='relu', param=None))
        self.a_head.apply(init(weight_scale=0.01, constant_bias=0.0))
        if self.hps.s2r2:
            self.r_decoder.apply(init(nonlin='relu', param=None))
            self.r_head.apply(init(weight_scale=0.01, constant_bias=0.0))

    def act(self, ob):
        out = self.forward(ob)
        return out[0]  # ac

    def s2r2(self, ob):
        if self.hps.s2r2:
            out = self.forward(ob)
            return out[1]  # reward
        else:
            raise ValueError("should not be called")

    def forward(self, ob):
        x = self.encoder(ob)
        ac = float(self.ac_max) * torch.tanh(self.a_head(self.a_decoder(x)))
        out = [ac]
        if self.hps.s2r2:
            reward = self.r_head(self.r_decoder(x))
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
        self.q_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(256, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))

        self.q_head = nn.Linear(256, 1)
        # Perform initialization
        self.q_trunk.apply(init(nonlin='relu', param=None))
        self.q_head.apply(init(weight_scale=0.01, constant_bias=0.0))

    def Q(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        x = torch.cat([ob, ac], dim=-1)
        x = self.q_trunk(x)
        x = self.q_head(x)
        return x

    @property
    def out_params(self):
        return [p for p in self.q_head.parameters()]


class MinimalVanillaCritic(nn.Module):

    def __init__(self, env, hps):
        super(MinimalVanillaCritic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.hps = hps
        self.qs_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.qa_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ac_dim, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.q_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(256, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.q_head = nn.Linear(256, 1)
        # Perform initialization
        self.qs_trunk.apply(init(nonlin='relu', param=None))
        self.qa_trunk.apply(init(nonlin='relu', param=None))
        self.q_trunk.apply(init(nonlin='relu', param=None))
        self.q_head.apply(init(weight_scale=0.01, constant_bias=0.0))

    def encode(self, ob, ac):
        assert sum([isinstance(x, torch.Tensor) for x in [ob, ac]]) in [0, 2]
        if not isinstance(ob, torch.Tensor):  # then ac is not neither
            ob = torch.FloatTensor(ob)
            ac = torch.FloatTensor(ac)
        # Transfer to cpu
        ob = ob.cpu()
        ac = ac.cpu()
        # Encode
        if self.hps.state_only:
            return self.qs_trunk(ob).detach()
        else:
            return (self.qs_trunk(ob) +
                    self.qa_trunk(ac)).detach()

    def Q(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        xs = self.qs_trunk(ob)
        xa = self.qa_trunk(ac)
        x = self.q_trunk(xs + xa)
        x = self.q_head(x)
        return x

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
        self.z_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(256, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.z_head = nn.Linear(256, num_z_heads)
        # Perform initialization
        self.z_trunk.apply(init(nonlin='relu', param=None))
        self.z_head.apply(init(weight_scale=0.01, constant_bias=0.0))

    def Z(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        x = torch.cat([ob, ac], dim=-1)
        x = self.z_trunk(x)
        z = self.z_head(x)
        if self.hps.use_c51:
            # Return a categorical distribution
            z = F.log_softmax(z, dim=1).exp()
        return z


class MinimalC51QRCritic(nn.Module):

    def __init__(self, env, hps):
        """Distributional critic, C51 and QR-DQN version
        based on the value networks used in C51 and QR-DQN
        (C51 paper: http://arxiv.org/abs/1707.06887)
        (QR-DQN paper: http://arxiv.org/abs/1710.10044)
        """
        super(MinimalC51QRCritic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        assert sum([hps.use_c51, hps.use_qr]) == 1 and not hps.use_iqn
        num_z_heads = hps.c51_num_atoms if hps.use_c51 else hps.num_tau
        self.hps = hps
        self.zs_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.za_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ac_dim, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.z_trunk = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(256, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.z_head = nn.Linear(256, num_z_heads)
        # Perform initialization
        self.zs_trunk.apply(init(nonlin='relu', param=None))
        self.za_trunk.apply(init(nonlin='relu', param=None))
        self.z_trunk.apply(init(nonlin='relu', param=None))
        self.z_head.apply(init(weight_scale=0.01, constant_bias=0.0))

    def encode(self, ob, ac):
        assert sum([isinstance(x, torch.Tensor) for x in [ob, ac]]) in [0, 2]
        if not isinstance(ob, torch.Tensor):  # then ac is not neither
            ob = torch.FloatTensor(ob)
            ac = torch.FloatTensor(ac)
        # Transfer to cpu
        ob = ob.cpu()
        ac = ac.cpu()
        # Encode
        if self.hps.state_only:
            return self.zs_trunk(ob).detach()
        else:
            return (self.z_trunk(self.zs_trunk(ob) +
                                 self.za_trunk(ac))).detach()

    def Z(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        zs = self.zs_trunk(ob)
        za = self.za_trunk(ac)
        z = self.z_trunk(zs + za)
        z = self.z_head(z)
        if self.hps.use_c51:
            # Return a categorical distribution
            z = F.log_softmax(z, dim=1).exp()
        return z


class IQNCritic(nn.Module):

    def __init__(self, env, hps):
        """Distributional critic, IQN version
        (IQN paper: https://arxiv.org/abs/1806.06923)
        """
        super(IQNCritic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.hps = hps
        self.psi = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(256, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.phi = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.hps.quantile_emb_dim, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.hadamard = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(256, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.z_head = nn.Linear(256, 1)
        # Perform initialization
        self.psi.apply(init(nonlin='relu', param=None))
        self.phi.apply(init(nonlin='relu', param=None))
        self.hadamard.apply(init(nonlin='relu', param=None))
        self.z_head.apply(init(weight_scale=0.01, constant_bias=0.0))

    def Z(self, num_quantiles, ob, ac):
        return self.forward(num_quantiles, ob, ac)

    def forward(self, num_quantiles, ob, ac):
        # Psi embedding
        psi = self.psi(torch.cat([ob, ac], dim=-1))
        psi = psi.repeat(num_quantiles, 1)
        # Tau embedding, equation 4 in IQN paper
        # Create tau and populate with unit uniform noise
        tau = torch.FloatTensor(self.hps.batch_size * num_quantiles, 1)
        tau.uniform_(0., 1.)
        # Expand the quantiles, e.g. [tau1, tau2, tau3] tiled with [1, dim]
        # becomes [[tau1, tau2, tau3], [tau1, tau2, tau3], ...]
        emb = tau.repeat(1, self.hps.quantile_emb_dim)
        # Craft the embedding used in the IQN paper
        indices = torch.arange(1, self.hps.quantile_emb_dim + 1, dtype=torch.float32).to(tau)
        pi = math.pi * torch.ones(self.hps.quantile_emb_dim, dtype=torch.float32).to(tau)
        assert indices.shape == pi.shape
        emb *= torch.mul(indices, pi)
        emb = torch.cos(emb)
        # Pass through the phi net
        phi = self.phi(emb)
        # Verify that the shapes of the embeddings are compatible
        assert psi.shape == phi.shape, "{}, {}".format(psi.shape, phi.shape)
        # Multiply the embedding element-wise (hadamard product)
        had = psi * (1.0 + phi)
        had = self.hadamard(had)
        z = F.relu(self.z_head(had))
        z = z.view(-1, num_quantiles, 1)
        return z, tau
