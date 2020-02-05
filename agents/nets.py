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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Models.

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
                ('fc', U.spectral_norm(nn.Linear(in_dim, 100))),
                ('nl', nn.LeakyReLU(negative_slope=self.leak, inplace=True)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', U.spectral_norm(nn.Linear(100, 100))),
                ('nl', nn.LeakyReLU(negative_slope=self.leak, inplace=True)),
            ]))),
        ]))
        self.score_head = nn.Linear(100, 1)
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
            non_satur_reward = F.logsigmoid(self.D(ob, ac).detach())
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


class Actor(nn.Module):

    def __init__(self, env, hps):
        super(Actor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.ac_max = env.action_space.high[0]
        self.hps = hps
        self.s_encoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.a_decoder = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(256, 256)),
                ('ln', nn.LayerNorm(256)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.a_skip_co = nn.Sequential()
        self.a_head = nn.Linear(256, ac_dim)
        if self.hps.s2r2:
            self.r_decoder = nn.Sequential(OrderedDict([
                ('fc_block_1', nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(256, 256)),
                    ('ln', nn.LayerNorm(256)),
                    ('nl', nn.ReLU(inplace=True)),
                ]))),
            ]))
            self.r_skip_co = nn.Sequential()
            self.r_head = nn.Linear(256, 1)
        # Perform initialization
        self.s_encoder.apply(init(nonlin='relu', param=None))
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
        x = self.s_encoder(ob)
        ac = float(self.ac_max) * torch.tanh(self.a_head(self.a_decoder(x) + self.a_skip_co(x)))
        out = [ac]
        if self.hps.s2r2:
            reward = self.r_head(self.r_decoder(x) + self.r_skip_co(x))
            out.append(reward)
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
        self.c_encoder = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, 400)),
                ('ln', nn.LayerNorm(400)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.c_decoder = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(400, 300)),
                ('ln', nn.LayerNorm(300)),
                ('nl', nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.c_head = nn.Linear(300, num_heads)
        # Perform initialization
        self.c_encoder.apply(init(nonlin='relu', param=None))
        self.c_decoder.apply(init(nonlin='relu', param=None))
        self.c_head.apply(init(weight_scale=0.01, constant_bias=0.0))

    def QZ(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        x = torch.cat([ob, ac], dim=-1)
        x = self.c_encoder(x)
        x = self.c_decoder(x)
        x = self.c_head(x)
        if self.hps.use_c51:
            # Return a categorical distribution
            x = F.log_softmax(x, dim=1).exp()
        return x

    @property
    def out_params(self):
        return [p for p in self.c_head.parameters()]
