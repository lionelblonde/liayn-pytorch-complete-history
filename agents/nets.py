import math

import torch
import torch.nn as nn
import torch.nn.modules.rnn as rnn
import torch.nn.functional as F
from torch import autograd

from helpers.spectral_norm import SNLinear


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.

def ortho_init(module, nonlinearity=None, weight_scale=1.0, constant_bias=0.0):
    """Applies orthogonal initialization for the parameters of a given module"""

    if nonlinearity is not None:
        gain = nn.init.calculate_gain(nonlinearity)
    else:
        gain = weight_scale

    if isinstance(module, (nn.RNNBase, rnn.RNNCellBase)):
        for name, param in module.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias_' in name:
                nn.init.constant_(param, constant_bias)
    else:  # other modules with single .weight and .bias
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, constant_bias)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Networks.

class Actor(nn.Module):

    def __init__(self, env, hps):
        super(Actor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.ac_max = env.action_space.high[0]
        # Define feature extractor
        self.fc_1 = nn.Linear(ob_dim, 64)
        self.fc_2 = nn.Linear(64, 64)
        ortho_init(self.fc_1, nonlinearity='leaky_relu', constant_bias=0.0)
        ortho_init(self.fc_2, nonlinearity='leaky_relu', constant_bias=0.0)
        # Define action head
        self.ac_head = nn.Linear(64, ac_dim)
        ortho_init(self.ac_head, weight_scale=0.01, constant_bias=0.0)
        # Determine which parameters are perturbable
        self.perturbable_params = [p for p in self.state_dict()
                                   if 'layer_norm' not in p]
        self.nonperturbable_params = [p for p in self.state_dict()
                                      if 'layer_norm' in p]
        assert (set(self.perturbable_params + self.nonperturbable_params) ==
                set(self.state_dict().keys()))
        # Following the paper 'Parameter Space Noise for Exploration', we do not
        # perturb the conv2d layers, only the fully-connected part of the network.
        # Additionally, the extra variables introduced by layer normalization should remain
        # unperturbed as they do not play any role in exploration.
        self.ln_1 = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x
        self.ln_2 = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x

    def forward(self, ob):
        plop = ob
        # Stack fully-connected layers
        plop = F.leaky_relu(self.ln_1(self.fc_1(plop)))
        plop = F.leaky_relu(self.ln_2(self.fc_2(plop)))
        ac = float(self.ac_max) * torch.tanh(self.ac_head(plop))
        return ac


class Critic(nn.Module):

    def __init__(self, env, hps):
        super(Critic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        # Create fully-connected layers
        self.fc_1 = nn.Linear(ob_dim + ac_dim, 64)
        self.fc_2 = nn.Linear(64, 64)
        ortho_init(self.fc_1, nonlinearity='leaky_relu', constant_bias=0.0)
        ortho_init(self.fc_2, nonlinearity='leaky_relu', constant_bias=0.0)
        # Define Q head
        self.q_head = nn.Linear(64, 1)
        ortho_init(self.q_head, weight_scale=0.01, constant_bias=0.0)
        # Define layernorm layers
        self.ln_1 = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x
        self.ln_2 = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x

    def Q(self, ob, ac):
        # Concatenate observations and actions
        plop = torch.cat([ob, ac], dim=-1)
        # Stack fully-connected layers
        plop = F.leaky_relu(self.ln_1(self.fc_1(plop)))
        plop = F.leaky_relu(self.ln_2(self.fc_2(plop)))
        q = self.q_head(plop)
        return q

    def forward(self, ob, ac):
        q = self.Q(ob, ac)
        return q


class QuantileCritic(nn.Module):

    def __init__(self, env, hps):
        """Distributional critic, based on IQNs
        (IQN paper: https://arxiv.org/abs/1806.06923)
        """
        super(QuantileCritic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.quantile_emb_dim = hps.quantile_emb_dim
        emb_out_dim = 64

        # Define embedding layers
        self.psi_fc_1 = nn.Linear(ob_dim + ac_dim, 64)
        self.psi_fc_2 = nn.Linear(64, emb_out_dim)
        self.phi_fc = nn.Linear(self.quantile_emb_dim, emb_out_dim)
        self.hadamard_fc = nn.Linear(emb_out_dim, 64)
        ortho_init(self.psi_fc_1, nonlinearity='leaky_relu', constant_bias=0.0)
        ortho_init(self.psi_fc_2, nonlinearity='leaky_relu', constant_bias=0.0)
        ortho_init(self.phi_fc, nonlinearity='relu', constant_bias=0.0)
        ortho_init(self.hadamard_fc, nonlinearity='relu', constant_bias=0.0)

        # Define Z head
        self.z_head = nn.Linear(64, 1)
        ortho_init(self.z_head, weight_scale=0.01, constant_bias=0.0)

        # Define layernorm layers
        self.ln_psi_1 = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x
        self.ln_psi_2 = nn.LayerNorm(emb_out_dim) if hps.with_layernorm else lambda x: x
        self.ln_phi = nn.LayerNorm(emb_out_dim) if hps.with_layernorm else lambda x: x
        self.ln_hadamard = nn.LayerNorm(64) if hps.with_layernorm else lambda x: x

    def psi_emb(self, ob, ac):
        plop = torch.cat([ob, ac], dim=-1)
        plop = F.leaky_relu(self.ln_psi_1(self.psi_fc_1(plop)))
        plop = F.leaky_relu(self.ln_psi_2(self.psi_fc_2(plop)))
        return plop

    def phi_emb(self, quantiles):
        """Equation 4 in IQN paper"""
        # Retrieve device from quantiles tensor
        device = quantiles.device
        # Reshape
        batch_size, num_quantiles = list(quantiles.shape)[:-1]
        quantiles = quantiles.view(batch_size * num_quantiles, 1)
        # Expand the quantiles, e.g. [tau1, tau2, tau3] tiled with [1, dim]
        # becomes [[tau1, tau2, tau3], [tau1, tau2, tau3], ...]
        plop = quantiles.repeat(1, self.quantile_emb_dim)
        indices = torch.arange(1, self.quantile_emb_dim + 1, dtype=torch.float32).to(device)
        pi = math.pi * torch.ones(self.quantile_emb_dim, dtype=torch.float32).to(device)
        assert indices.shape == pi.shape
        plop *= torch.mul(indices, pi)
        plop = torch.cos(plop)
        # Wrap with unique layer
        plop = F.relu(self.ln_phi(self.phi_fc(plop)))
        return plop

    def Z(self, ob, ac, quantiles):
        # Embed state and action
        psi = self.psi_emb(ob, ac)
        batch_size, num_quantiles = list(quantiles.shape)[:-1]
        psi = psi.repeat(num_quantiles, 1)
        # Embed quantiles
        phi = self.phi_emb(quantiles)

        assert psi.shape == phi.shape, "{}, {}".format(psi.shape, phi.shape)

        # Multiply the embedding element-wise
        hadamard = psi * (1.0 + phi)
        hadamard = F.relu(self.ln_hadamard(self.hadamard_fc(hadamard)))
        z = F.relu(self.z_head(hadamard))
        z = z.view(batch_size, num_quantiles, 1)
        return z

    def forward(self, ob, ac, quantiles):
        z = self.Z(ob, ac, quantiles)
        return z

    def Q(self, ob, ac, quantiles):
        batch_size, num_quantiles = list(quantiles.shape)[:-1]
        z = self.Z(ob, ac, quantiles).view(batch_size, num_quantiles)
        q = z.mean(dim=-1).view(batch_size, 1)
        return q


class Discriminator(nn.Module):

    def __init__(self, env, hps):
        super(Discriminator, self).__init__()
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.hps = hps

        if self.hps.state_only:
            in_dim = self.ob_dim
        else:
            in_dim = self.ob_dim + self.ac_dim

        # Define hidden layers
        self.fc_1 = SNLinear(in_dim, 64)
        self.fc_2 = SNLinear(64, 64)
        ortho_init(self.fc_1, activ='leaky_relu', constant_bias=0.0)
        ortho_init(self.fc_2, activ='leaky_relu', constant_bias=0.0)

        # Define layernorm layers
        self.ln_1 = nn.LayerNorm(64) if self.hps.with_layernorm else lambda x: x
        self.ln_2 = nn.LayerNorm(64) if self.hps.with_layernorm else lambda x: x

        # Define score head
        self.score_head = SNLinear(64, 1)
        ortho_init(self.score_head, activ='linear', constant_bias=0.0)

    def get_grad_pen(self, p_ob, p_ac, e_ob, e_ac, lambda_=10):
        """Add a gradient penalty (motivation from WGANs (Gulrajani),
        but empirically useful in JS-GANs (Lucic et al. 2017)) and later
        in (Karol et al. 2018)
        """
        # Retrieve device from either input tensor
        device = p_ob.device
        # Assemble interpolated state-action pair
        ob_eps = torch.rand(self.ob_dim).to(device)
        ac_eps = torch.rand(self.ac_dim).to(device)
        ob_interp = ob_eps * p_ob + (1. - ob_eps) * e_ob
        ac_interp = ac_eps * p_ac + (1. - ac_eps) * e_ac
        # Set `requires_grad=True` to later have access to
        # gradients w.r.t. the inputs (not populated by default)
        ob_interp.requires_grad = True
        ac_interp.requires_grad = True
        # Create the operation of interest
        score = self.D(ob_interp, ac_interp)
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
        return lambda_ * (grads_concat.norm(2, dim=-1) - 1.).pow(2).mean()

    def get_reward(self, ob, ac):
        """Craft surrogate reward"""
        ob = torch.FloatTensor(ob).cpu()
        ac = torch.FloatTensor(ac).cpu()

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
        if self.hps.state_only:
            plop = ob
        else:
            plop = torch.cat([ob, ac], dim=-1)
        # Add hidden layers
        plop = F.leaky_relu(self.ln_1(self.fc_1(plop)))
        plop = F.leaky_relu(self.ln_2(self.fc_2(plop)))
        # Add output layer
        score = self.score_head(plop)
        return score

    def forward(self, ob, ac):
        score = self.D(ob, ac)
        return score
