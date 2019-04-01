import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as U
import torch.nn.functional as F
import torch.distributions as D

from algorithms.helpers import logger
from algorithms.helpers.console_util import log_module_info
from algorithms.helpers.distributed_util import average_gradients, sync_with_root
from algorithms.ddpg.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer


def huber_quant_reg_loss(td_errors, quantile, kappa=1.0):
    """Huber regression loss (introduced in 1964) following the definition
    in section 2.3 in the IQN paper (https://arxiv.org/abs/1806.06923).
    The loss involves a disjunction of 2 cases:
        case one: |td_errors| <= kappa
        case two: |td_errors| > kappa
    """
    aux = (0.5 * td_errors ** 2 * (torch.abs(td_errors) <= kappa).float() +
           kappa * (torch.abs(td_errors) - (0.5 * kappa)) * (torch.abs(td_errors) > kappa).float())
    return torch.abs(quantile - ((td_errors < 0).float()).detach()) * aux / kappa


class Actor(nn.Module):

    def __init__(self, ob_space, ac_space, hps):
        super(Actor, self).__init__()
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.ob_dim = self.ob_space.shape[0]
        self.ac_dim = self.ac_space.shape[0]
        self.hps = hps

        self.hid_fc_1 = nn.Linear(self.ob_dim, 300)
        if self.hps.with_layernorm:
            self.layer_norm_1 = nn.LayerNorm(300)
        self.hid_fc_2 = nn.Linear(300, 200)
        if self.hps.with_layernorm:
            self.layer_norm_2 = nn.LayerNorm(200)
        self.out_fc_3 = nn.Linear(200, self.ac_dim)

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

    def forward(self, ob):
        plop = ob
        plop = F.leaky_relu(self.hid_fc_1(plop))
        if self.hps.with_layernorm:
            plop = self.layer_norm_1(plop)
        plop = F.leaky_relu(self.hid_fc_2(plop))
        if self.hps.with_layernorm:
            plop = self.layer_norm_2(plop)
        ac = float(self.ac_space.high[0]) * torch.tanh(self.out_fc_3(plop))
        return ac


class Critic(nn.Module):

    def __init__(self, ob_space, ac_space, hps):
        """Distributional critic, based on IQNs
        (IQN paper: https://arxiv.org/abs/1806.06923)
        """
        super(Critic, self).__init__()
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.ob_dim = self.ob_space.shape[0]
        self.ac_dim = self.ac_space.shape[0]
        self.hps = hps

        self.uniform = D.uniform.Uniform(0., 1.)

        self.emb_out_dim = 64
        # self.psi_hid_fc_1 = nn.Linear(self.ob_dim + self.ac_dim, 128)
        # self.psi_hid_fc_2 = nn.Linear(128, self.emb_out_dim)
        self.psi_hid_fc = nn.Linear(self.ob_dim + self.ac_dim, self.emb_out_dim)

        self.phi_hid_fc = nn.Linear(self.hps.quantile_emb_dim, self.emb_out_dim)

        self.hadamard_hid_fc = nn.Linear(self.emb_out_dim, 64)
        self.out_fc = nn.Linear(64, 1)

    def psi_embedding(self, ob, ac):
        embedding = torch.cat([ob, ac], dim=-1)
        # embedding = F.leaky_relu(self.psi_hid_fc_1(embedding))
        # embedding = F.leaky_relu(self.psi_hid_fc_2(embedding))
        embedding = F.leaky_relu(self.psi_hid_fc(embedding))
        return embedding

    def phi_embedding(self, num_quantiles, out_dim, batch_size, device):
        """Equation 4 in IQN paper"""

        shape = [batch_size, num_quantiles, 1]

        # Generate quantiles (`self` used to grab them from outside the class)
        quantiles = self.uniform.rsample(sample_shape=shape).to(device)

        self.sampled_quantiles = quantiles  # grabbable from outside

        # Reshape
        quantiles = quantiles.view(batch_size * num_quantiles, 1)
        # Expand the quantiles, e.g. [tau1, tau2, tau3] tiled with [1, dim]
        # becomes [[tau1, tau2, tau3], [tau1, tau2, tau3], ...]
        embedding = quantiles.repeat(1, self.hps.quantile_emb_dim)
        indices = torch.arange(1, self.hps.quantile_emb_dim + 1, dtype=torch.float32).to(device)
        pi = math.pi * torch.ones(self.hps.quantile_emb_dim, dtype=torch.float32).to(device)
        embedding *= torch.mul(indices, pi)
        embedding = torch.cos(embedding)
        # Wrap with unique layer
        embedding = F.relu(self.phi_hid_fc(embedding))
        return embedding

    def Z(self, ob, ac, num_quantiles, device):
        # Embed state and action
        psi = self.psi_embedding(ob, ac)
        psi = psi.repeat(num_quantiles, 1)
        # Embed quantiles
        batch_size = ob.shape[0]
        out_dim = ob.shape[-1]
        phi = self.phi_embedding(num_quantiles, out_dim, batch_size, device)
        # Multiply the embedding element-wise
        hadamard = psi * (1.0 + phi)
        hadamard = F.relu(self.hadamard_hid_fc(hadamard))
        z = F.relu(self.out_fc(hadamard))
        z = z.view(batch_size, num_quantiles, 1)
        return z

    def forward(self, ob, ac, num_quantiles, device):
        z = self.Z(ob, ac, num_quantiles, device)
        return z

    def Q(self, ob, ac, num_quantiles, device):
        batch_size = ob.shape[0]
        z = self.Z(ob, ac, num_quantiles, device).view(batch_size, num_quantiles)
        q = z.mean(dim=-1).view(batch_size, 1)
        return q


class EvadeAgent(object):

    def __init__(self, env, device, hps, comm):
        self.env = env
        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape
        self.ob_dim = self.ob_shape[-1]  # num dims
        self.ac_dim = self.ac_shape[-1]  # num dims

        self.device = device

        self.hps = hps
        assert self.hps.n > 1 or not self.hps.n_step_returns

        self.comm = comm

        # Define action clipping range
        assert all(self.ac_space.low == -self.ac_space.high)
        self.max_ac = self.ac_space.high[0].astype('float32')
        assert all(ac_comp == self.max_ac for ac_comp in self.ac_space.high)

        # Parse the noise types
        self.param_noise, self.ac_noise = self.parse_noise_type(self.hps.noise_type)

        # Create online and target nets, and initilize the target nets
        self.actor = Actor(self.ob_space, self.ac_space, self.hps).to(self.device)
        sync_with_root(self.actor, comm)
        self.targ_actor = Actor(self.ob_space, self.ac_space, self.hps).to(self.device)
        self.targ_actor.load_state_dict(self.actor.state_dict())
        self.critic = Critic(self.ob_space, self.ac_space, self.hps).to(self.device)
        sync_with_root(self.critic, comm)
        self.targ_critic = Critic(self.ob_space, self.ac_space, self.hps).to(self.device)
        self.targ_critic.load_state_dict(self.critic.state_dict())
        if self.hps.enable_clipped_double:
            # Create second ('twin') critic and target critic
            # TD3, https://arxiv.org/abs/1802.09477
            self.twin_critic = Critic(self.ob_space, self.ac_space, self.hps).to(self.device)
            self.targ_twin_critic = Critic(self.ob_space, self.ac_space, self.hps).to(self.device)
            self.targ_twin_critic.load_state_dict(self.targ_twin_critic.state_dict())

        if self.param_noise is not None:
            # Create parameter-noise-perturbed ('pnp') actor
            self.pnp_actor = Actor(self.ob_space, self.ac_space, self.hps).to(self.device)
            self.pnp_actor.load_state_dict(self.actor.state_dict())
            # Create adaptive-parameter-noise-perturbed ('apnp') actor
            self.apnp_actor = Actor(self.ob_space, self.ac_space, self.hps).to(self.device)
            self.apnp_actor.load_state_dict(self.actor.state_dict())

        # Set up replay buffer
        self.setup_replay_buffer()

        # Set up the optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.hps.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.hps.critic_lr,
                                                 weight_decay=self.hps.wd_scale)
        if self.hps.enable_clipped_double:
            self.twin_critic_optimizer = torch.optim.Adam(self.twin_critic.parameters(),
                                                          lr=self.hps.critic_lr,
                                                          weight_decay=self.hps.wd_scale)

        log_module_info(logger, 'actor', self.actor)
        log_module_info(logger, 'critic', self.critic)

    def parse_noise_type(self, noise_type):
        """Parse the `noise_type` hyperparameter"""
        ac_noise = None
        param_noise = None
        logger.info("parsing noise type")
        # Parse the comma-seprated (with possible whitespaces) list of noise params
        for cur_noise_type in noise_type.split(','):
            cur_noise_type = cur_noise_type.strip()  # remove all whitespaces (start and end)
            # If the specified noise type is literally 'none'
            if cur_noise_type == 'none':
                pass
            # If 'adaptive-param' is in the specified string for noise type
            elif 'adaptive-param' in cur_noise_type:
                # Set parameter noise
                from algorithms.ddpg.param_noise import AdaptiveParamNoise
                _, std = cur_noise_type.split('_')
                std = float(std)
                param_noise = AdaptiveParamNoise(initial_std=std, delta=std)
                logger.info("  {} configured".format(param_noise))
            elif 'normal' in cur_noise_type:
                _, std = cur_noise_type.split('_')
                # Spherical (isotropic) gaussian action noise
                from algorithms.ddpg.ac_noise import NormalAcNoise
                ac_noise = NormalAcNoise(mu=np.zeros(self.ac_dim),
                                         sigma=float(std) * np.ones(self.ac_dim))
                logger.info("  {} configured".format(ac_noise))
            elif 'ou' in cur_noise_type:
                _, std = cur_noise_type.split('_')
                # Ornstein-Uhlenbeck action noise
                from algorithms.ddpg.ac_noise import OUAcNoise
                ac_noise = OUAcNoise(mu=np.zeros(self.ac_dim),
                                     sigma=(float(std) * np.ones(self.ac_dim)))
                logger.info("  {} configured".format(ac_noise))
            else:
                raise RuntimeError("unknown specified noise type: '{}'".format(cur_noise_type))
        return param_noise, ac_noise

    def setup_replay_buffer(self):
        """Setup experiental memory unit"""
        logger.info("setting up replay buffer")
        if self.hps.prioritized_replay:
            if self.hps.unreal:  # Unreal prioritized experience replay
                self.replay_buffer = UnrealReplayBuffer(self.hps.mem_size,
                                                        self.ob_shape,
                                                        self.ac_shape)
            else:  # Vanilla prioritized experience replay
                self.replay_buffer = PrioritizedReplayBuffer(self.hps.mem_size,
                                                             self.ob_shape,
                                                             self.ac_shape,
                                                             alpha=self.hps.alpha,
                                                             beta=self.hps.beta,
                                                             ranked=self.hps.ranked)
        else:  # Vanilla experience replay
            self.replay_buffer = ReplayBuffer(self.hps.mem_size,
                                              self.ob_shape,
                                              self.ac_shape)
        # Summarize replay buffer creation (relies on `__repr__` method)
        logger.info("  {} configured".format(self.replay_buffer))

    def predict(self, ob, apply_noise):
        """Predict an action, with or without perturbation,
        and optionaly compute and return the associated Q value.
        """
        # Create tensor from the state (`require_grad=False` by default)
        ob = torch.FloatTensor(ob[None]).to(self.device)

        if apply_noise and self.param_noise is not None:
            # Predict following a parameter-noise-perturbed actor
            ac = self.pnp_actor(ob)
        else:
            # Predict following the non-perturbed actor
            ac = self.actor(ob)

        # Place on cpu and collapse into one dimension
        # tau tilde
        q = self.critic.Q(ob, ac, self.hps.num_tau_tilde, self.device).cpu().detach().numpy().flatten()
        ac = ac.cpu().detach().numpy().flatten()

        if apply_noise and self.ac_noise is not None:
            # Apply additive action noise once the action has been predicted,
            # in combination with parameter noise, or not.
            noise = self.ac_noise.generate()
            assert noise.shape == ac.shape
            ac += noise

        ac = ac.clip(-self.max_ac, self.max_ac)

        return ac, q

    def store_transition(self, ob0, ac, rew, ob1, done1):
        """Store a experiental transition in the replay buffer"""
        # Scale the reward
        rew *= self.hps.reward_scale

        # Store the transition in the replay buffer
        self.replay_buffer.append(ob0, ac, rew, ob1, done1)

    def train(self, update_actor=True):
        """Train the agent"""

        # Get a batch of transitions from the replay buffer
        if self.hps.n_step_returns:
            batch = self.replay_buffer.lookahead_sample(self.hps.batch_size,
                                                        n=self.hps.n,
                                                        gamma=self.hps.gamma)
        else:
            batch = self.replay_buffer.sample(self.hps.batch_size)

        # Create tensors from the inputs (`require_grad=False` by default)
        MiniBatch = namedtuple('MiniBatch', batch.keys())
        mb = MiniBatch(**batch)
        state = torch.FloatTensor(mb.obs0).to(self.device)
        action = torch.FloatTensor(mb.acs).to(self.device)
        next_state = torch.FloatTensor(mb.obs1).to(self.device)
        reward = torch.FloatTensor(mb.rews).to(self.device)
        done = torch.FloatTensor(mb.dones1.astype('float32')).to(self.device)
        if self.hps.prioritized_replay:
            iws = torch.FloatTensor(mb.iws).to(self.device)
            # idxs = torch.FloatTensor(mb.idxs).to(self.device)
        if self.hps.n_step_returns:
            td_len = torch.FloatTensor(mb.td_len).to(self.device)
        else:
            td_len = torch.ones_like(done).to(self.device)

        if self.hps.enable_targ_actor_smoothing:
            n_ = torch.FloatTensor(mb.acs).data.normal_(0, self.hps.td3_std).to(self.device)
            n_ = n_.clamp(-self.hps.c, self.hps.c)
            next_action = (self.targ_actor(next_state) + n_).clamp(-self.max_ac, self.max_ac)
        else:
            next_action = self.targ_actor(next_state)

        # Compute Q estimate(s)
        # tau
        q = self.critic(state, action, self.hps.num_tau, self.device)
        if self.hps.enable_clipped_double:
            # twin tau
            twin_q = self.twin_critic(state, action, self.hps.num_tau, self.device)

        # Compute target Q estimate
        # tau prime
        q_prime = self.targ_critic(next_state, next_action, self.hps.num_tau_prime, self.device)
        if self.hps.enable_clipped_double:
            # Define Q' as the minimum Q value between TD3's twin Q's
            # twin tau prime
            twin_q_prime = self.targ_twin_critic(next_state, next_action, self.hps.num_tau_prime, self.device)
            q_prime = torch.min(q_prime, twin_q_prime)
        # Reshape rewards to be of shape [batch_size x num_tau_prime, 1]
        reward = reward.repeat(self.hps.num_tau_prime, 1)
        # Reshape product of gamma and mask to be of shape [batch_size x num_tau_prime, 1]
        gamma_mask = ((self.hps.gamma ** td_len) * (1 - done)).repeat(self.hps.num_tau_prime, 1)
        # Reshape Q prime to be of shape [batch_size x num_tau_prime, 1]
        q_prime = q_prime.view(-1, 1)

        targ_q = (reward + gamma_mask * q_prime).detach()
        # Reshape target to be of shape [batch_size, num_tau_prime, 1]
        targ_q = targ_q.view(-1, self.hps.num_tau_prime, 1)

        # Critic(s) loss(es)

        # Retrieve the quantile values sampled during the forward pass
        quantiles = self.critic.sampled_quantiles
        if self.hps.enable_clipped_double:
            twin_quantiles = self.twin_critic.sampled_quantiles
        # Quantiles are shaped as: [batch_size, num_tau, 1]

        # Tile by num_tau_prime_samples along a new dimension, the goal is to give
        # the Huber quantile regression loss quantiles that have the same shape
        # as the TD errors for it to deal with each component as expected
        quantiles = quantiles[:, None, :, :].repeat(1, self.hps.num_tau_prime, 1, 1)
        # The resulting shape is [batch_size, num_tau_prime, num_tau, 1]
        if self.hps.enable_clipped_double:
            twin_quantiles = twin_quantiles[:, None, :, :].repeat(1, self.hps.num_tau_prime, 1, 1)

        # Compute the TD error loss
        # Note: online version has shape [batch_size, num_tau, 1],
        # while the target version has shape [batch_size, num_tau_prime, 1].
        td_errors = targ_q[:, :, None, :] - q[:, None, :, :]  # broadcasting
        # The resulting shape is [batch_size, num_tau_prime, num_tau, 1]
        if self.hps.enable_clipped_double:
            twin_td_errors = targ_q[:, :, None, :] - twin_q[:, None, :, :]  # broadcasting

        # Assemble the Huber Quantile Regression loss
        huber_td_errors = huber_quant_reg_loss(td_errors, quantiles)
        # The resulting shape is [batch_size, num_tau_prime, num_tau, 1]
        if self.hps.enable_clipped_double:
            twin_huber_td_errors = huber_quant_reg_loss(twin_td_errors, twin_quantiles)

        if self.hps.prioritized_replay:
            # Adjust with importance weights
            huber_td_errors *= iws
            if self.hps.enable_clipped_double:
                twin_huber_td_errors *= iws

        # Sum over current quantile value (tau, N in paper) dimension, and
        # average over target quantile value (tau prime, N' in paper) dimension.
        critic_loss = huber_td_errors.sum(dim=2)
        # Resulting shape is [batch_size, num_tau_prime, 1]
        if self.hps.enable_clipped_double:
            twin_critic_loss = twin_huber_td_errors.sum(dim=2)
        critic_loss = critic_loss.mean(dim=1)
        # Resulting shape is [batch_size, 1]
        if self.hps.enable_clipped_double:
            twin_critic_loss = twin_critic_loss.mean(dim=1)

        # Average across the minibatch
        critic_loss = critic_loss.mean()
        if self.hps.enable_clipped_double:
            twin_critic_loss = twin_critic_loss.mean()

        # Critic(s) grads
        self.critic_optimizer.zero_grad()
        if self.hps.enable_clipped_double:
            self.twin_critic_optimizer.zero_grad()

        critic_loss.backward()
        critic_gradnorm = U.clip_grad_norm_(self.critic.parameters(),
                                            self.hps.clip_norm)
        if self.hps.enable_clipped_double:
            twin_critic_loss.backward()
            twin_critic_gradnorm = U.clip_grad_norm_(self.twin_critic.parameters(),
                                                     self.hps.clip_norm)

        # Update critic(s)
        average_gradients(self.critic, self.comm, self.device)
        self.critic_optimizer.step()
        if self.hps.enable_clipped_double:
            average_gradients(self.twin_critic, self.comm, self.device)
            self.twin_critic_optimizer.step()

        # Actor loss
        # tau tilde
        actor_loss = -self.critic.Q(state, self.actor(state), self.hps.num_tau_tilde, self.device).mean()

        # Actor grads
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_gradnorm = U.clip_grad_norm_(self.actor.parameters(),
                                           self.hps.clip_norm)

        if update_actor:
            # Update actor
            average_gradients(self.actor, self.comm, self.device)
            self.actor_optimizer.step()

            # Update target nets
            self.update_target_net()

        if self.hps.prioritized_replay:
            # Update priorities
            td_errors = q - targ_q
            new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6  # epsilon from paper
            self.replay_buffer.update_priorities(mb.idxs, new_priorities)

        # Aggregate the elements to return
        losses = {'actor': actor_loss.clone().cpu().data.numpy(),
                  'critic': critic_loss.clone().cpu().data.numpy()}
        gradnorms = {'actor': actor_gradnorm,
                     'critic': critic_gradnorm}
        if self.hps.enable_clipped_double:
            losses.update({'twin_critic': twin_critic_loss.clone().cpu().data.numpy()})
            gradnorms.update({'twin_critic': twin_critic_gradnorm})

        return losses, gradnorms

    def update_target_net(self):
        """Update the target networks by slowly tracking their non-target counterparts"""
        for param, targ_param in zip(self.actor.parameters(), self.targ_actor.parameters()):
            targ_param.data.copy_(self.hps.polyak * param.data +
                                  (1. - self.hps.polyak) * targ_param.data)
        for param, targ_param in zip(self.critic.parameters(), self.targ_critic.parameters()):
            targ_param.data.copy_(self.hps.polyak * param.data +
                                  (1. - self.hps.polyak) * targ_param.data)
        if self.hps.enable_clipped_double:
            for param, targ_param in zip(self.twin_critic.parameters(),
                                         self.targ_twin_critic.parameters()):
                targ_param.data.copy_(self.hps.polyak * param.data +
                                      (1. - self.hps.polyak) * targ_param.data)

    def adapt_param_noise(self):
        """Adapt the parameter noise standard deviation"""
        # Perturb separate copy of the policy to adjust the scale for the next 'real' perturbation

        batch = self.replay_buffer.sample(self.hps.batch_size)

        state = torch.FloatTensor(batch['obs0']).to(self.device)

        # Update the perturbable params
        for p in self.actor.perturbable_params:
            param = (self.actor.state_dict()[p]).clone()
            param_ = param.clone()
            noise = param_.data.normal_(0, self.param_noise.cur_std)
            self.apnp_actor.state_dict()[p].data.copy_((param + noise).data)
        # Update the non-perturbable params
        for p in self.actor.nonperturbable_params:
            param = self.actor.state_dict()[p].clone()
            self.apnp_actor.state_dict()[p].data.copy_(param.data)

        # Compute distance between actor and adaptive-parameter-noise-perturbed actor predictions
        self.pn_dist = torch.sqrt(F.mse_loss(self.actor(state), self.apnp_actor(state)))

        self.pn_dist = self.pn_dist.cpu().data.numpy()

        # Adapt the parameter noise
        self.param_noise.adapt_std(self.pn_dist)

    def reset_noise(self):
        """Reset noise processes at episode termination"""

        # Reset action noise
        if self.ac_noise is not None:
            self.ac_noise.reset()

        # Reset parameter-noise-perturbed actor vars by redefining the pnp actor
        # w.r.t. the actor (by applying additive gaussian noise with current std)
        if self.param_noise is not None:
            # Update the perturbable params
            for p in self.actor.perturbable_params:
                param = (self.actor.state_dict()[p]).clone()
                param_ = param.clone()
                noise = param_.data.normal_(0, self.param_noise.cur_std)
                self.pnp_actor.state_dict()[p].data.copy_((param + noise).data)
            # Update the non-perturbable params
            for p in self.actor.nonperturbable_params:
                param = self.actor.state_dict()[p].clone()
                self.pnp_actor.state_dict()[p].data.copy_(param.data)

    def save(self, path, iters):
        SaveBundle = namedtuple('SaveBundle', ['model', 'optimizer'])
        actor_bundle = SaveBundle(model=self.actor.state_dict(),
                                  optimizer=self.actor_optimizer.state_dict())
        critic_bundle = SaveBundle(model=self.critic.state_dict(),
                                   optimizer=self.critic_optimizer.state_dict())
        torch.save(actor_bundle._asdict(), "{}_actor_iter{}.pth".format(path, iters))
        torch.save(critic_bundle._asdict(), "{}_critic_iter{}.pth".format(path, iters))

    def load(self, path, iters):
        experiment = path.split('/')[-1]
        actor_bundle = torch.load("{}/{}_actor_iter{}.pth".format(path, experiment, iters))
        self.actor.load_state_dict(actor_bundle['model'])
        self.actor_optimizer.load_state_dict(actor_bundle['optimizer'])
        critic_bundle = torch.load("{}/{}_critic_iter{}.pth".format(path, experiment, iters))
        self.critic.load_state_dict(critic_bundle['model'])
        self.critic_optimizer.load_state_dict(critic_bundle['optimizer'])
