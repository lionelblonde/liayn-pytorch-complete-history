from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as U
import torch.nn.functional as F

from algorithms.helpers import logger
from algorithms.helpers.console_util import log_module_info
from algorithms.helpers.distributed_util import average_gradients, sync_with_root
from algorithms.ddpg.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer


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
        super(Critic, self).__init__()
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.ob_dim = self.ob_space.shape[0]
        self.ac_dim = self.ac_space.shape[0]
        self.hps = hps

        self.hid_fc_1 = nn.Linear(self.ob_dim + self.ac_dim, 400)
        if self.hps.with_layernorm:
            self.layer_norm_1 = nn.LayerNorm(400)
        self.hid_fc_2 = nn.Linear(400, 300)
        if self.hps.with_layernorm:
            self.layer_norm_2 = nn.LayerNorm(300)
        self.out_fc_3 = nn.Linear(300, 1)

    def Q(self, ob, ac):
        plop = torch.cat([ob, ac], dim=-1)
        plop = F.leaky_relu(self.hid_fc_1(plop))
        if self.hps.with_layernorm:
            plop = self.layer_norm_1(plop)
        plop = F.leaky_relu(self.hid_fc_2(plop))
        if self.hps.with_layernorm:
            plop = self.layer_norm_2(plop)
        q = self.out_fc_3(plop)
        return q

    def forward(self, ob, ac):
        q = self.Q(ob, ac)
        return q


class DDPGAgent(object):

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
            sync_with_root(self.twin_critic, comm)
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
        q = self.critic.Q(ob, ac).cpu().numpy().flatten()
        ac = ac.cpu().numpy().flatten()

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
            # n_ = n_.clamp(-self.hps.c, self.hps.c)
            n_ = n_.clamp(-0.2, 0.2)
            next_action = (self.targ_actor(next_state) + n_).clamp(-self.max_ac, self.max_ac)
        else:
            next_action = self.targ_actor(next_state)

        # Compute Q estimate(s)
        q = self.critic(state, action)

        if self.hps.enable_clipped_double:
            twin_q = self.twin_critic(state, action)

        # Compute target Q estimate
        q_prime = self.targ_critic(next_state, next_action)
        if self.hps.enable_clipped_double:
            # Define Q' as the minimum Q value between TD3's twin Q's
            twin_q_prime = self.targ_twin_critic(next_state, next_action)
            q_prime = torch.min(q_prime, twin_q_prime)
        targ_q = (reward + (self.hps.gamma ** td_len) * (1 - done) * q_prime).detach()

        # Actor loss
        actor_loss = -self.critic.Q(state, self.actor(state)).mean()

        # Critic(s) loss(es)
        if self.hps.prioritized_replay:
            # Adjust with importance weights
            squared_td_errors = F.mse_loss(q, targ_q, reduction='none')
            critic_loss = (iws * squared_td_errors).mean()
        else:
            critic_loss = F.mse_loss(q, targ_q)

        if self.hps.enable_clipped_double:
            if self.hps.prioritized_replay:
                # Adjust with importance weights
                twin_squared_td_errors = F.mse_loss(twin_q, targ_q, reduction='none')
                twin_critic_loss = (iws * twin_squared_td_errors).mean()
            else:
                twin_critic_loss = F.mse_loss(twin_q, targ_q)

        # Actor grads
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_gradnorm = U.clip_grad_norm_(self.actor.parameters(),
                                           self.hps.clip_norm)

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

        if update_actor:
            # Update actor
            average_gradients(self.actor, self.comm, self.device)
            self.actor_optimizer.step()

            # Update target nets
            self.update_target_net()

        if self.hps.prioritized_replay:
            # Update priorities
            td_errors = q - targ_q
            if self.hps.enable_clipped_double:
                td_errors = torch.min(q - targ_q, twin_q - targ_q)
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
