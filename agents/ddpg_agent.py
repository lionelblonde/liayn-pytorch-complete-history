from collections import namedtuple
import os.path as osp

import numpy as np
import torch
import torch.nn.utils as U
import torch.nn.functional as F
from torch.utils.data import DataLoader

from helpers import logger
from helpers.console_util import log_env_info, log_module_info
from helpers.dataset import Dataset
from helpers.math_util import huber_quant_reg_loss
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from agents.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer
from agents.nets import Actor, VanillaCritic, C51QRCritic, IQNCritic, Discriminator
from agents.param_noise import AdaptiveParamNoise
from agents.ac_noise import NormalAcNoise, OUAcNoise


class DDPGAgent(object):

    def __init__(self, env, device, hps, expert_dataset):
        self.env = env
        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape

        log_env_info(logger, self.env)

        self.ob_dim = self.ob_shape[0]  # num dims
        self.ac_dim = self.ac_shape[0]  # num dims

        self.device = device
        self.hps = hps
        assert self.hps.lookahead > 1 or not self.hps.n_step_returns

        # Define action clipping range
        assert all(self.ac_space.low == -self.ac_space.high)
        self.max_ac = self.ac_space.high[0].astype('float32')
        assert all(ac_comp == self.max_ac for ac_comp in self.ac_space.high)

        # Define critic to use
        assert sum([self.hps.use_c51, self.hps.use_qr, self.hps.use_iqn]) <= 1
        if self.hps.use_c51:
            assert not self.hps.clipped_double
            Critic = C51QRCritic
            c51_supp_range = (self.hps.c51_vmin,
                              self.hps.c51_vmax,
                              self.hps.c51_num_atoms)
            self.c51_supp = torch.linspace(*c51_supp_range).to(self.device)
            self.c51_delta = ((self.hps.c51_vmax - self.hps.c51_vmin) /
                              (self.hps.c51_num_atoms - 1))
            c51_offset_range = (0,
                                (self.hps.batch_size - 1) * self.hps.c51_num_atoms,
                                self.hps.batch_size)
            c51_offset = torch.linspace(*c51_offset_range)
            self.c51_offset = c51_offset.long().unsqueeze(1).expand(self.hps.batch_size,
                                                                    self.hps.c51_num_atoms)
        elif self.hps.use_qr:
            assert not self.hps.clipped_double
            Critic = C51QRCritic
            qr_cum_density = np.array([((2 * i) + 1) / (2.0 * self.hps.num_tau)
                                       for i in range(self.hps.num_tau)])
            qr_cum_density = torch.FloatTensor(qr_cum_density).to(self.device)
            self.qr_cum_density = qr_cum_density.view(1, 1, -1, 1).expand(self.hps.batch_size,
                                                                          self.hps.num_tau,
                                                                          self.hps.num_tau,
                                                                          -1)
        elif self.hps.use_iqn:
            assert not self.hps.clipped_double
            Critic = IQNCritic
        else:
            Critic = VanillaCritic

        # Parse the noise types
        self.param_noise, self.ac_noise = self.parse_noise_type(self.hps.noise_type)

        # Create online and target nets, and initilize the target nets
        self.actr = Actor(self.env, self.hps).to(self.device)
        sync_with_root(self.actr)
        self.targ_actr = Actor(self.env, self.hps).to(self.device)
        self.targ_actr.load_state_dict(self.actr.state_dict())
        self.crit = Critic(self.env, self.hps).to(self.device)
        sync_with_root(self.crit)
        self.targ_crit = Critic(self.env, self.hps).to(self.device)
        self.targ_crit.load_state_dict(self.crit.state_dict())
        if self.hps.clipped_double:
            # Create second ('twin') critic and target critic
            # TD3, https://arxiv.org/abs/1802.09477
            self.twin = Critic(self.env, self.hps).to(self.device)
            sync_with_root(self.twin)
            self.targ_twin = Critic(self.env, self.hps).to(self.device)
            self.targ_twin.load_state_dict(self.twin.state_dict())

        if self.param_noise is not None:
            # Create parameter-noise-perturbed ('pnp') actor
            self.pnp_actr = Actor(self.env, self.hps).to(self.device)
            self.pnp_actr.load_state_dict(self.actr.state_dict())
            # Create adaptive-parameter-noise-perturbed ('apnp') actor
            self.apnp_actr = Actor(self.env, self.hps).to(self.device)
            self.apnp_actr.load_state_dict(self.actr.state_dict())

        # Set up replay buffer
        self.setup_replay_buffer()

        # Set up the optimizers
        self.actr_opt = torch.optim.Adam(self.actr.parameters(),
                                         lr=self.hps.actor_lr)
        self.crit_opt = torch.optim.Adam(self.crit.parameters(),
                                         lr=self.hps.critic_lr,
                                         weight_decay=self.hps.wd_scale)
        if self.hps.clipped_double:
            self.twin_opt = torch.optim.Adam(self.twin.parameters(),
                                             lr=self.hps.critic_lr,
                                             weight_decay=self.hps.wd_scale)

        # Set up the learning rate schedule
        def _lr(t):  # flake8: using a def instead of a lambda
            if self.hps.with_scheduler:
                return (1.0 - ((t - 1.0) / (self.hps.num_timesteps //
                                            self.hps.rollout_len)))
            else:
                return 1.0

        self.actr_sched = torch.optim.lr_scheduler.LambdaLR(self.actr_opt, _lr)
        self.crit_sched = torch.optim.lr_scheduler.LambdaLR(self.crit_opt, _lr)
        if self.hps.clipped_double:
            self.twin_sched = torch.optim.lr_scheduler.LambdaLR(self.twin_opt, _lr)

        if expert_dataset is not None:
            # Set up demonstrations dataset
            self.e_dataloader = DataLoader(expert_dataset, self.hps.batch_size, shuffle=True)
            # Create discriminator
            self.disc = Discriminator(self.env, self.hps).to(self.device)
            sync_with_root(self.disc)
            # Create optimizer
            self.disc_opt = torch.optim.Adam(self.disc.parameters(),
                                             lr=self.hps.d_lr)

        log_module_info(logger, 'actr', self.actr)
        log_module_info(logger, 'crit', self.crit)
        if self.hps.clipped_double:
            log_module_info(logger, 'twin', self.crit)
        if hasattr(self, 'disc'):
            log_module_info(logger, 'disc', self.disc)

        if not self.hps.pixels:
            self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=True)

        if self.hps.popart:
            self.rms_ret = RunMoms(shape=(1,), use_mpi=True)

    def norm_rets(self, x):
        if self.hps.popart:
            return ((x - torch.FloatTensor(self.rms_ret.mean)).to(self.device) /
                    torch.FloatTensor(np.sqrt(self.rms_ret.var)).to(self.device) + 1e-8)
        else:
            return x

    def denorm_rets(self, x):
        if self.hps.popart:
            return ((x * torch.FloatTensor(np.sqrt(self.rms_ret.var))).to(self.device) +
                    torch.FloatTensor(self.rms_ret.mean).to(self.device))
        else:
            return x

    def parse_noise_type(self, noise_type):
        """Parse the `noise_type` hyperparameter"""
        ac_noise = None
        param_noise = None
        logger.info(">>>> parsing noise type")
        # Parse the comma-seprated (with possible whitespaces) list of noise params
        for cur_noise_type in noise_type.split(','):
            cur_noise_type = cur_noise_type.strip()  # remove all whitespaces (start and end)
            # If the specified noise type is literally 'none'
            if cur_noise_type == 'none':
                pass
            # If 'adaptive-param' is in the specified string for noise type
            elif 'adaptive-param' in cur_noise_type:
                # Set parameter noise
                _, std = cur_noise_type.split('_')
                std = float(std)
                param_noise = AdaptiveParamNoise(initial_std=std, delta=std)
                logger.info("[INFO] {} configured".format(param_noise))
            elif 'normal' in cur_noise_type:
                _, std = cur_noise_type.split('_')
                # Spherical (isotropic) gaussian action noise
                ac_noise = NormalAcNoise(mu=np.zeros(self.ac_dim),
                                         sigma=float(std) * np.ones(self.ac_dim))
                logger.info("[INFO] {} configured".format(ac_noise))
            elif 'ou' in cur_noise_type:
                _, std = cur_noise_type.split('_')
                # Ornstein-Uhlenbeck action noise
                ac_noise = OUAcNoise(mu=np.zeros(self.ac_dim),
                                     sigma=(float(std) * np.ones(self.ac_dim)))
                logger.info("[INFO] {} configured".format(ac_noise))
            else:
                raise RuntimeError("unknown specified noise type: '{}'".format(cur_noise_type))
        return param_noise, ac_noise

    def setup_replay_buffer(self):
        """Setup experiental memory unit"""
        logger.info(">>>> setting up replay buffer")
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
        logger.info("[INFO] {} configured".format(self.replay_buffer))

    def normalize_clip_ob(self, ob):
        # Normalize with running mean and running std
        ob = ((ob - self.rms_obs.mean) /
              (np.sqrt(self.rms_obs.var) + 1e-8))
        # Clip
        ob = np.clip(ob, -5.0, 5.0)
        return ob

    def predict(self, ob, apply_noise):
        """Predict an action, with or without perturbation,
        and optionaly compute and return the associated Q value.
        """
        # Create tensor from the state (`require_grad=False` by default)
        ob = ob[None] if self.hps.pixels else self.normalize_clip_ob(ob[None])
        ob = torch.FloatTensor(ob).to(self.device)

        if apply_noise and self.param_noise is not None:
            # Predict following a parameter-noise-perturbed actor
            ac = self.pnp_actr.act(ob)
        else:
            # Predict following the non-perturbed actor
            ac = self.actr.act(ob)

        # Place on cpu and collapse into one dimension
        ac = ac.cpu().detach().numpy().flatten()

        if apply_noise and self.ac_noise is not None:
            # Apply additive action noise once the action has been predicted,
            # in combination with parameter noise, or not.
            noise = self.ac_noise.generate()
            assert noise.shape == ac.shape
            ac += noise

        ac = ac.clip(-self.max_ac, self.max_ac)

        return ac

    def store_transition(self, ob0, ac, rew, ob1, done1):
        """Store a experiental transition in the replay buffer"""
        # Scale the reward
        rew *= self.hps.reward_scale
        # Store the transition in the replay buffer
        self.replay_buffer.append(ob0, ac, rew, ob1, done1)

    def train(self, update_critic, update_actor, iters_so_far):
        """Train the agent"""
        # Get a batch of transitions from the replay buffer
        if self.hps.n_step_returns:
            batch = self.replay_buffer.lookahead_sample(self.hps.batch_size,
                                                        n=self.hps.lookahead,
                                                        gamma=self.hps.gamma)
        else:
            batch = self.replay_buffer.sample(self.hps.batch_size)

        if not self.hps.pixels:
            # Standardize and clip observations
            batch['obs0'] = self.normalize_clip_ob(batch['obs0'])
            batch['obs1'] = self.normalize_clip_ob(batch['obs1'])

        # Create tensors from the inputs
        state = torch.FloatTensor(batch['obs0']).to(self.device)
        action = torch.FloatTensor(batch['acs']).to(self.device)
        next_state = torch.FloatTensor(batch['obs1']).to(self.device)
        reward = torch.FloatTensor(batch['rews']).to(self.device)
        done = torch.FloatTensor(batch['dones1'].astype('float32')).to(self.device)
        if self.hps.prioritized_replay:
            iws = torch.FloatTensor(batch['iws']).to(self.device)
        if self.hps.n_step_returns:
            td_len = torch.FloatTensor(batch['td_len']).to(self.device)
        else:
            td_len = torch.ones_like(done).to(self.device)

        if self.hps.targ_actor_smoothing:
            n_ = action.clone().detach().data.normal_(0, self.hps.td3_std).to(self.device)
            n_ = n_.clamp(-self.hps.td3_c, self.hps.td3_c)
            next_action = (self.targ_actr.act(next_state) + n_).clamp(-self.max_ac, self.max_ac)
        else:
            next_action = self.targ_actr.act(next_state)

        if hasattr(self, 'disc'):
            # Create data loaders
            dataset = Dataset(batch)
            dataloader = DataLoader(dataset, self.hps.batch_size, shuffle=True)
            # Iterable over 1 element, but cleaner that way
            # Collect recent pairs uniformly from the experience replay buffer
            recency = int(self.replay_buffer.capacity * 0.02)
            assert recency >= self.hps.batch_size, "must have recency >= batch_size"
            recent_batch = self.replay_buffer.sample_recent(self.hps.batch_size, recency)
            recent_dataset = Dataset(recent_batch)
            recent_dataloader = DataLoader(recent_dataset, self.hps.batch_size, shuffle=True)
            # Iterable over 1 element, but cleaner that way

        # Compute losses

        if self.hps.use_c51:

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> C51-like.

            # Compute Z estimate
            z = self.crit.Z(state, action).unsqueeze(-1)
            z.data.clamp_(0.01, 0.99)
            # Compute target Z estimate
            z_prime = self.targ_crit.Z(next_state, next_action)
            z_prime.data.clamp_(0.01, 0.99)
            gamma_mask = ((self.hps.gamma ** td_len) * (1 - done))
            Tz = reward + (gamma_mask * self.c51_supp.view(1, self.hps.c51_num_atoms))
            Tz = Tz.clamp(self.hps.c51_vmin, self.hps.c51_vmax)
            b = (Tz - self.hps.c51_vmin) / self.c51_delta
            l = b.floor().long()  # noqa
            u = b.ceil().long()
            l[(u > 0) * (l == u)] -= 1  # noqa
            u[(l < (self.hps.c51_num_atoms - 1)) * (l == u)] += 1  # noqa
            targ_z = z_prime.clone().zero_()
            targ_z.view(-1).index_add_(0,
                                       (l + self.c51_offset).view(-1),
                                       (z_prime * (u.float() - b)).view(-1))
            targ_z.view(-1).index_add_(0,
                                       (u + self.c51_offset).view(-1),
                                       (z_prime * (b - l.float())).view(-1))
            # Reshape target to be of shape [batch_size, self.hps.c51_num_atoms, 1]
            targ_z = targ_z.view(-1, self.hps.c51_num_atoms, 1)

            # Critic loss
            kls = -(targ_z.detach() * torch.log(z + 1e-8)).sum(dim=1)

            if self.hps.prioritized_replay:
                # Update priorities
                new_priorities = np.abs(kls.sum(dim=1).detach().cpu().numpy()) + 1e-6
                self.replay_buffer.update_priorities(batch['idxs'], new_priorities)
                # Adjust with importance weights
                kls *= iws

            crit_loss = kls.mean()

            # Actor loss
            actr_loss = -self.crit.Z(state, self.actr.act(state))
            actr_loss = actr_loss.matmul(self.c51_supp).mean()

        elif self.hps.use_qr:

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> QR-like.

            # Compute Z estimate
            z = self.crit.Z(state, action).unsqueeze(-1)

            # Compute target Z estimate
            z_prime = self.targ_crit.Z(next_state, next_action)
            # Reshape rewards to be of shape [batch_size x num_tau, 1]
            reward = reward.repeat(self.hps.num_tau, 1)
            # Reshape product of gamma and mask to be of shape [batch_size x num_tau, 1]
            gamma_mask = ((self.hps.gamma ** td_len) * (1 - done)).repeat(self.hps.num_tau, 1)
            z_prime = z_prime.view(-1, 1)
            targ_z = reward + (gamma_mask * z_prime)
            # Reshape target to be of shape [batch_size, num_tau, 1]
            targ_z = targ_z.view(-1, self.hps.num_tau, 1)

            # Critic loss
            # Compute the TD error loss
            # Note: online version has shape [batch_size, num_tau, 1],
            # while the target version has shape [batch_size, num_tau, 1].
            td_errors = targ_z[:, :, None, :].detach() - z[:, None, :, :]  # broadcasting
            # The resulting shape is [batch_size, num_tau, num_tau, 1]

            # Assemble the Huber Quantile Regression loss
            huber_td_errors = huber_quant_reg_loss(td_errors, self.qr_cum_density)
            # The resulting shape is [batch_size, num_tau_prime, num_tau, 1]

            if self.hps.prioritized_replay:
                # Adjust with importance weights
                huber_td_errors *= iws
                # Update priorities
                new_priorities = np.abs(td_errors.sum(dim=2).mean(dim=1).detach().cpu().numpy())
                new_priorities += 1e-6
                self.replay_buffer.update_priorities(batch['idxs'], new_priorities)

            # Sum over current quantile value (tau, N in paper) dimension, and
            # average over target quantile value (tau prime, N' in paper) dimension.
            crit_loss = huber_td_errors.sum(dim=2)
            # Resulting shape is [batch_size, num_tau_prime, 1]
            crit_loss = crit_loss.mean(dim=1)
            # Resulting shape is [batch_size, 1]
            # Average across the minibatch
            crit_loss = crit_loss.mean()

            # Actor loss
            actr_loss = -self.crit.Z(state, self.actr.act(state)).mean()

        elif self.hps.use_iqn:

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IQN-like.

            # Compute Z estimate
            z, tau = self.crit.Z(state, action, self.hps.num_tau)

            # Compute target Z estimate
            z_prime, _ = self.targ_crit.Z(next_state, next_action, self.hps.num_tau_prime)
            # Reshape rewards to be of shape [batch_size x num_tau_prime, 1]
            reward_reps = reward.repeat(self.hps.num_tau_prime, 1)
            # Reshape product of gamma and mask to be of shape [batch_size x num_tau_prime, 1]
            gamma_mask = ((self.hps.gamma ** td_len) * (1 - done)).repeat(self.hps.num_tau_prime, 1)
            # Reshape Z prime to be of shape [batch_size x num_tau_prime, 1]
            z_prime = z_prime.view(-1, 1)
            targ_z = reward_reps + (gamma_mask * z_prime.detach())
            # Reshape target to be of shape [batch_size, num_tau_prime, 1]
            targ_z = targ_z.view(-1, self.hps.num_tau_prime, 1)

            # Critic loss
            # Tile by num_tau_prime_samples along a new dimension, the goal is to give
            # the Huber quantile regression loss quantiles that have the same shape
            # as the TD errors for it to deal with each component as expected
            tau = tau.view(self.hps.batch_size, self.hps.num_tau, 1)[:, None, :, :]
            tau = tau.repeat(1, self.hps.num_tau_prime, 1, 1)

            # Compute the TD error loss
            # Note: online version has shape [batch_size, num_tau, 1],
            # while the target version has shape [batch_size, num_tau_prime, 1].
            td_errors = targ_z[:, :, None, :] - z[:, None, :, :]  # broadcasting
            # The resulting shape is [batch_size, num_tau_prime, num_tau, 1]

            # Assemble the Huber Quantile Regression loss
            huber_td_errors = huber_quant_reg_loss(td_errors, tau)
            # The resulting shape is [batch_size, num_tau_prime, num_tau, 1]

            if self.hps.prioritized_replay:
                # Adjust with importance weights
                huber_td_errors *= iws
                # Update priorities
                new_priorities = np.abs(td_errors.sum(dim=2).mean(dim=1).detach().cpu().numpy())
                new_priorities += 1e-6
                self.replay_buffer.update_priorities(batch['idxs'], new_priorities)

            # Sum over current quantile value (tau, N in paper) dimension, and
            # average over target quantile value (tau prime, N' in paper) dimension.
            crit_loss = huber_td_errors.sum(dim=2)
            # Resulting shape is [batch_size, num_tau_prime, 1]
            crit_loss = crit_loss.mean(dim=1)
            # Resulting shape is [batch_size, 1]
            # Average across the minibatch
            crit_loss = crit_loss.mean()

            # Actor loss
            actr_loss = -self.crit.Z(state, self.actr.act(state), self.hps.num_tau_tilde)[0].mean()

        else:

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VANILLA

            # Compute Q estimate
            q = self.denorm_rets(self.crit.Q(state, action))
            if self.hps.clipped_double:
                twin_q = self.denorm_rets(self.twin.Q(state, action))

            # Compute target Q estimate
            q_prime = self.targ_crit.Q(next_state, next_action)
            if self.hps.clipped_double:
                # Define Q' as the minimum Q value between TD3's twin Q's
                twin_q_prime = self.targ_twin.Q(next_state, next_action)
                q_prime = torch.min(q_prime, twin_q_prime)
            targ_q = (reward +
                      (self.hps.gamma ** td_len) * (1. - done) *
                      self.denorm_rets(q_prime).detach())  # make q_prime unnormalized
            # Normalize target Q with running statistics
            targ_q = self.norm_rets(targ_q)

            if self.hps.popart:
                # Apply POP-ART, https://arxiv.org/pdf/1602.07714.pdf
                # Save the pre-update running stats
                old_mean = torch.FloatTensor(self.rms_ret.mean).to(self.device)
                old_std = torch.FloatTensor(np.sqrt(self.rms_ret.var) + 1e-8).to(self.device)
                # Update the running stats
                self.rms_ret.update(targ_q)
                # Get the post-update running statistics
                new_mean = torch.FloatTensor(self.rms_ret.mean).to(self.device)
                new_std = torch.FloatTensor(np.sqrt(self.rms_ret.var) + 1e-8).to(self.device)
                # Preserve the output from before the change of normalization old->new
                # for both online and target critic(s)
                outs = [self.crit.out_params, self.targ_crit.out_params]
                if self.hps.clipped_double:
                    outs.extend([self.twin.out_params, self.targ_twin.out_params])
                for out in outs:
                    w, b = out
                    w.data.copy_(w.data * old_std / new_std)
                    b.data.copy_(((b.data * old_std) + old_mean - new_mean) / new_std)

            # Critic loss
            huber_td_errors = F.smooth_l1_loss(q, targ_q, reduction='none')
            if self.hps.clipped_double:
                twin_huber_td_errors = F.smooth_l1_loss(twin_q, targ_q, reduction='none')

            if self.hps.prioritized_replay:
                # Adjust with importance weights
                huber_td_errors *= iws
                if self.hps.clipped_double:
                    twin_huber_td_errors *= iws
                # Update priorities
                td_errors = q - targ_q
                if self.hps.clipped_double:
                    td_errors = torch.min(q - targ_q, twin_q - targ_q)
                new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
                self.replay_buffer.update_priorities(batch['idxs'], new_priorities)

            crit_loss = huber_td_errors.mean()
            if self.hps.clipped_double:
                twin_loss = twin_huber_td_errors.mean()

            # Actor loss
            actr_loss = -self.crit.Q(state, self.actr.act(state)).mean()

        # ######################################################################

        if self.hps.reward_control:
            # Reward control
            actr_loss += 0.1 * (reward - self.actr.rc(state)).pow(2).mean()
            args = [state, action]
            if self.hps.use_iqn:
                args.append(self.hps.num_tau)
            crit_loss += 0.1 * (reward - self.crit.rc(*args)).pow(2).mean()
            if self.hps.clipped_double:
                twin_loss += 0.1 * (reward - self.twin.rc(*args)).pow(2).mean()

        # Compute gradients
        self.actr_opt.zero_grad()
        actr_loss.backward()
        average_gradients(self.actr, self.device)
        actr_gradnorm = U.clip_grad_norm_(self.actr.parameters(), self.hps.clip_norm)
        self.crit_opt.zero_grad()
        crit_loss.backward()
        average_gradients(self.crit, self.device)
        crit_gradnorm = U.clip_grad_norm_(self.crit.parameters(), self.hps.clip_norm)
        if self.hps.clipped_double:
            self.twin_opt.zero_grad()
            twin_loss.backward()
            average_gradients(self.twin, self.device)
            twin_gradnorm = U.clip_grad_norm_(self.twin.parameters(), self.hps.clip_norm)

        # Perform model updates
        if update_critic:
            # Update critic
            self.crit_opt.step()
            self.crit_sched.step(iters_so_far)
            if self.hps.clipped_double:
                # Upsate twin critic
                self.twin_opt.step()
                self.twin_sched.step(iters_so_far)
            if update_actor:
                # Update actor
                self.actr_opt.step()
                self.actr_sched.step(iters_so_far)
                # Update target nets
                self.update_target_net(iters_so_far)
        if hasattr(self, 'disc'):
            for _ in range(self.hps.d_update_ratio):
                for chunk, e_chunk in zip(dataloader, self.e_dataloader):
                    self.update_disc(chunk, e_chunk)
                for chunk, e_chunk in zip(recent_dataloader, self.e_dataloader):
                    self.update_disc(chunk, e_chunk)

        # Aggregate the elements to return
        losses = {'actr': actr_loss.clone().cpu().data.numpy(),
                  'crit': crit_loss.clone().cpu().data.numpy()}
        gradnorms = {'actr': actr_gradnorm,
                     'crit': crit_gradnorm}
        if self.hps.clipped_double:
            losses.update({'twin': twin_loss.clone().cpu().data.numpy()})
            gradnorms.update({'twin': twin_gradnorm})

        lrnows = {'actr': self.actr_sched.get_lr(),
                  'crit': self.crit_sched.get_lr()}
        if self.hps.clipped_double:
            lrnows.update({'twin': self.twin_sched.get_lr()})

        return losses, gradnorms, lrnows

    def update_disc(self, chunk, e_chunk):
        # Create tensors from the inputs
        p_state = torch.FloatTensor(chunk['obs0']).to(self.device)
        p_action = torch.FloatTensor(chunk['acs']).to(self.device)
        e_state = torch.FloatTensor(e_chunk['obs0']).to(self.device)
        e_action = torch.FloatTensor(e_chunk['acs']).to(self.device)
        # Compute scores
        p_scores = self.disc(p_state, p_action)
        e_scores = self.disc(e_state, e_action)
        # Create entropy loss
        scores = torch.cat([p_scores, e_scores], dim=0)
        entropy = F.binary_cross_entropy_with_logits(input=scores,
                                                     target=torch.sigmoid(scores))
        entropy_loss = -self.hps.ent_reg_scale * entropy
        # Create labels
        fake_labels = torch.zeros_like(p_scores).to(self.device)
        real_labels = torch.ones_like(e_scores).to(self.device)
        # Label smoothing, suggested in 'Improved Techniques for Training GANs',
        # Salimans 2016, https://arxiv.org/abs/1606.03498
        # The paper advises on the use of one-sided label smoothing, i.e.
        # only smooth out the positive (real) targets side.
        # Extra comment explanation: https://github.com/openai/improved-gan/blob/
        # 9ff96a7e9e5ac4346796985ddbb9af3239c6eed1/imagenet/build_model.py#L88-L121
        # Additional material: https://github.com/soumith/ganhacks/issues/10
        real_labels.uniform_(0.7, 1.2)
        # Create binary classification (cross-entropy) losses
        p_loss = F.binary_cross_entropy_with_logits(input=p_scores, target=fake_labels)
        e_loss = F.binary_cross_entropy_with_logits(input=e_scores, target=real_labels)
        # Aggregated loss
        d_loss = p_loss + e_loss + entropy_loss

        if self.hps.grad_pen:
            # Create gradient penalty loss (coefficient from the original paper)
            grad_pen = 10. * self.disc.grad_pen(p_state, p_action, e_state, e_action)
            d_loss += grad_pen

        # Update parameters
        self.disc_opt.zero_grad()
        d_loss.backward()
        average_gradients(self.disc, self.device)
        U.clip_grad_norm_(self.disc.parameters(), self.hps.clip_norm)
        self.disc_opt.step()

        if self.hps.overwrite:
            assert hasattr(self, 'disc')
            # Overwrite discriminator reward in buffer with current network predition
            idxs = chunk['idxs'].cpu().numpy().astype(np.int8)
            new_rewards = self.disc.get_reward(p_state, p_action)
            self.replay_buffer.update_rewards(idxs, new_rewards)

    def update_target_net(self, iters_so_far):
        """Update the target networks"""
        if sum([self.hps.use_c51, self.hps.use_qr, self.hps.use_iqn]) == 0:
            # If non-distributional, targets slowly track their non-target counterparts
            for param, targ_param in zip(self.actr.parameters(), self.targ_actr.parameters()):
                targ_param.data.copy_(self.hps.polyak * param.data +
                                      (1. - self.hps.polyak) * targ_param.data)
            for param, targ_param in zip(self.crit.parameters(), self.targ_crit.parameters()):
                targ_param.data.copy_(self.hps.polyak * param.data +
                                      (1. - self.hps.polyak) * targ_param.data)
            if self.hps.clipped_double:
                for param, targ_param in zip(self.twin.parameters(), self.targ_twin.parameters()):
                    targ_param.data.copy_(self.hps.polyak * param.data +
                                          (1. - self.hps.polyak) * targ_param.data)
        else:
            # If distributional, periodically set target weights with online's
            if iters_so_far % self.hps.targ_up_freq == 0:
                self.targ_actr.load_state_dict(self.actr.state_dict())
                self.targ_crit.load_state_dict(self.crit.state_dict())
                if self.hps.clipped_double:
                    self.targ_twin.load_state_dict(self.twin.state_dict())

    def adapt_param_noise(self):
        """Adapt the parameter noise standard deviation"""

        # Perturb separate copy of the policy to adjust the scale for the next 'real' perturbation
        batch = self.replay_buffer.sample(self.hps.batch_size)
        state = torch.FloatTensor(batch['obs0']).to(self.device)
        # Update the perturbable params
        for p in self.actr.perturbable_params:
            param = (self.actr.state_dict()[p]).clone()
            param_ = param.clone()
            noise = param_.data.normal_(0, self.param_noise.cur_std)
            self.apnp_actr.state_dict()[p].data.copy_((param + noise).data)
        # Update the non-perturbable params
        for p in self.actr.non_perturbable_params:
            param = self.actr.state_dict()[p].clone()
            self.apnp_actr.state_dict()[p].data.copy_(param.data)

        # Compute distance between actor and adaptive-parameter-noise-perturbed actor predictions
        self.pn_dist = torch.sqrt(F.mse_loss(self.actr.act(state), self.apnp_actr.act(state)))

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
            for p in self.actr.perturbable_params:
                param = (self.actr.state_dict()[p]).clone()
                param_ = param.clone()
                noise = param_.data.normal_(0, self.param_noise.cur_std)
                self.pnp_actr.state_dict()[p].data.copy_((param + noise).data)
            # Update the non-perturbable params
            for p in self.actr.non_perturbable_params:
                param = self.actr.state_dict()[p].clone()
                self.pnp_actr.state_dict()[p].data.copy_(param.data)

    def save(self, path, iters):
        SaveBundle = namedtuple('SaveBundle', ['model', 'optimizer', 'scheduler'])
        actr_bundle = SaveBundle(
            model=self.actr.state_dict(),
            optimizer=self.actr_opt.state_dict(),
            scheduler=self.actr_sched.state_dict(),
        )
        crit_bundle = SaveBundle(
            model=self.crit.state_dict(),
            optimizer=self.crit_opt.state_dict(),
            scheduler=self.crit_sched.state_dict(),
        )
        torch.save(actr_bundle._asdict(), osp.join(path, "actr_iter{}.pth".format(iters)))
        torch.save(crit_bundle._asdict(), osp.join(path, "crit_iter{}.pth".format(iters)))
        if self.hps.clipped_double:
            twin_bundle = SaveBundle(
                model=self.twin.state_dict(),
                optimizer=self.twin_opt.state_dict(),
                scheduler=self.twin_sched.state_dict(),
            )
            torch.save(twin_bundle._asdict(), osp.join(path, "twin_iter{}.pth".format(iters)))
        if hasattr(self, 'disc'):
            disc_bundle = SaveBundle(
                model=self.disc.state_dict(),
                optimizer=self.disc_opt.state_dict(),
                scheduler=None,
            )
            torch.save(disc_bundle._asdict(), osp.join(path, "disc_iter{}.pth".format(iters)))

    def load(self, path, iters):
        actr_bundle = torch.load(osp.join(path, "actr_iter{}.pth".format(iters)))
        self.actr.load_state_dict(actr_bundle['model'])
        self.actr_opt.load_state_dict(actr_bundle['optimizer'])
        self.actr_sched.load_state_dict(actr_bundle['scheduler'])
        crit_bundle = torch.load(osp.join(path, "crit_iter{}.pth".format(iters)))
        self.crit.load_state_dict(crit_bundle['model'])
        self.crit_opt.load_state_dict(crit_bundle['optimizer'])
        self.crit_sched.load_state_dict(crit_bundle['scheduler'])
        if self.hps.clipped_double:
            twin_bundle = torch.load(osp.join(path, "twin_iter{}.pth".format(iters)))
            self.twin.load_state_dict(twin_bundle['model'])
            self.twin_opt.load_state_dict(twin_bundle['optimizer'])
            self.twin_sched.load_state_dict(twin_bundle['scheduler'])
        if hasattr(self, 'disc'):
            disc_bundle = torch.load(osp.join(path, "disc_iter{}.pth".format(iters)))
            self.disc.load_state_dict(disc_bundle['model'])
            self.disc_opt.load_state_dict(disc_bundle['optimizer'])
