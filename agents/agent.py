from collections import namedtuple, defaultdict
import os.path as osp
import math

import numpy as np
import torch
import torch.nn.utils as U
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import autograd
from torch.autograd import Variable

from helpers import logger
from helpers.console_util import log_env_info, log_module_info
from helpers.dataset import Dataset
from helpers.math_util import huber_quant_reg_loss
from helpers.distributed_util import average_gradients, sync_with_root
from agents.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer
from agents.nets import Actor, Critic, Discriminator, KYEDiscriminator
from agents.param_noise import AdaptiveParamNoise
from agents.ac_noise import NormalAcNoise, OUAcNoise


class Agent(object):

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
        assert self.hps.rollout_len <= self.hps.batch_size
        assert sum([self.hps.kye_p_binning, self.hps.kye_p_regress]) in [0, 1], "not both"
        assert self.hps.clip_norm >= 0
        if self.hps.clip_norm <= 0:
            logger.info("[WARN] clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm))

        # Define demo dataset
        self.expert_dataset = expert_dataset
        self.eval_mode = self.expert_dataset is None

        # Define action clipping range
        assert all(self.ac_space.low == -self.ac_space.high)
        self.max_ac = self.ac_space.high[0].astype('float32')
        assert all(ac_comp == self.max_ac for ac_comp in self.ac_space.high)

        # Define critic to use
        assert sum([self.hps.use_c51, self.hps.use_qr]) <= 1
        if self.hps.use_c51:
            assert not self.hps.clipped_double
            c51_supp_range = (self.hps.c51_vmin,
                              self.hps.c51_vmax,
                              self.hps.c51_num_atoms)
            self.c51_supp = torch.linspace(*c51_supp_range).to(self.device)
            self.c51_delta = ((self.hps.c51_vmax - self.hps.c51_vmin) /
                              (self.hps.c51_num_atoms - 1))
        elif self.hps.use_qr:
            assert not self.hps.clipped_double
            qr_cum_density = np.array([((2 * i) + 1) / (2.0 * self.hps.num_tau)
                                       for i in range(self.hps.num_tau)])
            qr_cum_density = torch.Tensor(qr_cum_density).to(self.device)
            self.qr_cum_density = qr_cum_density.view(1, 1, -1, 1).expand(self.hps.batch_size,
                                                                          self.hps.num_tau,
                                                                          self.hps.num_tau,
                                                                          -1).to(self.device)

        # Parse the noise types
        self.param_noise, self.ac_noise = self.parse_noise_type(self.hps.noise_type)

        # Parse the label smoothing types
        self.apply_ls_fake = self.parse_label_smoothing_type(self.hps.fake_ls_type)
        self.apply_ls_real = self.parse_label_smoothing_type(self.hps.real_ls_type)

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
        if self.hps.wrap_absorb:
            ob_dim = self.ob_dim + 1
            ac_dim = self.ac_dim + 1
        else:
            ob_dim = self.ob_dim
            ac_dim = self.ac_dim
        self.replay_buffer = self.setup_replay_buffer({
            "obs0": (ob_dim,),
            "obs1": (ob_dim,),
            "acs": (ac_dim,),
            "rews": (1,),
            "dones1": (1,),
        })

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

        if not self.eval_mode:
            # Set up demonstrations dataset
            self.e_batch_size = min(len(self.expert_dataset), self.hps.batch_size)
            self.e_dataloader = DataLoader(
                self.expert_dataset,
                self.e_batch_size,
                shuffle=True,
                drop_last=True,
            )
            assert len(self.e_dataloader) > 0
            # Create discriminator
            if self.hps.kye_d_regress:
                self.disc = KYEDiscriminator(self.env, self.hps).to(self.device)
            else:
                self.disc = Discriminator(self.env, self.hps).to(self.device)
            sync_with_root(self.disc)
            # Create optimizer
            self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.hps.d_lr)

        log_module_info(logger, 'actr', self.actr)
        log_module_info(logger, 'crit', self.crit)
        if self.hps.clipped_double:
            log_module_info(logger, 'twin', self.crit)
        if not self.eval_mode:
            log_module_info(logger, 'disc', self.disc)

    def parse_noise_type(self, noise_type):
        """Parse the `noise_type` hyperparameter"""
        ac_noise = None
        param_noise = None
        logger.info("[INFO] parsing noise type")
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
                param_noise = AdaptiveParamNoise(initial_std=float(std), delta=float(std))
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
                raise RuntimeError("unknown noise type: '{}'".format(cur_noise_type))
        return param_noise, ac_noise

    def parse_label_smoothing_type(self, ls_type):
        """Parse the `label_smoothing_type` hyperparameter"""
        logger.info("[INFO] parsing label smoothing type")
        if ls_type == 'none':

            def _apply(labels):
                pass

        elif 'random-uniform' in ls_type:
            # Label smoothing, suggested in 'Improved Techniques for Training GANs',
            # Salimans 2016, https://arxiv.org/abs/1606.03498
            # The paper advises on the use of one-sided label smoothing, i.e.
            # only smooth out the positive (real) targets side.
            # Extra comment explanation: https://github.com/openai/improved-gan/blob/
            # 9ff96a7e9e5ac4346796985ddbb9af3239c6eed1/imagenet/build_model.py#L88-L121
            # Additional material: https://github.com/soumith/ganhacks/issues/10
            _, lb, ub = ls_type.split('_')

            def _apply(labels):
                # Replace labels by uniform noise from the interval
                labels.uniform_(float(lb), float(ub))

        elif 'soft-labels' in ls_type:
            # Traditional soft labels, giving confidence to wrong classes uniformly (all)
            _, alpha = ls_type.split('_')

            def _apply(labels):
                labels.data.copy_((labels * (1. - float(alpha))) + (float(alpha) / 2.))

        elif 'disturb-label' in ls_type:
            # DisturbLabel paper: disturb the label of each sample with probability alpha.
            # For each disturbed sample, the label is randomly drawn from a uniform distribution
            # over the whole label set, regarless of the true label.
            _, alpha = ls_type.split('_')

            def _apply(labels):
                flip = (labels.clone().detach().data.uniform_() <= float(alpha)).float()
                labels.data.copy_(torch.abs(labels.data - flip.data))

        else:
            raise RuntimeError("unknown label smoothing type: '{}'".format(ls_type))
        return _apply

    def setup_replay_buffer(self, shapes):
        """Setup experiental memory unit"""
        logger.info(">>>> setting up replay buffer")
        # Create the buffer
        if self.hps.prioritized_replay:
            if self.hps.unreal:  # Unreal prioritized experience replay
                replay_buffer = UnrealReplayBuffer(
                    self.hps.mem_size,
                    shapes,
                )
            else:  # Vanilla prioritized experience replay
                replay_buffer = PrioritizedReplayBuffer(
                    self.hps.mem_size,
                    shapes,
                    alpha=self.hps.alpha,
                    beta=self.hps.beta,
                    ranked=self.hps.ranked,
                )
        else:  # Vanilla experience replay
            replay_buffer = ReplayBuffer(
                self.hps.mem_size,
                shapes,
            )
        # Summarize replay buffer creation (relies on `__repr__` method)
        logger.info("[INFO] {} configured".format(replay_buffer))
        return replay_buffer

    def store_transition(self, transition):
        """Store the transition in memory and update running moments"""
        # Store transition in the replay buffer
        self.replay_buffer.append(transition)
        # Update the running moments for all the networks (online and targets)
        _state = transition['obs0']
        if self.hps.wrap_absorb:
            if np.all(np.equal(_state, np.append(np.zeros_like(_state[0:-1]), 1.))):
                # logger.info("[INFO] absorbing -> not using it to update rms_obs")
                return
            _state = _state[0:-1]
        self.actr.rms_obs.update(_state)
        self.crit.rms_obs.update(_state)
        self.targ_actr.rms_obs.update(_state)
        self.targ_crit.rms_obs.update(_state)
        if self.hps.clipped_double:
            self.twin.rms_obs.update(_state)
            self.targ_twin.rms_obs.update(_state)
        if self.param_noise is not None:
            self.pnp_actr.rms_obs.update(_state)
            self.apnp_actr.rms_obs.update(_state)

    def sample_batch(self):
        """Sample a batch of transitions from the replay buffer"""
        # Create patcher if needed
        patcher = None
        if self.hps.historical_patching:

            def patcher(x, y, z):
                return self.get_reward(x, y, z)[0].cpu().numpy()

        # Get a batch of transitions from the replay buffer
        if self.hps.n_step_returns:
            batch = self.replay_buffer.lookahead_sample(
                self.hps.batch_size,
                self.hps.lookahead,
                self.hps.gamma,
                patcher=patcher,
            )
        else:
            batch = self.replay_buffer.sample(
                self.hps.batch_size,
                patcher=patcher,
            )
        return batch

    def predict(self, ob, apply_noise):
        """Predict an action, with or without perturbation,
        and optionaly compute and return the associated QZ value.
        """
        # Create tensor from the state (`require_grad=False` by default)
        ob = torch.Tensor(ob[None]).to(self.device)
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

    def remove_absorbing(self, x):
        non_absorbing_rows = []
        for j, row in enumerate([x[i, :] for i in range(x.shape[0])]):
            if torch.all(torch.eq(row, torch.cat([torch.zeros_like(row[0:-1]),
                                                  torch.Tensor([1.]).to(self.device)], dim=-1))):
                # logger.info("[INFO] removing absorbing row (#{})".format(j))
                pass
            else:
                non_absorbing_rows.append(j)
        return x[non_absorbing_rows, :], non_absorbing_rows

    def update_actor_critic(self, batch, update_actor, iters_so_far):
        """Train the actor and critic networks"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Create tensors from the inputs
        state = torch.Tensor(batch['obs0']).to(self.device)
        action = torch.Tensor(batch['acs']).to(self.device)
        reward = torch.Tensor(batch['rews']).to(self.device)
        next_state = torch.Tensor(batch['obs1']).to(self.device)
        done = torch.Tensor(batch['dones1'].astype('float32')).to(self.device)
        if self.hps.prioritized_replay:
            iws = torch.Tensor(batch['iws']).to(self.device)
        if self.hps.n_step_returns:
            td_len = torch.Tensor(batch['td_len']).to(self.device)
        else:
            td_len = torch.ones_like(done).to(self.device)

        if self.hps.wrap_absorb:
            _, indices_a = self.remove_absorbing(state)
            _, indices_b = self.remove_absorbing(next_state)
            indices = sorted(list(set(indices_a) & set(indices_b)))  # intersection
            state = state[indices, 0:-1]
            action = action[indices, 0:-1]
            reward = reward[indices, :]
            next_state = next_state[indices, 0:-1]
            done = done[indices, :]
            if self.hps.prioritized_replay:
                iws = iws[indices, :]
            if self.hps.n_step_returns:
                td_len = td_len[indices, :]
            else:
                td_len = torch.ones_like(done).to(self.device)

        if self.hps.targ_actor_smoothing:
            n_ = action.clone().detach().data.normal_(0, self.hps.td3_std).to(self.device)
            n_ = n_.clamp(-self.hps.td3_c, self.hps.td3_c)
            next_action = (self.targ_actr.act(next_state) + n_).clamp(-self.max_ac, self.max_ac)
        else:
            next_action = self.targ_actr.act(next_state)

        # Compute losses

        if self.hps.use_c51:

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> C51.

            # Compute QZ estimate
            z = self.crit.QZ(state, action).unsqueeze(-1)
            z.data.clamp_(0.01, 0.99)
            # Compute target QZ estimate
            z_prime = self.targ_crit.QZ(next_state, next_action)
            z_prime.data.clamp_(0.01, 0.99)

            gamma_mask = ((self.hps.gamma ** td_len) * (1 - done))
            Tz = reward + (gamma_mask * self.c51_supp.view(1, self.hps.c51_num_atoms))
            Tz = Tz.clamp(self.hps.c51_vmin, self.hps.c51_vmax)
            b = (Tz - self.hps.c51_vmin) / self.c51_delta
            l = b.floor().long()  # noqa
            u = b.ceil().long()
            targ_z = z_prime.clone().zero_()
            z_prime_l = z_prime * (u + (l == u).float() - b)  # noqa
            z_prime_u = z_prime * (b - l.float())  # noqa
            for i in range(targ_z.size(0)):
                targ_z[i].index_add_(0, l[i], z_prime_l[i])
                targ_z[i].index_add_(0, u[i], z_prime_u[i])

            # Reshape target to be of shape [batch_size, self.hps.c51_num_atoms, 1]
            targ_z = targ_z.view(-1, self.hps.c51_num_atoms, 1)

            # Critic loss
            ce_losses = -(targ_z.detach() * torch.log(z)).sum(dim=1)

            if self.hps.prioritized_replay:
                # Update priorities
                new_priorities = np.abs(ce_losses.sum(dim=1).detach().cpu().numpy()) + 1e-6
                self.replay_buffer.update_priorities(batch['idxs'].reshape(-1), new_priorities)
                # Adjust with importance weights
                ce_losses *= iws

            crit_loss = ce_losses.mean()

            # Actor loss
            actr_loss = -self.crit.QZ(state, self.actr.act(state))  # [batch_size, num_atoms]
            actr_loss = actr_loss.matmul(self.c51_supp).unsqueeze(-1)  # [batch_size, 1]

            actr_loss = actr_loss.mean()

        elif self.hps.use_qr:

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> QR.

            # Compute QZ estimate
            z = self.crit.QZ(state, action).unsqueeze(-1)

            # Compute target QZ estimate
            z_prime = self.targ_crit.QZ(next_state, next_action)
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
                self.replay_buffer.update_priorities(batch['idxs'].reshape(-1), new_priorities)

            # Sum over current quantile value (tau, N in paper) dimension, and
            # average over target quantile value (tau prime, N' in paper) dimension.
            crit_loss = huber_td_errors.sum(dim=2)
            # Resulting shape is [batch_size, num_tau_prime, 1]
            crit_loss = crit_loss.mean(dim=1)
            # Resulting shape is [batch_size, 1]
            # Average across the minibatch
            crit_loss = crit_loss.mean()

            # Actor loss
            actr_loss = -self.crit.QZ(state, self.actr.act(state)).mean()

        else:

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VANILLA.

            # Compute QZ estimate
            q = self.crit.QZ(state, action)
            if self.hps.clipped_double:
                twin_q = self.twin.QZ(state, action)

            # Compute target QZ estimate
            q_prime = self.targ_crit.QZ(next_state, next_action)
            if self.hps.clipped_double:
                # Define QZ' as the minimum QZ value between TD3's twin QZ's
                twin_q_prime = self.targ_twin.QZ(next_state, next_action)
                q_prime = (0.75 * torch.min(q_prime, twin_q_prime) +
                           0.25 * torch.max(q_prime, twin_q_prime))  # soft minimum from BCQ
            targ_q = reward + (self.hps.gamma ** td_len) * (1. - done) * q_prime.detach()

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
                new_priorities = np.abs((q - targ_q).detach().cpu().numpy()) + 1e-6
                self.replay_buffer.update_priorities(batch['idxs'].reshape(-1), new_priorities)

            crit_loss = huber_td_errors.mean()
            if self.hps.clipped_double:
                twin_loss = twin_huber_td_errors.mean()

            # Actor loss
            actr_loss = -self.crit.QZ(state, self.actr.act(state)).mean()

        # Log metrics
        metrics['actr_loss'].append(actr_loss)
        metrics['crit_loss'].append(crit_loss)
        if self.hps.clipped_double:
            metrics['twin_loss'].append(twin_loss)
        if self.hps.prioritized_replay:
            metrics['iws'].append(iws)

        # Update parameters
        self.actr_opt.zero_grad()
        actr_loss.backward()
        average_gradients(self.actr, self.device)
        if self.hps.clip_norm > 0:
            U.clip_grad_norm_(self.actr.parameters(), self.hps.clip_norm)
        self.crit_opt.zero_grad()
        crit_loss.backward()
        average_gradients(self.crit, self.device)
        if self.hps.clipped_double:
            self.twin_opt.zero_grad()
            twin_loss.backward()
            average_gradients(self.twin, self.device)
        self.crit_opt.step()
        self.crit_sched.step(iters_so_far)
        if self.hps.clipped_double:
            self.twin_opt.step()
            self.twin_sched.step(iters_so_far)
        if update_actor:
            self.actr_opt.step()
            self.actr_sched.step(iters_so_far)
            # Update target nets
            self.update_target_net(iters_so_far)  # not an error

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}

        lrnows = {'actr': self.actr_sched.get_last_lr(),
                  'crit': self.crit_sched.get_last_lr()}
        if self.hps.clipped_double:
            lrnows.update({'twin': self.twin_sched.get_last_lr()})

        return metrics, lrnows

    def update_reward_control(self, batch, iters_so_far):
        """Update the policy and value networks"""

        # Transfer to device
        _state = torch.Tensor(batch['obs0']).to(self.device)
        state = torch.Tensor(batch['obs0']).to(self.device)
        next_state = torch.Tensor(batch['obs1']).to(self.device)
        action = torch.Tensor(batch['acs']).to(self.device)
        if self.hps.wrap_absorb:
            _, indices_a = self.remove_absorbing(state)
            _, indices_b = self.remove_absorbing(next_state)
            indices = sorted(list(set(indices_a) & set(indices_b)))  # intersection
            _state = _state[indices, 0:-1]
            state = state[indices, :]
            next_state = next_state[indices, :]
            action = action[indices, :]

        if self.hps.kye_mixing:
            # Get a minibatch of expert data
            e_batch = next(iter(self.e_dataloader))
            _state_e = e_batch['obs0']
            state_e = e_batch['obs0']
            next_state_e = e_batch['obs1']
            action_e = e_batch['acs']
            if self.hps.wrap_absorb:
                _, indices_a = self.remove_absorbing(state_e)
                _, indices_b = self.remove_absorbing(next_state_e)
                indices = sorted(list(set(indices_a) & set(indices_b)))  # intersection
                _state_e = _state_e[indices, 0:-1]
                state_e = state_e[indices, :]
                next_state_e = next_state_e[indices, :]
                action_e = action_e[indices, :]

        if self.hps.kye_p_binning:
            aux_loss = F.cross_entropy(
                input=self.actr.auxo(_state),
                target=self.get_reward(state, action, next_state)[1]
            )
            if self.hps.kye_mixing:
                aux_loss += F.cross_entropy(
                    input=self.actr.auxo(_state_e),
                    target=self.get_reward(state_e, action_e, next_state_e)[1]
                )
        elif self.hps.kye_p_regress:
            aux_loss = F.smooth_l1_loss(
                input=self.actr.auxo(_state),
                target=self.get_reward(state, action, next_state)[0]
            )
            if self.hps.kye_mixing:
                aux_loss += F.smooth_l1_loss(
                    input=self.actr.auxo(_state_e),
                    target=self.get_reward(state_e, action_e, next_state_e)[0]
                )

        aux_loss *= self.hps.kye_p_scale

        # Update parameters
        self.actr_opt.zero_grad()
        aux_loss.backward()
        average_gradients(self.actr, self.device)
        if self.hps.clip_norm > 0:
            U.clip_grad_norm_(self.actr.parameters(), self.hps.clip_norm)
        self.actr_opt.step()

    def update_discriminator(self, batch):
        """Update the discriminator network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Create DataLoader object to iterate over transitions in rollouts
        d_keys = ['obs0', 'acs']
        d_dataset = Dataset({k: batch[k] for k in d_keys})
        d_dataloader = DataLoader(
            d_dataset,
            self.e_batch_size,
            shuffle=True,
            drop_last=True,
        )

        for e_batch in self.e_dataloader:

            # Get a minibatch of policy data
            d_batch = next(iter(d_dataloader))

            # Transfer to device
            p_state = d_batch['obs0'].to(self.device)
            p_action = d_batch['acs'].to(self.device)
            e_state = e_batch['obs0'].to(self.device)
            e_action = e_batch['acs'].to(self.device)

            # Update running moments
            _state = torch.cat([p_state, e_state], dim=0)
            if self.hps.wrap_absorb:
                _state = self.remove_absorbing(_state)[0][:, 0:-1]
            self.disc.rms_obs.update(_state)

            # Compute scores
            p_scores = self.disc.D(p_state, p_action)
            e_scores = self.disc.D(e_state, e_action)

            # Create entropy loss
            scores = torch.cat([p_scores, e_scores], dim=0)
            entropy = F.binary_cross_entropy_with_logits(input=scores, target=torch.sigmoid(scores))
            entropy_loss = -self.hps.ent_reg_scale * entropy

            # Create labels
            fake_labels = 0. * torch.ones_like(p_scores).to(self.device)
            real_labels = 1. * torch.ones_like(e_scores).to(self.device)

            # Parse and apply label smoothing
            self.apply_ls_fake(fake_labels)
            self.apply_ls_real(real_labels)

            if self.hps.use_purl:
                # Create positive-unlabeled binary classification (cross-entropy) losses
                beta = 0.0  # hard-coded, using standard value from the original paper
                p_e_loss = -self.hps.purl_eta * torch.log(1. - torch.sigmoid(e_scores) + 1e-8)
                p_e_loss += -torch.max(-beta * torch.ones_like(p_scores),
                                       (F.logsigmoid(e_scores) -
                                        (self.hps.purl_eta * F.logsigmoid(p_scores))))
            else:
                # Create positive-negative binary classification (cross-entropy) losses
                p_loss = F.binary_cross_entropy_with_logits(input=p_scores,
                                                            target=fake_labels,
                                                            reduction='none')
                e_loss = F.binary_cross_entropy_with_logits(input=e_scores,
                                                            target=real_labels,
                                                            reduction='none')
                p_e_loss = p_loss + e_loss
            # Averate out over the batch
            p_e_loss = p_e_loss.mean()

            # Aggregated loss
            d_loss = p_e_loss + entropy_loss

            # Log metrics
            metrics['entropy_loss'].append(entropy_loss)
            metrics['p_e_loss'].append(p_e_loss)

            if self.hps.grad_pen:
                # Create gradient penalty loss (coefficient from the original paper)
                grad_pen_in = [p_state, p_action, e_state, e_action]
                grad_pen = 10. * self.grad_pen(*grad_pen_in)
                d_loss += grad_pen
                # Log metrics
                metrics['grad_pen'].append(grad_pen)

            if self.hps.kye_d_regress:
                # Compute gradient of the feature exctractor for d_loss
                self.disc_opt.zero_grad()
                d_loss.backward(retain_graph=True)
                if self.hps.spectral_norm:
                    grads_d_loss = self.disc.ob_encoder.fc_block.fc.weight_orig.grad
                else:
                    grads_d_loss = self.disc.ob_encoder.fc_block.fc.weight.grad
                grads_d_loss = grads_d_loss.mean(dim=0)
                self.disc_opt.zero_grad()
                # Create and add auxiliary loss
                if self.hps.wrap_absorb:
                    _, p_indices = self.remove_absorbing(p_state)
                    _p_state = p_state[p_indices, 0:-1]
                    p_state = p_state[p_indices, :]
                    p_action = p_action[p_indices, :]
                else:
                    _p_state = p_state
                aux_loss = F.smooth_l1_loss(
                    input=self.disc.auxo(p_state, p_action),
                    target=self.actr.act(_p_state)
                )
                if self.hps.kye_mixing:
                    if self.hps.wrap_absorb:
                        _, e_indices = self.remove_absorbing(e_state)
                        _e_state = e_state[e_indices, 0:-1]
                        e_state = e_state[e_indices, :]
                        e_action = e_action[e_indices, :]
                    else:
                        _e_state = e_state
                    aux_loss += F.smooth_l1_loss(
                        input=self.disc.auxo(e_state, e_action),
                        target=self.actr.act(_e_state)
                    )
                # Compute gradient of the feature exctractor for aux_loss
                self.disc_opt.zero_grad()
                aux_loss.backward(retain_graph=True)
                if self.hps.spectral_norm:
                    grads_aux_loss = self.disc.ob_encoder.fc_block.fc.weight_orig.grad
                else:
                    grads_aux_loss = self.disc.ob_encoder.fc_block.fc.weight.grad
                grads_aux_loss = grads_aux_loss.mean(dim=0)
                self.disc_opt.zero_grad()
                # Compute the angle between the two computed gradients
                angle = torch.acos(torch.dot(grads_d_loss, grads_aux_loss) /
                                   (torch.norm(grads_d_loss) * torch.norm(grads_aux_loss)))
                angle = angle.detach()
                angle = angle / math.pi * 180.
                # Log metrics
                metrics['aux_loss'].append(aux_loss)
                metrics['angle'].append(angle)
                # Assemble losses
                aux_loss *= self.hps.kye_d_scale
                d_loss += aux_loss
                # Log metrics
                metrics['aux_loss_scaled'].append(aux_loss)

            metrics['disc_loss'].append(d_loss)

            # Update parameters
            self.disc_opt.zero_grad()
            d_loss.backward()
            average_gradients(self.disc, self.device)
            self.disc_opt.step()

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics

    def grad_pen(self, p_ob, p_ac, e_ob, e_ac):
        """Gradient penalty regularizer (motivation from Wasserstein GANs (Gulrajani),
        but empirically useful in JS-GANs (Lucic et al. 2017)) and later in (Karol et al. 2018).
        """
        # Assemble interpolated state-action pair
        if self.hps.wrap_absorb:
            ob_dim = self.ob_dim + 1
            ac_dim = self.ac_dim + 1
        else:
            ob_dim = self.ob_dim
            ac_dim = self.ac_dim
        ob_eps = torch.rand(ob_dim).to(p_ob.device)
        ac_eps = torch.rand(ac_dim).to(p_ob.device)
        ob_interp = ob_eps * p_ob + ((1. - ob_eps) * e_ob)
        ac_interp = ac_eps * p_ac + ((1. - ac_eps) * e_ac)
        # Set `requires_grad=True` to later have access to
        # gradients w.r.t. the inputs (not populated by default)
        ob_interp = Variable(ob_interp, requires_grad=True)
        ac_interp = Variable(ac_interp, requires_grad=True)
        # Create the operation of interest
        score = self.disc.D(ob_interp, ac_interp)
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

    def get_reward(self, curr_ob, ac, next_ob):
        # Define the obeservation to get the reward of
        ob = next_ob if self.hps.state_only else curr_ob
        # Craft surrogate reward
        assert sum([isinstance(x, torch.Tensor) for x in [ob, ac]]) in [0, 2]
        if not isinstance(ob, torch.Tensor):  # then ac is not neither
            ob = torch.Tensor(ob)
            ac = torch.Tensor(ac)
        # Transfer to cpu
        ob = ob.cpu()
        ac = ac.cpu()
        # Compure score
        score = self.disc.D(ob, ac).detach().view(-1, 1)
        sigscore = torch.sigmoid(score)  # squashed in [0, 1]
        # Counterpart of GAN's minimax (also called "saturating") loss
        # Numerics: 0 for non-expert-like states, goes to +inf for expert-like states
        # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
        # e.g. walking simulations that get cut off when the robot falls over
        minimax_reward = -torch.log(1. - torch.sigmoid(score) + 1e-8)
        if self.hps.minimax_only:
            reward = minimax_reward
        else:
            # Counterpart of GAN's non-saturating loss
            # Recommended in the original GAN paper and later in (Fedus et al. 2017)
            # Numerics: 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            non_satur_reward = F.logsigmoid(score)
            # Return the sum the two previous reward functions (as in AIRL, Fu et al. 2018)
            # Numerics: might be better might be way worse
            reward = non_satur_reward + minimax_reward
        # Perform binning
        num_bins = 3  # arbitrarily
        binned = (torch.abs(sigscore - 1e-8) // (1 / num_bins)).long().squeeze(-1)
        for i in range(binned.size(0)):
            if binned.view(-1)[i] > 2. or binned.view(-1)[i] < 0.:
                # This should never happen, flag, but don't rupt
                logger.info("[WARN] binned.view(-1)[{}]={}.".format(i, binned.view(-1)[i]))
        return self.hps.syn_rew_scale * reward, binned

    def update_target_net(self, iters_so_far):
        """Update the target networks"""
        if sum([self.hps.use_c51, self.hps.use_qr]) == 0:
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
        batch = self.replay_buffer.sample(self.hps.batch_size, patcher=None)
        state = torch.Tensor(batch['obs0']).to(self.device)
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
        if self.hps.wrap_absorb:
            state = self.remove_absorbing(state)[0][:, 0:-1]
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
        if not self.eval_mode:
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
        if not self.eval_mode:
            disc_bundle = torch.load(osp.join(path, "disc_iter{}.pth".format(iters)))
            self.disc.load_state_dict(disc_bundle['model'])
            self.disc_opt.load_state_dict(disc_bundle['optimizer'])
