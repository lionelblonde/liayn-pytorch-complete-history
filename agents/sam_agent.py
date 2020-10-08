from collections import defaultdict
import os.path as osp

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
from helpers.math_util import huber_quant_reg_loss, LRScheduler
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from agents.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer
from agents.nets import Actor, Critic, Discriminator
from agents.param_noise import AdaptiveParamNoise
from agents.ac_noise import NormalAcNoise, OUAcNoise

from agents.red import RandomExpertDistillation
from agents.kye import KnowYourEnemy
from agents.dyn import Forward


class SAMAgent(object):

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
        if self.hps.clip_norm <= 0:
            logger.info("clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm))

        # Define demo dataset
        self.expert_dataset = expert_dataset
        self.eval_mode = self.expert_dataset is None

        # Define action clipping range
        self.max_ac = max(np.abs(np.amax(self.ac_space.high.astype('float32'))),
                          np.abs(np.amin(self.ac_space.low.astype('float32'))))

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
        shapes = {
            'obs0': (ob_dim,),
            'obs1': (ob_dim,),
            'acs': (ac_dim,),
            'rews': (1,),
            'dones1': (1,),
        }

        if self.hps.wrap_absorb:
            shapes.update({
                'obs0_orig': (self.ob_dim,),
                'obs1_orig': (self.ob_dim,),
                'acs_orig': (self.ac_dim,),
            })
        self.replay_buffer = self.setup_replay_buffer(shapes)

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

        # Set up lr scheduler
        self.actr_sched = LRScheduler(
            optimizer=self.actr_opt,
            initial_lr=self.hps.actor_lr,
            lr_schedule=self.hps.lr_schedule,
            total_num_steps=self.hps.num_timesteps,
        )

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
            self.disc = Discriminator(self.env, self.hps).to(self.device)
            sync_with_root(self.disc)
            # Create optimizer
            self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.hps.d_lr)

        assert self.hps.ret_norm or not self.hps.popart
        assert not (self.hps.use_c51 and self.hps.ret_norm)
        assert not (self.hps.use_qr and self.hps.ret_norm)
        if self.hps.ret_norm:
            # Create return normalizer that maintains running statistics
            self.rms_ret = RunMoms(shape=(1,), use_mpi=False)  # Careful, set to False here

        log_module_info(logger, 'actr', self.actr)
        log_module_info(logger, 'crit', self.crit)
        if self.hps.clipped_double:
            log_module_info(logger, 'twin', self.crit)
        if not self.eval_mode:
            log_module_info(logger, 'disc', self.disc)

        if self.hps.reward_type in ['red', 'gail_red_mod', 'gail_red_grad_mod']:
            # Create nets
            self.red = RandomExpertDistillation(
                self.env,
                self.device,
                self.hps,
                self.expert_dataset,
            )
            # Train the predictor network on expert dataset
            logger.info(">>>> RED training: BEG")
            for epoch in range(int(self.hps.red_epochs)):
                metrics, _ = self.red.train()
                logger.info("epoch: {}/{} | loss: {}".format(str(epoch).zfill(3),
                                                             self.hps.red_epochs,
                                                             metrics['loss']))
            logger.info(">>>> RED training: END")

        if self.hps.reward_type == 'gail_kye_mod':
            self.kye = KnowYourEnemy(
                self.env,
                self.device,
                self.hps,
            )

        if self.hps.reward_type == 'gail_dyn_mod':
            self.dyn = Forward(
                self.env,
                self.device,
                self.hps,
            )

        if self.hps.reward_type in ['gail_grad_mod', 'gail_red_grad_mod']:
            self.grad_pen_store = []
            self.rms_grad_pens = RunMoms(shape=(1,), use_mpi=False)

    def norm_rets(self, x):
        """Standardize if return normalization is used, do nothing otherwise"""
        if self.hps.ret_norm:
            return self.rms_ret.standardize(x)
        else:
            return x

    def denorm_rets(self, x):
        """Standardize if return denormalization is used, do nothing otherwise"""
        if self.hps.ret_norm:
            return self.rms_ret.destandardize(x)
        else:
            return x

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
                _, std = cur_noise_type.split('_')
                param_noise = AdaptiveParamNoise(initial_std=float(std), delta=float(std))
                logger.info("{} configured".format(param_noise))
            elif 'normal' in cur_noise_type:
                _, std = cur_noise_type.split('_')
                # Spherical (isotropic) gaussian action noise
                ac_noise = NormalAcNoise(mu=np.zeros(self.ac_dim),
                                         sigma=float(std) * np.ones(self.ac_dim))
                logger.info("{} configured".format(ac_noise))
            elif 'ou' in cur_noise_type:
                _, std = cur_noise_type.split('_')
                # Ornstein-Uhlenbeck action noise
                ac_noise = OUAcNoise(mu=np.zeros(self.ac_dim),
                                     sigma=(float(std) * np.ones(self.ac_dim)))
                logger.info("{} configured".format(ac_noise))
            else:
                raise RuntimeError("unknown noise type: '{}'".format(cur_noise_type))
        return param_noise, ac_noise

    def parse_label_smoothing_type(self, ls_type):
        """Parse the `label_smoothing_type` hyperparameter"""
        logger.info("parsing label smoothing type")
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
        logger.info("{} configured".format(replay_buffer))
        return replay_buffer

    def store_transition(self, transition):
        """Store the transition in memory and update running moments"""
        # Store transition in the replay buffer
        self.replay_buffer.append(transition)
        # Update the running moments for all the networks (online and targets)
        _state = transition['obs0']
        if self.hps.wrap_absorb:
            if np.all(np.equal(_state, np.append(np.zeros_like(_state[0:-1]), 1.))):
                # logger.info("absorbing -> not using it to update rms_obs")
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
        if self.hps.d_batch_norm:
            self.disc.rms_obs.update(_state)  # only with agent samples, no expert ones

        if self.hps.reward_type == 'gail_kye_mod' and self.hps.kye_batch_norm:
            self.kye.pred_net.rms_obs.update(_state)
        if self.hps.reward_type == 'gail_dyn_mod' and self.hps.dyn_batch_norm:
            self.dyn.pred_net.rms_obs.update(_state)

    def sample_batch(self):
        """Sample a batch of transitions from the replay buffer"""
        # Create patcher if needed
        patcher = None
        if self.hps.historical_patching:

            def patcher(x, y, z):
                return self.get_syn_rew(x, y, z).detach().cpu().numpy()  # redundant detach

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
                # logger.info("removing absorbing row (#{})".format(j))
                pass
            else:
                non_absorbing_rows.append(j)
        return x[non_absorbing_rows, :], non_absorbing_rows

    def update_actor_critic(self, batch, update_actor, iters_so_far):
        """Train the actor and critic networks"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Transfer to device
        if self.hps.wrap_absorb:
            state = torch.Tensor(batch['obs0_orig']).to(self.device)
            action = torch.Tensor(batch['acs_orig']).to(self.device)
            next_state = torch.Tensor(batch['obs1_orig']).to(self.device)
            if self.hps.reward_type in ['gail_dyn_mod', 'gail_kye_mod']:
                state_a = torch.Tensor(batch['obs0']).to(self.device)
                action_a = torch.Tensor(batch['acs']).to(self.device)
                next_state_a = torch.Tensor(batch['obs1_td1']).to(self.device)  # not n-step next!
        else:
            state = torch.Tensor(batch['obs0']).to(self.device)
            action = torch.Tensor(batch['acs']).to(self.device)
            next_state = torch.Tensor(batch['obs1']).to(self.device)
        reward = torch.Tensor(batch['rews']).to(self.device)
        done = torch.Tensor(batch['dones1'].astype('float32')).to(self.device)
        if self.hps.prioritized_replay:
            iws = torch.Tensor(batch['iws']).to(self.device)
        if self.hps.n_step_returns:
            td_len = torch.Tensor(batch['td_len']).to(self.device)
        else:
            td_len = torch.ones_like(done).to(self.device)

        if self.hps.targ_actor_smoothing:
            n_ = action.clone().detach().data.normal_(0., self.hps.td3_std).to(self.device)
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
            _actr_loss = -self.crit.QZ(state, self.actr.act(state))  # [batch_size, num_atoms]
            _actr_loss = _actr_loss.matmul(self.c51_supp).unsqueeze(-1)  # [batch_size, 1]

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
            _actr_loss = -self.crit.QZ(state, self.actr.act(state))

        else:

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VANILLA.

            # Compute QZ estimate
            q = self.denorm_rets(self.crit.QZ(state, action))
            if self.hps.clipped_double:
                twin_q = self.denorm_rets(self.twin.QZ(state, action))

            # Compute target QZ estimate
            q_prime = self.targ_crit.QZ(next_state, next_action)
            if self.hps.clipped_double:
                # Define QZ' as the minimum QZ value between TD3's twin QZ's
                twin_q_prime = self.targ_twin.QZ(next_state, next_action)
                q_prime = (0.75 * torch.min(q_prime, twin_q_prime) +
                           0.25 * torch.max(q_prime, twin_q_prime))  # soft minimum from BCQ
            targ_q = (reward +
                      (self.hps.gamma ** td_len) * (1. - done) *
                      self.denorm_rets(q_prime).detach())
            targ_q = self.norm_rets(targ_q)

            if self.hps.ret_norm:
                if self.hps.popart:
                    # Apply Pop-Art, https://arxiv.org/pdf/1602.07714.pdf
                    # Save the pre-update running stats
                    old_mean = torch.Tensor(self.rms_ret.mean).to(self.device)
                    old_std = torch.Tensor(self.rms_ret.std).to(self.device)
                    # Update the running stats
                    self.rms_ret.update(targ_q)
                    # Get the post-update running statistics
                    new_mean = torch.Tensor(self.rms_ret.mean).to(self.device)
                    new_std = torch.Tensor(self.rms_ret.std).to(self.device)
                    # Preserve the output from before the change of normalization old->new
                    # for both online and target critic(s)
                    outs = [self.crit.out_params, self.targ_crit.out_params]
                    if self.hps.clipped_double:
                        outs.extend([self.twin.out_params, self.targ_twin.out_params])
                    for out in outs:
                        w, b = out
                        w.data.copy_(w.data * old_std / new_std)
                        b.data.copy_(((b.data * old_std) + old_mean - new_mean) / new_std)
                else:
                    # Update the running stats
                    self.rms_ret.update(targ_q)

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
            _actr_loss = -self.crit.QZ(state, self.actr.act(state))

        actr_loss = _actr_loss.mean()

        # Log metrics
        metrics['crit_loss'].append(crit_loss)
        if self.hps.clipped_double:
            metrics['twin_loss'].append(twin_loss)
        if self.hps.prioritized_replay:
            metrics['iws'].append(iws)
        metrics['actr_loss'].append(actr_loss)

        if self.hps.kye_p:

            if self.hps.wrap_absorb:
                _state = torch.Tensor(batch['obs0_orig']).to(self.device)
            else:
                _state = torch.Tensor(batch['obs0']).to(self.device)
            state = torch.Tensor(batch['obs0']).to(self.device)
            if self.hps.n_step_returns:
                next_state = torch.Tensor(batch['obs1_td1']).to(self.device)  # not the n-step next!
            else:
                next_state = torch.Tensor(batch['obs1']).to(self.device)
            action = torch.Tensor(batch['acs']).to(self.device)
            if self.hps.wrap_absorb:
                _, indices = self.remove_absorbing(state)
                _state = _state[indices, :]
                state = state[indices, :]
                next_state = next_state[indices, :]
                action = action[indices, :]

            input_a = state
            if self.hps.state_only:
                input_b = next_state
            else:
                input_b = action

            _aux_loss = F.mse_loss(
                input=self.actr.auxo(_state),
                target=self.disc.D(input_a, input_b),
                reduction='none',
            )

            if self.hps.adaptive_aux_scaling:
                inputs = []
                for n, p in self.actr.fc_stack.named_parameters():
                    if p.requires_grad:
                        if len(p.shape) == 1:  # ignore the bias vectors
                            continue
                        inputs.append(p)
                grads_a_list = []
                grads_b_list = []
                for i in range(_state.size(0)):
                    # Compute the gradients of the shared weights for the main task
                    grads_a = autograd.grad(
                        outputs=[_actr_loss[i, ...]],
                        inputs=[*inputs],
                        only_inputs=True,
                        grad_outputs=[torch.ones_like(_actr_loss[i, ...])],
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )
                    # Compute the gradients of the shared weights for the auxiliary task
                    grads_b = autograd.grad(
                        outputs=[_aux_loss[i, ...]],
                        inputs=[*inputs],
                        only_inputs=True,
                        grad_outputs=[torch.ones_like(_aux_loss[i, ...])],
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )
                    grads_a = torch.cat(list(grads_a), dim=-1)
                    grads_b = torch.cat(list(grads_b), dim=-1)
                    grads_a_list.append(grads_a)
                    grads_b_list.append(grads_b)
                grads_a = torch.stack(grads_a_list, dim=0)
                grads_b = torch.stack(grads_b_list, dim=0)
                cos_sims = F.cosine_similarity(grads_a, grads_b).unsqueeze(-1)
                cos_sims = cos_sims.detach()  # safety measure
                metrics['cos_sim_aux'].append(cos_sims.mean())
                cos_sims = cos_sims.mean(dim=1)
                assert _aux_loss.shape == cos_sims.shape, "shape mismatch"
                _aux_loss *= torch.max(torch.zeros_like(cos_sims), cos_sims)

            _aux_loss *= self.hps.kye_p_scale

            aux_loss = _aux_loss.mean()
            metrics['aux_loss'].append(aux_loss)

            if self.hps.kye_mixing:
                # Add mixing auxiliary loss
                e_batch = next(iter(self.e_dataloader))  # get a minibatch of expert data
                if self.hps.wrap_absorb:
                    _state_e = e_batch['obs0_orig']
                else:
                    _state_e = e_batch['obs0']
                state_e = e_batch['obs0']
                next_state_e = e_batch['obs1']
                action_e = e_batch['acs']
                if self.hps.wrap_absorb:
                    _, indices = self.remove_absorbing(state_e)
                    _state_e = _state_e[indices, :]
                    state_e = state_e[indices, :]
                    next_state_e = next_state_e[indices, :]
                    action_e = action_e[indices, :]

                input_a_e = state_e
                if self.hps.state_only:
                    input_b_e = next_state_e
                else:
                    input_b_e = action_e

                aux_loss += self.hps.kye_p_scale * F.mse_loss(
                    input=self.actr.auxo(_state_e),
                    target=self.disc.D(input_a_e, input_b_e),
                )

            actr_loss += aux_loss

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
        if self.hps.clipped_double:
            self.twin_opt.step()

        if update_actor:

            self.actr_opt.step()

            _lr = self.actr_sched.step(steps_so_far=iters_so_far * self.hps.rollout_len)
            logger.info(f"lr is {_lr} after {iters_so_far} timesteps")

        # Update target nets
        self.update_target_net(iters_so_far)

        if self.hps.reward_type == 'gail_kye_mod':
            self.kye.update(state_a, action_a, next_state_a, self.disc.D)  # ignore returned var

        if self.hps.reward_type == 'gail_dyn_mod':
            self.dyn.update(state_a, action_a, next_state_a)  # ignore returned var

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        lrnows = {'actr': _lr}

        return metrics, lrnows

    def update_discriminator(self, batch):
        """Update the discriminator network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Create DataLoader object to iterate over transitions in rollouts
        d_keys = ['obs0']
        if self.hps.state_only:
            if self.hps.n_step_returns:
                d_keys.append('obs1_td1')
            else:
                d_keys.append('obs1')
        else:
            d_keys.append('acs')

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
            p_input_a = d_batch['obs0'].to(self.device)
            e_input_a = e_batch['obs0'].to(self.device)
            if self.hps.state_only:
                if self.hps.n_step_returns:
                    p_input_b = d_batch['obs1_td1'].to(self.device)
                else:
                    p_input_b = d_batch['obs1'].to(self.device)
                e_input_b = e_batch['obs1'].to(self.device)
            else:
                p_input_b = d_batch['acs'].to(self.device)
                e_input_b = e_batch['acs'].to(self.device)

            # Compute scores
            p_scores = self.disc.D(p_input_a, p_input_b)
            e_scores = self.disc.D(e_input_a, e_input_b)

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
                _p_e_loss = -self.hps.purl_eta * torch.log(1. - torch.sigmoid(e_scores) + 1e-8)
                _p_e_loss += -torch.max(-beta * torch.ones_like(p_scores),
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
                _p_e_loss = p_loss + e_loss  # leave different name
            # Average out over the batch
            p_e_loss = _p_e_loss.mean()

            # Aggregated loss
            d_loss = p_e_loss + entropy_loss

            # Log metrics
            metrics['entropy_loss'].append(entropy_loss)
            metrics['p_e_loss'].append(p_e_loss)
            metrics['disc_loss'].append(d_loss)

            if self.hps.grad_pen:
                # Create gradient penalty loss (coefficient from the original paper)
                grad_pen = self.grad_pen(self.hps.grad_pen_type,
                                         p_input_a, p_input_b, e_input_a, e_input_b)
                grad_pen *= self.hps.grad_pen_scale
                d_loss += grad_pen
                # Log metrics
                metrics['grad_pen'].append(grad_pen)

            # Update parameters
            self.disc_opt.zero_grad()
            d_loss.backward()
            average_gradients(self.disc, self.device)
            self.disc_opt.step()

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics

    def grad_pen(self, variant, p_input_a, p_input_b, e_input_a, e_input_b):
        """Define the gradient penalty regularizer"""
        if variant == 'wgan':
            # Assemble interpolated inputs
            eps_a = torch.rand(p_input_a.size(0), 1)
            eps_b = torch.rand(p_input_b.size(0), 1)
            input_a_i = eps_a * p_input_a + ((1. - eps_a) * e_input_a)
            input_b_i = eps_b * p_input_b + ((1. - eps_b) * e_input_b)
            input_a_i.requires_grad = True
            input_b_i.requires_grad = True
        elif variant == 'hyperwgan':
            # Assemble interpolated inputs
            eps_a = p_input_a.clone().detach().data.uniform_()
            eps_b = p_input_b.clone().detach().data.uniform_()
            input_a_i = eps_a * p_input_a + ((1. - eps_a) * e_input_a)
            input_b_i = eps_b * p_input_b + ((1. - eps_b) * e_input_b)
            input_a_i.requires_grad = True
            input_b_i.requires_grad = True
        elif variant == 'dragan':
            # Assemble interpolated inputs
            eps_a = p_input_a.clone().detach().data.normal_(0, 10)
            eps_b = p_input_b.clone().detach().data.normal_(0, 10)
            input_a_i = e_input_a + eps_a
            input_b_i = e_input_b + eps_b
            input_a_i.requires_grad = True
            input_b_i.requires_grad = True
        elif variant == 'nagard':
            eps_a = p_input_a.clone().detach().data.normal_(0, 10)
            eps_b = p_input_b.clone().detach().data.normal_(0, 10)
            input_a_i = p_input_a + eps_a
            input_b_i = p_input_b + eps_b
            input_a_i.requires_grad = True
            input_b_i.requires_grad = True
        elif variant == 'bare':
            input_a_i = Variable(p_input_a, requires_grad=True)
            input_b_i = Variable(p_input_b, requires_grad=True)
        else:
            raise NotImplementedError("invalid gradient penalty type")
        # Create the operation of interest
        score = self.disc.D(input_a_i, input_b_i)
        # Get the gradient of this operation with respect to its inputs
        grads = autograd.grad(
            outputs=score,
            inputs=[input_a_i, input_b_i],
            only_inputs=True,
            grad_outputs=[torch.ones_like(score)],
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )
        assert len(list(grads)) == 2, "length must be exactly 2"
        # Return the gradient penalty (try to induce 1-Lipschitzness)
        grads = torch.cat(list(grads), dim=-1)
        grads_norm = grads.norm(2, dim=-1)
        if self.hps.one_sided_pen:
            # Penalize the gradient for having a norm GREATER than 1
            _grad_pen = torch.max(torch.zeros_like(grads_norm),
                                  grads_norm - self.hps.grad_pen_targ).pow(2)
        else:
            # Penalize the gradient for having a norm LOWER OR GREATER than 1
            _grad_pen = (grads_norm - self.hps.grad_pen_targ).pow(2)

        if variant == 'bare':
            # Return before averaging over the minibatch
            return _grad_pen

        grad_pen = _grad_pen.mean()
        return grad_pen

    def get_syn_rew(self, state, action, next_state):
        # Define the discriminator inputs
        input_a = state
        if self.hps.state_only:
            input_b = next_state
        else:
            input_b = action
        # Craft surrogate reward
        assert sum([isinstance(x, torch.Tensor) for x in [input_a, input_b]]) in [0, 2]
        if not isinstance(input_a, torch.Tensor):  # then the other is not neither
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)
        # Transfer to cpu
        input_a = input_a.cpu()
        input_b = input_b.cpu()
        # Compure score
        score = self.disc.D(input_a, input_b).detach().view(-1, 1)
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

        if self.hps.reward_type == 'gail':
            return reward

        if self.hps.reward_type == 'gail_grad_mod':
            # Compute GP modulation
            grad_pen = self.grad_pen('bare', input_a, input_b, None, None).detach().view(-1, 1)
            self.grad_pen_store.append(grad_pen.mean())
            if len(self.grad_pen_store) == 10:
                self.rms_grad_pens.update(np.array(self.grad_pen_store))
                self.grad_pen_store = []
            rescaled_grad_pen = self.rms_grad_pens.divide_by_std(grad_pen)
            # return reward * F.softplus(-rescaled_grad_pen)
            return reward * torch.exp(-rescaled_grad_pen)

        if self.hps.reward_type in ['red', 'gail_red_mod']:
            # Compute reward
            red_reward = self.red.get_syn_rew(input_a, input_b)
            # Assemble and return reward
            if self.hps.reward_type == 'red':
                return red_reward
            elif self.hps.reward_type == 'gail_red_mod':
                assert red_reward.shape == reward.shape
                return red_reward * reward

        if self.hps.reward_type == 'gail_red_grad_mod':
            # Compute RED reward
            red_reward = self.red.get_syn_rew(input_a, input_b)
            # Compute GP modulation
            grad_pen = self.grad_pen('bare', input_a, input_b, None, None).detach().view(-1, 1)
            self.grad_pen_store.append(grad_pen.mean())
            if len(self.grad_pen_store) == 10:
                self.rms_grad_pens.update(np.array(self.grad_pen_store))
                self.grad_pen_store = []
            rescaled_grad_pen = self.rms_grad_pens.divide_by_std(grad_pen)
            return reward * red_reward * F.softplus(-rescaled_grad_pen)

        if self.hps.reward_type == 'gail_kye_mod':
            # Know your enemy
            kye_reward = self.kye.get_int_rew(input_a, input_b, self.disc.D)
            return kye_reward * reward

        if self.hps.reward_type == 'gail_dyn_mod':
            # Controlability
            if self.hps.state_only:
                input_c = action
            else:
                input_c = next_state
            if not isinstance(input_c, torch.Tensor):
                input_c = torch.Tensor(input_c)
            input_a = input_a.cpu()
            if self.hps.state_only:
                dyn_reward = self.dyn.get_int_rew(input_a, input_c, input_b)
            else:
                dyn_reward = self.dyn.get_int_rew(input_a, input_b, input_c)
            return dyn_reward * reward

        # If this point is reached, something fatally went wrong
        raise ValueError("invalid reward type")

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

    def save(self, path, iters_so_far):
        torch.save(self.actr.state_dict(), osp.join(path, f"actr_{iters_so_far}.pth"))
        torch.save(self.crit.state_dict(), osp.join(path, f"crit_{iters_so_far}.pth"))
        if self.hps.clipped_double:
            torch.save(self.twin.state_dict(), osp.join(path, f"twin_{iters_so_far}.pth"))
        if not self.eval_mode:
            torch.save(self.disc.state_dict(), osp.join(path, f"disc_{iters_so_far}.pth"))

    def load(self, path, iters_so_far):
        self.actr.load_state_dict(torch.load(osp.join(path, f"actr_{iters_so_far}.pth")))
        self.crit.load_state_dict(torch.load(osp.join(path, f"crit_{iters_so_far}.pth")))
        if self.hps.clipped_double:
            self.twin.load_state_dict(torch.load(osp.join(path, f"twin_{iters_so_far}.pth")))
        if not self.eval_mode:
            self.disc.load_state_dict(torch.load(osp.join(path, f"disc_{iters_so_far}.pth")))
