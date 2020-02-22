from collections import namedtuple
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
from helpers.math_util import huber_quant_reg_loss
from helpers.distributed_util import average_gradients, sync_with_root
from agents.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer
from agents.nets import Actor, Critic, Discriminator
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
        assert sum([self.hps.binned_aux_loss, self.hps.squared_aux_loss]) in [0, 1], "not both"

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
            qr_cum_density = torch.FloatTensor(qr_cum_density).to(self.device)
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

        if not self.eval_mode:
            # Set up demonstrations dataset
            self.e_dataloader = DataLoader(self.expert_dataset, self.hps.batch_size, shuffle=True)
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

    def setup_replay_buffer(self):
        """Setup experiental memory unit"""
        logger.info(">>>> setting up replay buffer")
        # Create the metadata
        self.shapes = {"obs0": self.ob_shape,
                       "acs": self.ac_shape,
                       "rews": (1,),
                       "dones1": (1,),
                       "obs1": self.ob_shape}
        # Create the buffer
        if self.hps.prioritized_replay:
            if self.hps.unreal:  # Unreal prioritized experience replay
                self.replay_buffer = UnrealReplayBuffer(
                    self.hps.mem_size,
                    self.shapes,
                )
            else:  # Vanilla prioritized experience replay
                self.replay_buffer = PrioritizedReplayBuffer(
                    self.hps.mem_size,
                    self.shapes,
                    alpha=self.hps.alpha,
                    beta=self.hps.beta,
                    ranked=self.hps.ranked,
                )
        else:  # Vanilla experience replay
            self.replay_buffer = ReplayBuffer(
                self.hps.mem_size,
                self.shapes,
            )
        # Summarize replay buffer creation (relies on `__repr__` method)
        logger.info("[INFO] {} configured".format(self.replay_buffer))

    def predict(self, ob, apply_noise):
        """Predict an action, with or without perturbation,
        and optionaly compute and return the associated QZ value.
        """
        # Create tensor from the state (`require_grad=False` by default)
        ob = torch.FloatTensor(ob[None]).to(self.device)
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

    def train(self, update_critic, update_actor, iters_so_far):
        """Train the agent"""

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

        # Create tensors from the inputs
        state = torch.FloatTensor(batch['obs0']).to(self.device)
        action = torch.FloatTensor(batch['acs']).to(self.device)
        reward = torch.FloatTensor(batch['rews']).to(self.device)
        next_state = torch.FloatTensor(batch['obs1']).to(self.device)
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

        # Create data loader (iterable over 1 elt, but using a dataloader is cleaner)
        p_dataset = Dataset(batch)
        p_dataloader = DataLoader(p_dataset, self.hps.batch_size, shuffle=True)

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

        # ######################################################################

        # Self-supervised auxiliary loss
        if self.hps.binned_aux_loss or self.hps.squared_aux_loss:
            ep_state = torch.cat([next(iter(self.e_dataloader))['obs0'], state])
            ep_action = torch.cat([next(iter(self.e_dataloader))['acs'], action])
            ep_next_state = torch.cat([next(iter(self.e_dataloader))['obs1'], next_state])
        if self.hps.binned_aux_loss:
            ss_aux_loss = F.cross_entropy(
                input=self.actr.ss_aux_loss(ep_state),
                target=self.get_reward(ep_state, ep_action, ep_next_state)[1]
            )
        elif self.hps.squared_aux_loss:
            ss_aux_loss = F.mse_loss(
                input=self.actr.ss_aux_loss(ep_state),
                target=self.get_reward(ep_state, ep_action, ep_next_state)[0]
            )
        if self.hps.binned_aux_loss or self.hps.squared_aux_loss:
            ss_aux_loss *= self.hps.ss_aux_loss_scale
            # Add to actor loss
            actr_loss += ss_aux_loss

        # Compute gradients
        self.actr_opt.zero_grad()
        actr_loss.backward()
        average_gradients(self.actr, self.device)
        actr_gradn = U.clip_grad_norm_(self.actr.parameters(), self.hps.clip_norm)
        self.crit_opt.zero_grad()
        crit_loss.backward()
        average_gradients(self.crit, self.device)
        if self.hps.clipped_double:
            self.twin_opt.zero_grad()
            twin_loss.backward()
            average_gradients(self.twin, self.device)

        # Perform model updates
        if update_critic:
            # Update critic
            self.crit_opt.step()
            self.crit_sched.step(iters_so_far)
            if self.hps.clipped_double:
                # Update twin critic
                self.twin_opt.step()
                self.twin_sched.step(iters_so_far)
            if update_actor:
                # Update actor
                self.actr_opt.step()
                self.actr_sched.step(iters_so_far)
                # Update target nets
                self.update_target_net(iters_so_far)
        # Update discriminator
        for _ in range(self.hps.d_update_ratio):
            for p_chunk, e_chunk in zip(p_dataloader, self.e_dataloader):
                disc_loss = self.update_disc(p_chunk, e_chunk)

        # Aggregate the elements to return
        losses = {'actr': actr_loss.clone().cpu().data.numpy(),
                  'crit': crit_loss.clone().cpu().data.numpy(),
                  'disc': disc_loss.clone().cpu().data.numpy()}
        gradns = {'actr': actr_gradn}
        if self.hps.clipped_double:
            losses.update({'twin': twin_loss.clone().cpu().data.numpy()})
        if self.hps.prioritized_replay:
            losses.update({'iws': iws.clone().cpu().data.numpy()})

        lrnows = {'actr': self.actr_sched.get_last_lr(),
                  'crit': self.crit_sched.get_last_lr()}
        if self.hps.clipped_double:
            lrnows.update({'twin': self.twin_sched.get_last_lr()})

        return losses, gradns, lrnows

    def update_disc(self, p_chunk, e_chunk):
        """Update the discriminator network"""
        # Create tensors from the inputs
        p_state = torch.FloatTensor(p_chunk['obs0']).to(self.device)
        p_action = torch.FloatTensor(p_chunk['acs']).to(self.device)
        e_state = torch.FloatTensor(e_chunk['obs0']).to(self.device)
        e_action = torch.FloatTensor(e_chunk['acs']).to(self.device)

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
            p_e_loss = -self.hps.purl_eta * torch.log(1. - torch.sigmoid(e_scores) + 1e-12)
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

        if self.hps.grad_pen:
            # Create gradient penalty loss (coefficient from the original paper)
            grad_pen_in = [p_state, p_action, e_state, e_action]
            grad_pen = 10. * self.grad_pen(*grad_pen_in)
            d_loss += grad_pen

        # Update parameters
        self.disc_opt.zero_grad()
        d_loss.backward()
        average_gradients(self.disc, self.device)
        self.disc_opt.step()
        return p_e_loss

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
            ob = torch.FloatTensor(ob)
            ac = torch.FloatTensor(ac)
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
        minimax_reward = -torch.log(1. - torch.sigmoid(score) + 1e-12)
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
        binned = (sigscore // ((1 / num_bins) + 1e-8)).long().squeeze(-1)
        # Note: the 1e-12 is here to avoid the edge case and keep the bins in {0, 1, 2}
        return reward, binned

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
