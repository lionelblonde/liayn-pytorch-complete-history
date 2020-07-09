import math
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from helpers import logger
from helpers.console_util import log_module_info
from agents.nets import init
from helpers.distributed_util import RunMoms


class PredNet(nn.Module):

    def __init__(self, env, hps):
        super(PredNet, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        if hps.wrap_absorb:
            ob_dim += 1
            ac_dim += 1
        self.hps = hps
        self.leak = 0.1
        if True:  # FIXME
            # Define observation whitening
            self.rms_obs = RunMoms(shape=env.observation_space.shape, use_mpi=False)
        # Assemble the layers
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(100, 1)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))

    def forward(self, obs, acs):
        if True:  # FIXME
            # Apply normalization
            if self.hps.wrap_absorb:
                obs_ = obs.clone()[:, 0:-1]
                obs_ = torch.clamp(self.rms_obs.standardize(obs_), -5., 5.)
                obs = torch.cat([obs_, obs[:, -1].unsqueeze(-1)], dim=-1)
            else:
                obs = torch.clamp(self.rms_obs.standardize(obs), -5., 5.)
        else:
            obs = torch.clamp(obs, -5., 5.)
        x = torch.cat([obs, acs], axis=-1)
        x = self.fc_stack(x)
        return x


class Smooth(object):

    def __init__(self, env, device, hps):
        self.env = env
        self.device = device
        self.hps = hps

        # Create nets
        self.pred_net = PredNet(self.env, self.hps).to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=3e-4)

        log_module_info(logger, 'Smooth Pred Network', self.pred_net)

    def grad_pen(self, input_a, input_b):
        """Define the gradient penalty regularizer"""
        eps_a = input_a.clone().detach().data.normal_(0, 10)
        eps_b = input_b.clone().detach().data.normal_(0, 10)
        input_a_i = input_a + eps_a
        input_b_i = input_b + eps_b
        input_a_i.requires_grad = True
        input_b_i.requires_grad = True
        # Create the operation of interest
        reward = self.pred_net(input_a_i, input_b_i)
        # Get the gradient of this operation with respect to its inputs
        grads = autograd.grad(
            outputs=reward,
            inputs=[input_a_i, input_b_i],
            only_inputs=True,
            grad_outputs=[torch.ones_like(reward)],
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )
        assert len(list(grads)) == 2, "length must be exactly 2"
        # Return the gradient penalty (try to induce 1-Lipschitzness)
        grads = torch.cat(list(grads), dim=-1)
        grads_norm = grads.norm(2, dim=-1)
        # Penalize the gradient for having a norm LOWER OR GREATER than 1
        _grad_pen = (grads_norm - 1.).pow(2)
        grad_pen = _grad_pen.mean()
        return grad_pen

    def update(self, batch):
        """Update the RND predictor network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Transfer to device
        state = torch.Tensor(batch['obs0']).to(self.device)
        action = torch.Tensor(batch['acs']).to(self.device)
        reward = torch.Tensor(batch['rews']).to(self.device)

        if True:  # FIXME
            if 'obs_orig' in batch:  # gail
                _state = torch.Tensor(batch['obs0_orig']).to(self.device)
            else:  # ppo
                _state = torch.Tensor(batch['obs0']).to(self.device)
            # Update running moments for observations
            self.pred_net.rms_obs.update(_state)

        # Compute loss
        _loss = F.mse_loss(
            self.pred_net(state, action),
            reward,
            reduction='none',
        )
        loss = _loss.mean(dim=-1)
        mask = loss.clone().detach().data.uniform_().to(self.device)
        mask = (mask < 0.9).float()
        loss = (mask * loss).sum() / torch.max(torch.Tensor([1.]), mask.sum())
        # Add gradient penalty
        loss += 10 * self.grad_pen(state, action)

        metrics['loss'].append(loss)

        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics

    def get_rew(self, state, action):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state)
        if not isinstance(action, torch.Tensor):
            action = torch.Tensor(action)
        # Transfer to cpu
        state = state.cpu()
        action = action.cpu()
        # Compute reward
        reward = self.pred_net(state, action).detach()
        return reward
