# Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch implementation of the Lion optimizer."""
import torch
from torch.optim.optimizer import Optimizer
import math

class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, **kwargs):
        """Initialize the hyperparameters.
        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

class Adafactor(Optimizer):
    """Implements Adafactor algorithm.

    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient
            (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(self, params, lr=None, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, beta1=None, weight_decay=0.0, scale_parameter=True,
                 relative_step=True, warmup_init=False):
        if lr is not None and relative_step:
            raise ValueError('Cannot combine manual lr and relative_step options')
        if warmup_init and not relative_step:
            raise ValueError('warmup_init requires relative_step=True')

        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold, decay_rate=decay_rate,
                        beta1=beta1, weight_decay=weight_decay, scale_parameter=scale_parameter,
                        relative_step=relative_step, warmup_init=warmup_init)
        super(Adafactor, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group['lr']
        if param_group['relative_step']:
            min_step = 1e-6 * param_state['step'] if param_group['warmup_init'] else 1e-2
            rel_step_sz = min(min_step, 1.0/math.sqrt(param_state['step']))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps'][1], param_state['RMS'])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)
        ).rsqrt_()
        c_factor = exp_avg_sq_col.rsqrt()
        return torch.mm(r_factor.unsqueeze(-1), c_factor.unsqueeze(0))

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError('Adafactor does not support sparse gradients.')

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).to(grad)
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0
                else:
                    if use_first_moment:
                        state['exp_avg'] = state['exp_avg'].to(grad)
                    if factored:
                        state['exp_avg_sq_row'] = state['exp_avg_sq_row'].to(grad)
                        state['exp_avg_sq_col'] = state['exp_avg_sq_col'].to(grad)
                    else:
                        state['exp_avg_sq'] = state['exp_avg_sq'].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)
                group['lr'] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = (grad**2) + group['eps'][0]
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-1))
                    exp_avg_sq_col.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-2))

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2t).add_(1.0 - beta2t, update)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group['clip_threshold']).clamp_(min=1.0)
                )
                update.mul_(group['lr'])

                if use_first_moment:
                    exp_avg = state['exp_avg']                  
                    exp_avg.mul_(group['beta1']).add_(1 - group['beta1'], update)
                    update = exp_avg

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss