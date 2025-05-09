# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple
import torch
from torch import nn
from torch.optim import Optimizer
import numpy as np
from transformers.utils.versions import require_version
from .galore_projector import GaLoreProjector
from .galore_projector_tensor import GaLoreProjectorTensor
import torch.optim as optim


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).
    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            no_deprecation_warning: bool = False,
            gap=100,
            name=None,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        self.gap = gap
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        # self.current_step=0
        self.total_step = 0
        self.grad_dict = {}
        self.moment_dict = {}
        self.name = name
        self.moment_second_dict = {}

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        # self.current_step+=1
        self.total_step += 1
        n = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                if 'dim' not in group:
                    group['dim'] = 2
                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                # self.trajectories_id.append(grad[50,50].item())
                # self.trajectories_mean.append()
                # #self.momentum_id.append(exp_avg[50,50].item())
                # self.momentum_mean.append(exp_avg.detach().cpu().unsqueeze(0))
                state["step"] += 1
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                if 'rank' in group:
                    if n == 20:
                        if n not in self.grad_dict.keys():
                            print("save n", n)
                            self.grad_dict[n] = []
                            self.moment_dict[n] = []
                            self.moment_second_dict[n] = []
                        self.grad_dict[n].append(grad.detach().cpu())
                    # self.moment_dict[n].append(exp_avg.detach().cpu())
                    # self.moment_second_dict[n].append(exp_avg_sq.detach().cpu())
                    n += 1
                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                # compute norm gradient
                norm_grad = exp_avg / denom
                # GaLore Projection Back
                # if "rank" in group:
                #     norm_grad = state["projector"].project_back(norm_grad)
                # if "rank" in group:
                #     mask=torch.rand_like(norm_grad).to(norm_grad.device)>group["rank"]
                #     norm_grad[mask]=0
                p.add_(norm_grad, alpha=-step_size)
                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        if state['step'] % 1000 == 0 and state['step'] < 1005:
            # np.save("./grad_id.npy",self.trajectories_id)
            grad_dict = {str(key): torch.stack(value).float().numpy() for key, value in self.grad_dict.items()}
            # moment_dict = {str(key): torch.stack(value).float().numpy() for key, value in self.moment_dict.items()}
            # moment_second_dict = {str(key): torch.stack(value).float().numpy() for key, value in self.moment_second_dict.items()}
            print("saving at", "/scratch-shared/HTJ1/" + self.name + "_grad_dict1.npz")
            np.savez_compressed("/scratch-shared/HTJ1/" + self.name + "_grad_dict1.npz", **grad_dict)
        return loss
