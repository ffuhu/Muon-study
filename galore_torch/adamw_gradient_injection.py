# copy dependencies from transformers/optimization.py
import os
import math
import copy
import warnings
from typing import Callable, Iterable, Tuple

import torch
from scipy.optimize import anderson
from torch import nn
from torch.optim import Optimizer
import numpy as np
from transformers.utils.versions import require_version

import sys
import h5py
import json
import torch

from .galore_projector import GaLoreProjector
from .galore_projector_tensor import GaLoreProjectorTensor
import torch.optim as optim
from tqdm import tqdm


class GaussianFunction:
    def __init__(self, step, duration, total_steps):
        """
        Initialize the Gaussian function parameters.

        Args:
            duration (int): The range around the center where values are close to 1 (>0.6).
        """
        self.mu = step
        self.sigma = duration // 2
        self.total_steps = total_steps
        self.steps = torch.arange(self.total_steps + 1, dtype=torch.bfloat16)
        # Gaussian formula: exp(-((x - mean)^2 / (2 * std^2)))
        self.gaussian_values = torch.exp(-((self.steps - self.mu) ** 2) / (2 * self.sigma ** 2))

    def get_value(self, t):
        """
        Get the value of the Gaussian at a specific step t.

        Args:
            t (int): The step at which to compute the Gaussian value.

        Returns:
            float: The value of the Gaussian at step t.
        """
        return self.gaussian_values[t]


# # for debugging
# import matplotlib.pyplot as plt
# gaussian = GaussianFunction(step=200, duration=10, total_steps=10000)
# plt.plot(gaussian.gaussian_values.numpy())
# plt.title("Gaussian Function"), plt.xlabel("Steps"), plt.ylabel("Value"), plt.grid(True)
# plt.show()

class AdamWGradientInjection(Optimizer):
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
            log_folder=None,
            grad_injection_step=None,
            grad_injection_factor=None,
            grad_injection_elements=None,
            grad_injection_layer_number=None,
            grad_injection_fn=None,
            grad_injection_duration=None,
            total_steps=None,
            save_every_N_steps=None,
            grad_save_layers=None,
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

        # for gradient spike detection
        # self.current_step=0
        self.total_step = 0
        self.grad_dict = {}
        self.moment_dict = {}
        self.name = name
        self.moment_second_dict = {}
        self.log_folder = log_folder
        self.save_every_N_steps = save_every_N_steps
        self.grad_save_layers = grad_save_layers

        # for gradient injection
        if grad_injection_step:
            self.grad_injection = {
                # TODO: fix to work with multiple values
                'optim_name': self.__class__.__name__,
                'step': grad_injection_step[0],
                'factor': grad_injection_factor[0],
                'elements': grad_injection_elements[0],
                'layer_number': grad_injection_layer_number,
                'fn': grad_injection_fn,
                'duration': grad_injection_duration[0],
                'total_steps': total_steps,
                'grad_save_layers': grad_save_layers,
            }

            if grad_injection_fn == "gaussian":
                self.gauss_values = GaussianFunction(step=self.grad_injection["step"],
                                                     duration=self.grad_injection["duration"],
                                                     total_steps=total_steps)

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

        # for gradient spike detection
        # self.current_step+=1
        self.total_step += 1
        # layer_n = 0

        for group in self.param_groups:
            for p_id, p_name, p in zip(group["ids"], group["names"], group["params"]):
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

                # for injecting gradient
                condition_injecting_gradients = (  # "embed" not in p_name and "head" not in p_name and
                        self.grad_injection["step"] != -1 and
                        (p_id in self.grad_injection["layer_number"] or
                         -1 in self.grad_injection["layer_number"])
                )
                if condition_injecting_gradients:  # inject

                    if self.grad_injection["fn"] == "gaussian":
                        gauss_mult = self.gauss_values.get_value(state["step"])
                        # if gauss_mult > 1e-2:
                        #     grad_injection_factor = self.grad_injection["factor"] * gauss_mult
                        #     # grad_injection_elements = self.grad_injection['elements']
                        #     grad_injection = torch.zeros_like(grad)
                        #     # grad_idxs_injection_i = (torch.rand(grad_injection_elements) * grad.shape[0]).to(torch.int32)
                        #     # grad_idxs_injection_j = (torch.rand(grad_injection_elements) * grad.shape[1]).to(torch.int32)
                        #     if self.grad_injection["elements"] < 1.0:
                        #         grad_idxs_injection_i = self.grad_injection[layer_n]["i"]
                        #         grad_idxs_injection_j = self.grad_injection[layer_n]["j"]
                        #         grad_injection[grad_idxs_injection_i, grad_idxs_injection_j] = grad_injection_factor
                        #     else:
                        #         grad_injection += grad_injection_factor
                        #
                        #     # inject
                        #     grad = grad + grad_injection
                        #     print(f"Gradient injected!\t"
                        #           f"(Elements: {self.grad_injection['elements']}, ",
                        #           f"Gauss mult: {gauss_mult:.4f}, "
                        #           f"Mult: {self.grad_injection['factor']}, "
                        #           f"Gauss*Mult: {gauss_mult * self.grad_injection['factor']:.4f})")

                    elif self.grad_injection["fn"] == "step":
                        d = self.grad_injection["duration"] // 2
                        if self.grad_injection["step"] - d <= state["step"] <= self.grad_injection["step"] + d:
                            grad_injection = torch.zeros_like(grad)
                            mask = torch.empty_like(grad).uniform_(0, 1) <= self.grad_injection["elements"]
                            random_values = torch.empty(mask.sum(),
                                                        dtype=grad.dtype,
                                                        device=grad.device).fill_(self.grad_injection["factor"])
                            grad_injection[mask] = random_values
                            grad.add_(grad_injection)

                    elif self.grad_injection["fn"] == "sign_step":
                        d = self.grad_injection["duration"] // 2
                        if self.grad_injection["step"] - d <= state["step"] <= self.grad_injection["step"] + d:
                            grad_injection = torch.zeros_like(grad)
                            # if self.grad_injection["elements"] < 1.0:
                            #     grad_idxs_injection_i = self.grad_injection[layer_n]["i"]
                            #     grad_idxs_injection_j = self.grad_injection[layer_n]["j"]
                            #     grad_injection[grad_idxs_injection_i, grad_idxs_injection_j] = self.grad_injection[
                            #         "factor"]
                            # else:
                            #     grad_injection = self.grad_injection["factor"]
                            grad_injection = torch.zeros_like(grad)
                            mask = torch.empty_like(grad).uniform_(0, 1) <= self.grad_injection["elements"]
                            random_values = torch.empty(mask.sum(),
                                                        dtype=grad.dtype,
                                                        device=grad.device).fill_(self.grad_injection["factor"])
                            grad_injection[mask] = random_values
                            grad.add_(grad_injection).mul_(-1.)

                    elif self.grad_injection["fn"] == "sign":
                        d = self.grad_injection["duration"] // 2
                        if self.grad_injection["step"] - d <= state["step"] <= self.grad_injection["step"] + d:
                            grad_injection = torch.ones_like(grad)
                            # if self.grad_injection["elements"] < 1.0:
                            #     grad_idxs_injection_i = self.grad_injection[layer_n]["i"]
                            #     grad_idxs_injection_j = self.grad_injection[layer_n]["j"]
                            #     grad_injection[grad_idxs_injection_i, grad_idxs_injection_j] = -1.
                            # else:
                            #     grad_injection = -1.
                            grad_injection = torch.ones_like(grad)
                            mask = torch.empty_like(grad).uniform_(0, 1) <= self.grad_injection["elements"]
                            grad_injection[mask] = -1.
                            grad.mul_(grad_injection)

                    elif self.grad_injection["fn"] == "sum_random_uniform":
                        d = self.grad_injection["duration"] // 2
                        if self.grad_injection["step"] - d <= state["step"] <= self.grad_injection["step"] + d:
                            # if self.grad_injection["elements"] < 1.0:
                            #     grad_injection = torch.zeros_like(grad)
                            #     grad_idxs_injection_i = self.grad_injection[layer_n]["i"]
                            #     grad_idxs_injection_j = self.grad_injection[layer_n]["j"]
                            #     f = self.grad_injection["factor"]
                            #     grad_injection_values = torch.empty(grad_idxs_injection_i.shape[0]).uniform_(-f, f)
                            #     grad_injection[grad_idxs_injection_i, grad_idxs_injection_j] = grad_injection_values
                            #     grad.add_(grad_injection)
                            # else:
                            #     grad_injection = torch.empty_like(grad).uniform_(-self.grad_injection["factor"],
                            #                                                      self.grad_injection["factor"])
                            grad_injection = torch.zeros_like(grad)
                            mask = torch.empty_like(grad).uniform_(0, 1) <= self.grad_injection["elements"]
                            random_values = torch.empty(mask.sum(),
                                                        dtype=grad.dtype,
                                                        device=grad.device).uniform_(-self.grad_injection["factor"],
                                                                                     self.grad_injection["factor"])
                            grad_injection[mask] = random_values
                            grad.add_(grad_injection)

                # save gradients
                condition_saving_gradients = (p_id in self.grad_injection['grad_save_layers']
                                              or self.grad_injection["grad_save_layers"] == -1)  #"embed" not in p_name and "head" not in p_name
                if condition_saving_gradients:
                    # for gradient spike detection - ALL LAYERS
                    # p_name = group["name"]
                    if p_name not in self.grad_dict.keys():
                        if state["step"] == 0:
                            optim_name = self.__class__.__name__
                            print(f"[{optim_name}] Save gradients for layer:\t{p_name}\t{grad.shape}")
                        # self.grad_dict[layer_n] = []
                        self.grad_dict[p_name] = np.zeros((self.save_every_N_steps, *grad.shape), dtype=np.float16)
                    # self.grad_dict[layer_n].append(grad.detach().cpu().to(dtype=torch.float16))
                    gradient_step = state["step"] % self.save_every_N_steps
                    self.grad_dict[p_name][gradient_step] = grad.detach().cpu().float().numpy()
                # layer_n += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                state["step"] += 1
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

        # for gradient saving
        if state['step'] % self.save_every_N_steps == 0 and 0 < state['step'] <= 1000:

            optim_name = self.__class__.__name__
            gradient_path = os.path.join(self.log_folder, f"{self.name}_{optim_name}_grads.h5")

            # Open or create an HDF5 file
            with h5py.File(gradient_path, 'a') as f:  # 'a' mode allows appending data
                pbar = tqdm(self.grad_dict.keys(), desc='Saving gradients')
                for layer_name in pbar:
                    layer_shape = self.grad_dict[layer_name].shape
                    layer_size = sys.getsizeof(self.grad_dict[layer_name]) / 1024 ** 2
                    pbar.set_description(f"Saving gradients for {layer_name} ({layer_size:.2f} MB)")
                    # Create a dataset to store the gradients of each layer
                    if layer_name not in f:
                        dset = f.create_dataset(
                            layer_name,
                            shape=(0, *layer_shape[1:]),  # Initial shape
                            maxshape=(None, *layer_shape[1:]),  # Allow expansion along axis 0
                            dtype='float16',
                            compression="gzip"  # Optional compression
                        )
                    else:
                        dset = f[layer_name]

                    # Resize the dataset to accommodate new data
                    current_size = dset.shape[0]
                    new_size = current_size + layer_shape[0]
                    dset.resize(new_size, axis=0)

                    # Write new data at the end of the dataset
                    dset[current_size:new_size] = self.grad_dict[layer_name]

            print("Saved at", gradient_path)
            self.grad_dict = {}

            # log grad injection params
            grad_info = copy.deepcopy(self.grad_injection)
            for k, v in grad_info.items():
                if isinstance(v, dict):
                    for k1, v1 in v.items():
                        if isinstance(v1, torch.Tensor):
                            grad_info[k][k1] = grad_info[k][k1].__repr__()
            with open(os.path.join(self.log_folder, self.name + "_grad_injection_adamw.json"), "w") as f:
                f.write(json.dumps(grad_info, indent=4))

        return loss
