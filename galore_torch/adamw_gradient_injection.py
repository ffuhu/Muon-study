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
            # AdaGN as in https://github.com/TianjinYellow/StableSPAM/blob/master/galore_torch/stablespam.py
            grad_norm_scaling=False,
            grad_norm_scaling_gammas=None,
            grad_norm_scalin_total_T=None,
            grad_norm_scaling_eta_min=None,
            grad_norm_scaling_scale=None,
            # adaclip
            grad_ada_clipping=False,
            grad_ada_clipping_theta=None,
            grad_centering=False,
            grad_apply_on_adam=False,
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
        self.grad_norm_scaling = grad_norm_scaling
        self.grad_norm_scaling_gammas = grad_norm_scaling_gammas
        self.grad_norm_scalin_total_T = grad_norm_scalin_total_T
        self.grad_norm_scaling_eta_min = grad_norm_scaling_eta_min
        self.grad_norm_scaling_scale = grad_norm_scaling_scale
        self.grad_ada_clipping = grad_ada_clipping
        self.grad_ada_clipping_theta = grad_ada_clipping_theta
        self.eps = 1e-8
        self.grad_centering = grad_centering
        self.grad_apply_on_adam = grad_apply_on_adam

        if self.grad_norm_scaling and self.grad_apply_on_adam:
            self.grad_dict_gns = {}
            self.gamma1 = self.grad_norm_scaling_gammas[0]
            self.gamma2 = self.grad_norm_scaling_gammas[1]

        if self.grad_ada_clipping and self.grad_apply_on_adam:
            self.grad_dict_agc = {}
            self.theta = self.grad_ada_clipping_theta

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
                'grad_norm_scaling': grad_norm_scaling,
                'grad_norm_scaling_gammas': grad_norm_scaling_gammas,
                'grad_norm_scalin_total_T': grad_norm_scalin_total_T,
                'grad_norm_scaling_eta_min': grad_norm_scaling_eta_min,
                'grad_norm_scaling_scale': grad_norm_scaling_scale,
                'grad_ada_clipping': grad_ada_clipping,
                'grad_ada_clipping_theta': grad_ada_clipping_theta,
                'grad_centering': grad_centering,
                'grad_apply_on_adam': grad_apply_on_adam,
            }


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
                g_shape = grad.shape
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

                ############################### grad injection start ###############################
                condition_injecting_gradients = (self.grad_injection["step"] != -1
                                                 and (p_id in self.grad_injection["layer_number"] or
                                                      -1 in self.grad_injection["layer_number"]))
                if condition_injecting_gradients:  # inject

                    if self.grad_injection["fn"] == "step":
                        d = self.grad_injection["duration"] // 2
                        if self.grad_injection["step"] - d <= state["step"] <= self.grad_injection["step"] + d:
                            grad_injection = torch.zeros_like(grad)
                            mask = torch.empty_like(grad).uniform_(0, 1) <= self.grad_injection["elements"]
                            random_values = torch.empty(mask.sum(),
                                                        dtype=grad.dtype,
                                                        device=grad.device).fill_(self.grad_injection["factor"])
                            grad_injection[mask] = random_values
                            grad.add_(grad_injection)
                            print(f"Injecting gradient to {p_name}! param %: {mask.sum() / grad.numel():.2f} "
                                  f"factor: {self.grad_injection['factor']}")
                ############################### grad injection end ###############################

                ############################### grad saving start ###############################
                condition_saving_gradients = (self.log_folder is not None and
                                              (p_id in self.grad_save_layers or
                                               self.grad_save_layers == -1))
                if condition_saving_gradients:
                    if p_name not in self.grad_dict.keys():
                        if state["step"] == 0:
                            optim_name = self.__class__.__name__
                            print(f"[{optim_name}] Save gradients for layer:\t{p_name}\t{g_shape}")

                        self.grad_dict[p_name] = np.zeros((self.save_every_N_steps, *g_shape),
                                                          dtype=np.float16)
                        if self.grad_norm_scaling and self.grad_apply_on_adam:
                            self.grad_dict_gns[p_name] = np.zeros_like(self.grad_dict[p_name])
                        if self.grad_ada_clipping and self.grad_apply_on_adam:
                            self.grad_dict_agc[p_name] = np.zeros_like(self.grad_dict[p_name])
                    # save gradients before orthogonalization
                    gradient_step = state["step"] % self.save_every_N_steps
                    self.grad_dict[p_name][gradient_step] = grad.detach().cpu().float().numpy().reshape(g_shape)
                ############################### grad saving end ###############################

                ############################### grad centering start ###############################
                condition_grad_centering = self.grad_centering and self.grad_apply_on_adam
                if condition_grad_centering:
                    g_dim = tuple(range(1, len(list(grad.size()))))
                    g_mean = grad.mean(dim=g_dim, keepdim=True)
                    grad.add_(-g_mean)
                ############################### grad centering end ###############################

                ############################### grad clipping start ###############################
                # adaptative spike-aware gradient clipping - AdaClip as in Stable SPAM (https://arxiv.org/pdf/2502.17055)
                condition_grad_ada_clipping = self.grad_ada_clipping and self.grad_apply_on_adam
                if condition_grad_ada_clipping:
                    if "m_max_t" not in state:
                        state["m_max_t"] = 0

                    m_max_t = state["m_max_t"]
                    max_gradient = torch.max(grad.abs())
                    m_max_t = self.theta * m_max_t + (1 - self.theta) * max_gradient
                    m_max_hat = m_max_t / (1 - self.theta ** (state["step"] + 1))

                    mask = grad.abs() > m_max_hat
                    if mask.sum() > 0:
                        grad[mask] = grad[mask] / max_gradient * m_max_hat

                    # to save gradients after adaclip
                    if condition_saving_gradients:
                        self.grad_dict_agc[p_name][gradient_step] = grad.detach().cpu().float().numpy().reshape(
                            g_shape)
                ############################### grad clipping end ###############################

                ############################### norm scaling start ###############################
                # adaptative gradient norm scaling - AdaGN as in Stable SPAM (https://arxiv.org/pdf/2502.17055)
                condition_grad_norm_scaling = self.grad_norm_scaling and self.grad_apply_on_adam
                if condition_grad_norm_scaling:
                    scale = 1.
                    # if self.grad_norm_scalin_total_T is not None and self.grad_norm_scaling_scale is not None:
                    #     scale = self.warmup.get_dr(state["step"] + 1)

                    if "m_norm_t" not in state:
                        state["m_norm_t"] = 0
                        state["v_norm_t"] = 0

                    grad_norm = torch.norm(grad)
                    m_norm_t, v_norm_t = state["m_norm_t"], state["v_norm_t"]
                    m_norm_t = self.gamma1 * scale * m_norm_t + (1 - self.gamma1 * scale) * grad_norm
                    v_norm_t = self.gamma2 * v_norm_t + (1 - self.gamma2) * grad_norm ** 2

                    m_norm_hat = m_norm_t / (1 - (self.gamma1 * scale) ** (state['step'] + 1))
                    v_norm_hat = v_norm_t / (1 - self.gamma2 ** (state['step'] + 1))

                    c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat) + self.eps)
                    # print("grad_norm",grad_norm,"c_norm",c_norm_t,"m_norm_t", m_norm_t,"v_norm_t", v_norm_t)

                    if grad_norm > 0:
                        grad = grad / grad_norm * c_norm_t

                    state["m_norm_t"], state["v_norm_t"] = m_norm_t, v_norm_t

                    # to save gradients after gradient norm clipping
                    if condition_saving_gradients:
                        self.grad_dict_gns[p_name][gradient_step] = grad.detach().cpu().float().numpy().reshape(
                            g_shape)
                ############################### norm scaling end ###############################

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
                        # f.create_dataset(layer_name, data=gradient, compression="gzip", chunks=True)
                        dset = f.create_dataset(
                            layer_name,
                            shape=(0, *layer_shape[-2:]),  # Initial shape
                            maxshape=(None, *layer_shape[-2:]),  # Allow expansion along axis 0
                            dtype='float16',
                            compression="gzip"  # Optional compression
                        )
                        if self.grad_norm_scaling and self.grad_apply_on_adam:
                            dset_gns = f.create_dataset(
                                layer_name + '_gns',
                                shape=(0, *layer_shape[-2:]),  # Initial shape
                                maxshape=(None, *layer_shape[-2:]),  # Allow expansion along axis 0
                                dtype='float16',
                                compression="gzip"  # Optional compression
                            )
                        if self.grad_ada_clipping and self.grad_apply_on_adam:
                            dset_agc = f.create_dataset(
                                layer_name + '_agc',
                                shape=(0, *layer_shape[-2:]),  # Initial shape
                                maxshape=(None, *layer_shape[-2:]),  # Allow expansion along axis 0
                                dtype='float16',
                                compression="gzip"  # Optional compression
                            )
                    else:
                        dset = f[layer_name]
                        if self.grad_norm_scaling and self.grad_apply_on_adam:
                            dset_gns = f[layer_name + '_gns']
                        if self.grad_ada_clipping and self.grad_apply_on_adam:
                            dset_agc = f[layer_name + '_agc']

                    # Resize the dataset to accommodate new data
                    current_size = dset.shape[0]
                    new_size = current_size + layer_shape[0]
                    dset.resize(new_size, axis=0)

                    # Write new data at the end of the dataset
                    dset[current_size:new_size] = self.grad_dict[layer_name]
                    if self.grad_norm_scaling and self.grad_apply_on_adam:
                        dset_gns.resize(new_size, axis=0)
                        dset_gns[current_size:new_size] = self.grad_dict_gns[layer_name]
                    if self.grad_ada_clipping and self.grad_apply_on_adam:
                        dset_agc.resize(new_size, axis=0)
                        dset_agc[current_size:new_size] = self.grad_dict_agc[layer_name]

            print("Saved at", gradient_path)
            self.grad_dict = {}
            self.grad_dict_gns = {}
            self.grad_dict_agc = {}

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
