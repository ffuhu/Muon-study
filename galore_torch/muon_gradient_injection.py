import os
import torch
import torch.distributed as dist
from torch import Tensor

import json
import copy
import sys
import h5py
import numpy as np
from tqdm import tqdm

# for gradient norm scaling - AdaGN (https://arxiv.org/pdf/2502.17055)
class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.5, last_epoch=-1):
        self.sgd = torch.optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max+1, eta_min, last_epoch)
        self.T_max=T_max
        self.eta_min=eta_min
    def step(self,current_step):
        self.cosine_stepper.step(current_step)

    def get_dr(self,current_step):
        self.step(current_step)
        return self.sgd.param_groups[0]['lr']


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonGradientInjection(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, rank=0,
                 world_size=1, name=None, log_folder=None, save_every_N_steps=None,
                 grad_injection_step=None,
                 grad_injection_factor=None,
                 grad_injection_elements=None,
                 grad_injection_layer_number=None,
                 grad_injection_fn=None,
                 grad_injection_duration=None,
                 total_steps=None,
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
                 ):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params["params"]: list[Tensor] = [*params["params"]]
        param_groups = []
        for size in {p.numel() for p in params["params"]}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params["params"] if p.numel() == size],
                         names=[n for n, p in zip(params["names"], params["params"]) if p.numel() == size],
                         ids=[i for i, p in zip(params["ids"], params["params"]) if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

        self.grad_dict = {}
        self.grad_dict_before = {}
        self.name = name
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

        if self.grad_norm_scaling:
            self.grad_dict_gns = {}
            self.gamma1 = self.grad_norm_scaling_gammas[0]
            self.gamma2 = self.grad_norm_scaling_gammas[1]
            # total_T is the total number of update steps
            if self.grad_norm_scaling_scale is not None:
                self.warmup = CosineDecay(1.0, self.grad_norm_scalin_total_T,
                                          eta_min=self.grad_norm_scaling_eta_min)

        if self.grad_ada_clipping:
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
                'muon_ns_steps': ns_steps,
                'grad_save_layers': grad_save_layers,
                'grad_norm_scaling': grad_norm_scaling,
                'grad_norm_scaling_gammas': grad_norm_scaling_gammas,
                'grad_norm_scalin_total_T': grad_norm_scalin_total_T,
                'grad_norm_scaling_eta_min': grad_norm_scaling_eta_min,
                'grad_norm_scaling_scale': grad_norm_scaling_scale,
                'grad_ada_clipping': grad_ada_clipping,
                'grad_ada_clipping_theta': grad_ada_clipping_theta,
                'grad_centering': grad_centering,
            }

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            names: list[str] = group["names"]
            ids: list[int] = group["ids"]
            handle = None
            params_world = None

            def update_prev():  # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5)

            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    g_shape = g.shape
                    assert g is not None
                    state = self.state[p]

                    if "step" not in state:
                        state["step"] = 0

                    p_id = ids[base_i + self.rank]
                    p_name = names[base_i + self.rank]

                    ############################### grad injection start ###############################
                    condition_injecting_gradients = (self.grad_injection["step"] != -1
                                                     and (p_id in self.grad_injection["layer_number"] or
                                                          -1 in self.grad_injection["layer_number"]))
                    if condition_injecting_gradients:  # inject

                        if self.grad_injection["fn"] == "step":
                            d = self.grad_injection["duration"] // 2
                            if self.grad_injection["step"] - d <= state["step"] <= self.grad_injection["step"] + d:
                                grad_injection = torch.zeros_like(g)
                                mask = torch.empty_like(g).uniform_(0, 1) <= self.grad_injection["elements"]
                                random_values = torch.empty(mask.sum(),
                                                            dtype=g.dtype,
                                                            device=g.device).fill_(self.grad_injection["factor"])
                                grad_injection[mask] = random_values
                                g.add_(grad_injection)
                                print(f"Injecting gradient to {p_name}! param %: {mask.sum()/g.numel():.2f} "
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
                            self.grad_dict_before[p_name] = np.zeros_like(self.grad_dict[p_name])
                            if self.grad_norm_scaling:
                                self.grad_dict_gns[p_name] = np.zeros_like(self.grad_dict[p_name])
                            if self.grad_ada_clipping:
                                self.grad_dict_agc[p_name] = np.zeros_like(self.grad_dict[p_name])
                        # save gradients before orthogonalization
                        gradient_step = state["step"] % self.save_every_N_steps
                        self.grad_dict_before[p_name][gradient_step] = g.detach().cpu().float().numpy().reshape(g_shape)
                    ############################### grad saving end ###############################

                    ############################### grad centering start ###############################
                    condition_grad_centering = self.grad_centering
                    if condition_grad_centering:
                        g_dim = tuple(range(1, len(list(g.size()))))
                        g_mean = g.mean(dim=g_dim, keepdim=True)
                        g.add_(-g_mean)
                    ############################### grad centering end ###############################

                    ############################### grad clipping start ###############################
                    # adaptative spike-aware gradient clipping - AdaClip as in Stable SPAM (https://arxiv.org/pdf/2502.17055)
                    condition_grad_ada_clipping = self.grad_ada_clipping
                    if condition_grad_ada_clipping:
                        if "m_max_t" not in state:
                            state["m_max_t"] = 0

                        m_max_t = state["m_max_t"]
                        max_gradient = torch.max(g.abs())
                        m_max_t = self.theta * m_max_t + (1 - self.theta) * max_gradient
                        m_max_hat = m_max_t / (1 - self.theta ** (state["step"] + 1))

                        mask = g.abs() > m_max_hat
                        if mask.sum() > 0:
                            g[mask] = g[mask] / max_gradient * m_max_hat

                        # to save gradients after adaclip
                        if condition_saving_gradients:
                            self.grad_dict_agc[p_name][gradient_step] = g.detach().cpu().float().numpy().reshape(
                                g_shape)
                    ############################### grad clipping end ###############################

                    ############################### norm scaling start ###############################
                    # adaptative gradient norm scaling - AdaGN as in Stable SPAM (https://arxiv.org/pdf/2502.17055)
                    condition_grad_norm_scaling = self.grad_norm_scaling
                    if condition_grad_norm_scaling:
                        scale = 1.
                        if self.grad_norm_scalin_total_T is not None and self.grad_norm_scaling_scale is not None:
                            scale = self.warmup.get_dr(state["step"] + 1)

                        if "m_norm_t" not in state:
                            state["m_norm_t"] = 0
                            state["v_norm_t"] = 0

                        grad_norm = torch.norm(g)
                        m_norm_t, v_norm_t = state["m_norm_t"], state["v_norm_t"]
                        m_norm_t = self.gamma1 * scale * m_norm_t + (1 - self.gamma1 * scale) * grad_norm
                        v_norm_t = self.gamma2 * v_norm_t + (1 - self.gamma2) * grad_norm ** 2

                        m_norm_hat = m_norm_t / (1 - (self.gamma1 * scale) ** (state['step'] + 1))
                        v_norm_hat = v_norm_t / (1 - self.gamma2 ** (state['step'] + 1))

                        c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat) + self.eps)
                        # print("grad_norm",grad_norm,"c_norm",c_norm_t,"m_norm_t", m_norm_t,"v_norm_t", v_norm_t)

                        if grad_norm > 0:
                            g = g / grad_norm * c_norm_t

                        state["m_norm_t"], state["v_norm_t"] = m_norm_t, v_norm_t

                        # to save gradients after gradient norm clipping
                        if condition_saving_gradients:
                            self.grad_dict_gns[p_name][gradient_step] = g.detach().cpu().float().numpy().reshape(
                                g_shape)
                    ############################### norm scaling end ###############################

                    ############################### MUON start ###############################
                    # orthogonalization - NS
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4:  # for the case of conv filters
                        g = g.view(len(g), -1)
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()

                    # to save gradients after orthogonalization
                    if condition_saving_gradients:
                        self.grad_dict[p_name][gradient_step] = g.detach().cpu().float().numpy().reshape(g_shape)

                    ############################### MUON end ###############################

                    state["step"] += 1
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()  # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i: base_i + self.world_size]
            update_prev()

        # for gradient saving
        if state['step'] % self.save_every_N_steps == 0 and 0 < state['step'] <= 1000:

            optim_name = self.__class__.__name__
            gradient_path = os.path.join(self.log_folder, f"{self.name}_{optim_name}_grads.h5")

            # Open or create an HDF5 file
            with h5py.File(gradient_path, 'a') as f:  # 'a' mode allows appending data
                pbar = tqdm(self.grad_dict.keys(), desc='Saving gradients')
                for layer_name in pbar:
                    layer_shape = self.grad_dict[layer_name].shape
                    layer_size = sys.getsizeof(self.grad_dict[layer_name]) / 1024**2
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
                        dset_before = f.create_dataset(
                            layer_name + '_before',
                            shape=(0, *layer_shape[-2:]),  # Initial shape
                            maxshape=(None, *layer_shape[-2:]),  # Allow expansion along axis 0
                            dtype='float16',
                            compression="gzip"  # Optional compression
                        )
                        if self.grad_norm_scaling:
                            dset_gns = f.create_dataset(
                                layer_name + '_gns',
                                shape=(0, *layer_shape[-2:]),  # Initial shape
                                maxshape=(None, *layer_shape[-2:]),  # Allow expansion along axis 0
                                dtype='float16',
                                compression="gzip"  # Optional compression
                            )
                        if self.grad_ada_clipping:
                            dset_agc = f.create_dataset(
                                layer_name + '_agc',
                                shape=(0, *layer_shape[-2:]),  # Initial shape
                                maxshape=(None, *layer_shape[-2:]),  # Allow expansion along axis 0
                                dtype='float16',
                                compression="gzip"  # Optional compression
                            )
                    else:
                        dset = f[layer_name]
                        if "muon" in optim_name.lower():
                            dset_before = f[layer_name + '_before']
                        if self.grad_norm_scaling:
                            dset_gns = f[layer_name + '_gns']
                        if self.grad_ada_clipping:
                            dset_agc = f[layer_name + '_agc']

                    # Resize the dataset to accommodate new data
                    current_size = dset.shape[0]
                    new_size = current_size + layer_shape[0]
                    dset.resize(new_size, axis=0)

                    # Write new data at the end of the dataset
                    dset[current_size:new_size] = self.grad_dict[layer_name]
                    if "muon" in optim_name.lower():
                        dset_before.resize(new_size, axis=0)
                        dset_before[current_size:new_size] = self.grad_dict_before[layer_name]
                    if self.grad_norm_scaling:
                        dset_gns.resize(new_size, axis=0)
                        dset_gns[current_size:new_size] = self.grad_dict_gns[layer_name]
                    if self.grad_ada_clipping:
                        dset_agc.resize(new_size, axis=0)
                        dset_agc[current_size:new_size] = self.grad_dict_agc[layer_name]

            print("Saved at", gradient_path)
            self.grad_dict = {}
            self.grad_dict_before = {}
            self.grad_dict_gns = {}
            self.grad_dict_agc = {}

            # log grad injection params
            grad_info = copy.deepcopy(self.grad_injection)
            for k, v in grad_info.items():
                if isinstance(v, dict):
                    for k1, v1 in v.items():
                        if isinstance(v1, torch.Tensor):
                            grad_info[k][k1] = grad_info[k][k1].__repr__()
            with open(os.path.join(self.log_folder, self.name + "_grad_injection_muon.json"), "w") as f:
                f.write(json.dumps(grad_info, indent=4))