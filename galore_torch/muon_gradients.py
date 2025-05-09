import os
import torch
import torch.distributed as dist
from torch import Tensor

import sys
import h5py
import numpy as np
from tqdm import tqdm

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


class MuonGradientSaving(torch.optim.Optimizer):
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
                 world_size=1, name=None, log_folder=None, save_every_N_steps=None):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params["params"]: list[Tensor] = [*params["params"]]
        param_groups = []
        for size in {p.numel() for p in params["params"]}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params["params"] if p.numel() == size],
                         names=[n for n, p in zip(params["names"], params["params"]) if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

        self.grad_dict = {}
        self.grad_dict_before = {}
        self.name = name
        self.log_folder = log_folder
        self.save_every_N_steps = save_every_N_steps

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            names: list[str] = group["names"]
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

                    # to save gradients
                    if self.log_folder is not None:
                        p_name = names[base_i + self.rank]
                        if p_name not in self.grad_dict.keys():
                            if state["step"] == 0:
                                optim_name = self.__class__.__name__
                                print(f"[{optim_name}] Save gradients for layer:\t{p_name}\t{g_shape}")

                            self.grad_dict[p_name] = np.zeros((self.save_every_N_steps, *g_shape),
                                                              dtype=np.float16)
                            self.grad_dict_before[p_name] = np.zeros((self.save_every_N_steps, *g_shape),
                                                                     dtype=np.float16)
                        # save gradients before orthogonalization
                        gradient_step = state["step"] % self.save_every_N_steps
                        self.grad_dict_before[p_name][gradient_step] = g.detach().cpu().float().numpy().reshape(g_shape)

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4:  # for the case of conv filters
                        g = g.view(len(g), -1)
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()

                    # to save gradients after orthogonalization
                    if self.log_folder is not None:
                        self.grad_dict[p_name][gradient_step] = g.detach().cpu().float().numpy().reshape(g_shape)

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
                        if "muon" in optim_name.lower():
                            dset_before = f.create_dataset(
                                layer_name + '_before',
                                shape=(0, *layer_shape[-2:]),  # Initial shape
                                maxshape=(None, *layer_shape[-2:]),  # Allow expansion along axis 0
                                dtype='float16',
                                compression="gzip"  # Optional compression
                            )
                    else:
                        dset = f[layer_name]
                        if "muon" in optim_name.lower():
                            dset_before = f[layer_name + '_before']

                    # Resize the dataset to accommodate new data
                    current_size = dset.shape[0]
                    new_size = current_size + layer_shape[0]
                    dset.resize(new_size, axis=0)
                    if "muon" in optim_name.lower():
                        dset_before.resize(new_size, axis=0)

                    # Write new data at the end of the dataset
                    dset[current_size:new_size] = self.grad_dict[layer_name]
                    if "muon" in optim_name.lower():
                        dset_before[current_size:new_size] = self.grad_dict_before[layer_name]

            print("Saved at", gradient_path)
            self.grad_dict = {}
            self.grad_dict_before = {}