import torch
import json
import os.path

import numpy as np
import math
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
import h5py
import pandas as pd


# for gradient norm clipping
# exp_date = "2025-05-06_11-12-32"
# exp_date = "2025-05-06_12-43-48"

exp_date = "2025-05-06_13-24-30_noINJ_noGNS_noGC"
exp_date = "2025-05-06_14-36-08_noINJ_yesGNS_noGC"
# exp_date = "2025-05-06_15-52-02_yesINJ0.05_yesGNS_noGC"
exp_date = "2025-05-06_17-07-44_yesINJ0.01_yesGNS_noGC"

# exp_date = "2025-05-06_20-08-27_yesINJ0.05_noGNS_noGC"
# exp_date = "2025-05-06_21-22-02_yesINJ0.01_noGNS_noGC"

# grad_path = f"logs/gradnormscaling_c4_llama_60m_none/{exp_date}/gradnormscalingc_4_llama_60m_none_AdamWGradientInjection_grads.h5"
grad_path = f"logs/gradnormscaling_c4_llama_60m_none/{exp_date}/gradnormscaling_c4_llama_60m_none_MuonGradientInjection_grads.h5"
loss_path = f"logs/gradnormscaling_c4_llama_60m_none/{exp_date}/gradnormscaling_c4_llama_60m_none_loss.npy"
model_path = f"logs/gradnormscaling_c4_llama_60m_none/{exp_date}/model_layers.json"
exp_info_path = f"logs/gradnormscaling_c4_llama_60m_none/{exp_date}/gradnormscaling_c4_llama_60m_none_grad_injection_muon.json"
thresh = 5#50
x_limit_ini = 0
x_limit_end = 700#25 #400
max_traj = 100
range_xs_spike_detection = 10
show_figs = True
save_figs = False
load_embed_and_head = False


# function used to summarize trajectories
def summarize_numbers(numbers, as_int=False):
    """Summarize consecutive numbers into ranges (e.g., [1,2,3,4] -> '1-4')."""
    if not len(numbers):
        return ""

    # Sort and remove duplicates
    sorted_numbers = sorted(set(numbers))

    ranges = []
    start = sorted_numbers[0]
    prev = start

    for num in sorted_numbers[1:]:
        if num == prev + 1:
            prev = num
        else:
            ranges.append((start, prev))
            start = prev = num
    ranges.append((start, prev))  # Add the last range

    if as_int:
        return ranges

    # Format ranges
    result = []
    for start, end in ranges:
        if start == end:
            result.append(str(start))
        else:
            result.append(f"{start}-{end}")

    return ", ".join(result)

num_spikes = []  # Used to hold the number of elements greater than thresh in each layer

# load model layers
model_layers = json.loads(open(model_path).read())
column_names = ["layer #", "layer_name", "layer_shape", "# params", "threshold", "max spike", "# spikes", "steps"]
spikes_info = pd.DataFrame(columns=column_names)
spikes_rows = []

# load exp info
exp_info = json.loads(open(exp_info_path).read())

model_name = model_path.split(os.sep)[1].replace("_", "-")
exp_info_desc = (f"EXPERIMENT INFO:\n\n"
                 f"model:\t{model_name}\n"
                 f"date:\t{exp_date}\n"
                 f"duration:\t{exp_info['duration']}\n"
                 f"elements:\t{exp_info['elements']}\n"
                 f"factor:\t{exp_info['factor']}\n"
                 f"fn:\t{exp_info['fn']}\n"
                 f"layer_number:\t{exp_info['layer_number']}\n"
                 f"step:\t{exp_info['step']}\n"
                 f"total_steps:\t{exp_info['total_steps']}\n"
                 )
print(exp_info_desc)

with h5py.File(grad_path, "r") as grad_data_file:
    for layer_info in model_layers.items(): # grad_data_file.keys():
        layer_idx = int(layer_info[0])
        layer_name = layer_info[1]["name"]  # + "_before"
        layer_shape = layer_info[1]["shape"]
        layer_nparams = math.prod(layer_shape)

        # layer_name = 'model.layers.2.self_attn.k_proj.weight'
        if not load_embed_and_head:
            if 'embed' in layer_name or 'head' in layer_name:
                continue

        # Get the gradient data for each layer
        try:
            grad_data = grad_data_file[layer_name][x_limit_ini:x_limit_end]
        except KeyError as e:
            print(f"{layer_name} not found in {grad_path}!")
            continue

        # get the loss
        llama_loss = np.load(loss_path)[x_limit_ini:x_limit_end]

        # Print layer info
        print(f"Layer grad: {layer_idx}\t{layer_name}", end='\t')
        print(f"Shape: {layer_shape}", end='\t')
        print(f"# params: {layer_nparams}", end='\t')

        # Calculate the absolute value and mean
        grad_abs = np.abs(grad_data)
        max_abs_grad = np.max(grad_abs)
        grad_mean = np.mean(grad_abs, axis=0)
        # Calculate the times value
        times = grad_abs / (grad_mean + 1e-6)
        max_spikes = np.nanmax(times, axis=(1, 2) if grad_data.ndim > 2 else 1)
        print(f"max(times): {max_spikes.max()}", end='\t')
        # Keep only above the threshold
        mask = times > thresh
        # Get the number of masks that are True
        count_above_thresh = np.sum(mask)
        print(f"# of elements > {thresh}: {count_above_thresh}", end='\t')
        # Save the number of elements > thresh in each layer
        num_spikes.append(count_above_thresh)

        # study spikes
        true_indices_detach = np.where(mask == True)
        steps_with_spike = np.unique(true_indices_detach[0])

        # Save information in a csv
        column_values = [layer_idx, layer_name, layer_shape, layer_nparams, thresh, max_spikes.max(), count_above_thresh,
                         steps_with_spike]
        spikes_row = dict(zip(column_names, column_values))
        spikes_rows.append(spikes_row)

        if len(steps_with_spike) == 0 and f"{layer_name}_before" not in grad_data_file.keys():
            # print(f"{layer_name} has no spikes (thresh={thresh}).")
            print()
            continue
        print(f"steps: {summarize_numbers(steps_with_spike)}")

        for step in steps_with_spike:
            steps_idxs = true_indices_detach[0] == step
            values_step_i = true_indices_detach[1][steps_idxs]
            if grad_data.ndim > 2:
                values_step_j = true_indices_detach[2][steps_idxs]
                assert len(values_step_i) == len(values_step_j), "Error: values_i don't match values_j"
            # print(f"i values for step: {step}: {len(values_step_i)}")
            # print(f"j values for step: {step}: {len(values_step_j)}")

        # if gradients_before are saved, plot them as well
        grad_data_before = grad_data_file[layer_name + '_before'][x_limit_ini:x_limit_end]

        # if gradients don't have spikes, compute gradients_before
        if len(steps_with_spike) == 0:
            # Calculate the absolute value and mean
            grad_abs = np.abs(grad_data_before)
            grad_mean = np.mean(grad_abs, axis=0)
            # Calculate the times value
            times = grad_abs / (grad_mean + 1e-6)
            max_spikes = np.nanmax(times, axis=(1,2) if grad_data.ndim > 2 else 1)
            print(f"[gradients_before] max(times): {max_spikes.max()}", end='\t')
            # Keep only above the threshold
            mask = times > thresh
            # Get the number of masks that are True
            count_above_thresh = np.sum(mask)
            print(f"# of elements > {thresh}: {count_above_thresh}", end='\t')

            # study spikes for _before
            true_indices_detach = np.where(mask == True)
            steps_with_spike = np.unique(true_indices_detach[0])

            if len(steps_with_spike) == 0:
                # print(f"{layer_name} has no spikes (thresh={thresh}).")
                print()
                continue
            print(f"steps: {summarize_numbers(steps_with_spike)}")

            for step in steps_with_spike:
                steps_idxs = true_indices_detach[0] == step
                values_step_i = true_indices_detach[1][steps_idxs]
                if grad_data.ndim > 2:
                    values_step_j = true_indices_detach[2][steps_idxs]
                    assert len(values_step_i) == len(values_step_j), "Error: values_i don't match values_j"
                    # print(f"i values for step: {step}: {len(values_step_i)}")
                    # print(f"j values for step: {step}: {len(values_step_j)}")

        fig = plt.figure(num=None, figsize=(10, 15), dpi=120, facecolor='w', edgecolor='k')
        fig.suptitle(exp_date.replace("_", " ") + " " +
                     model_name +
                     ", optim: " + exp_info["optim_name"] +
                     "\nfn: " + exp_info["fn"] +
                     ", step: " + str(exp_info["step"]) +
                     ", factor: " + str(exp_info["factor"]) +
                     ", p: " + str(exp_info["elements"]) +
                     ", layers: " + str(exp_info["layer_number"]) +
                     f"\nmin(loss): {llama_loss.min():.6f}")
        subfig = fig.add_subplot(4, 1, 1)
        subfig.set_title(f"{layer_idx}: {layer_name}")
        subfig.plot(range(x_limit_ini, x_limit_end), llama_loss)
        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x
        # subfig.set_xticks(range(0, x_limit + 1, 100))  # X-axis grid every 100 units

        num_traj = true_indices_detach[1].shape[0]
        subfig = fig.add_subplot(4, 1, 2)
        subfig.set_title(f"# of trajectories with spike in gradients after NS = 0")
        if grad_data.ndim > 2:
            grad_data_trajs = grad_data[:, true_indices_detach[1], true_indices_detach[2]]
        else:
            grad_data_trajs = grad_data[:, true_indices_detach[1]]
        traj_ids = np.argsort(-np.abs(grad_data_trajs).max(axis=(0)))
        grad_data_plot = grad_data_trajs[:, traj_ids[:max_traj]]
        subfig.plot(range(x_limit_ini, x_limit_end), grad_data_plot)
        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x
        # Increase grid size by setting custom ticks
        # subfig.set_xticks(range(0, x_limit + 1, 100))  # X-axis grid every 100 units

        subfig = fig.add_subplot(4, 1, 3)
        subfig.set_title(f"# of traj. with spike (before NS): {true_indices_detach[1].shape[0]} (showing {max_traj})"
                         f" in {len(steps_with_spike)} steps")
        subfig.plot(grad_data_before[:, true_indices_detach[1][:max_traj], true_indices_detach[2][:max_traj]])
        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x
        # Increase grid size by setting custom ticks
        # subfig.set_xticks(range(0, x_limit + 1, 100))  # X-axis grid every 100 units

        subfig.text(0.95, 0.95, f"steps: {summarize_numbers(steps_with_spike)}",
                    transform=subfig.transAxes,
                    horizontalalignment='right',
                    verticalalignment='top',
                    fontsize=10)


        subfig = fig.add_subplot(4, 1, 4)
        subfig.set_title(f"# grad gns")
        grad_data_gns = grad_data_file[layer_name + '_gns'][x_limit_ini:x_limit_end]
        subfig.plot(grad_data_gns[:, true_indices_detach[1][:max_traj], true_indices_detach[2][:max_traj]])
        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x

        fig_path = os.path.join(os.path.dirname(grad_path),
                                f"fig_{layer_idx}_{layer_name.replace('.', '-')}_grads.png")
        if save_figs:
            plt.savefig(fig_path)
        if show_figs:
            plt.show()
        plt.close()


        grad_data = torch.tensor(grad_data)
        # grad_data = torch.tensor(grad_data_before)
        grad_data_gns_recomputed = torch.zeros_like(grad_data)
        gns_values = {}
        state = {}
        scale = 1.
        gamma1 = 0.85
        gamma2 = 0.85 #0.99999
        theta = 0.999
        eps = 1e-8
        grad_steps = grad_data.shape[0]
        for i_grad in range(grad_steps):
            # simulate gns values
            if "m_norm_t" not in state:
                state["m_norm_t"] = 0
                state["v_norm_t"] = 0
                gns_values["m_norm_t"] = np.zeros(grad_steps)
                gns_values["v_norm_t"] = np.zeros(grad_steps)
                gns_values["m_norm_hat"] = np.zeros(grad_steps)
                gns_values["v_norm_hat"] = np.zeros(grad_steps)
                gns_values["c_norm_t"] = np.zeros(grad_steps)
                gns_values["ratio"] = np.zeros(grad_steps)
                gns_values["grad_avg"] = np.zeros(grad_steps)
                gns_values["grad_std"] = np.zeros(grad_steps)

            g = grad_data[i_grad]

            grad_norm = torch.norm(g)
            # grad_norm = g.mean()
            # print("\n\n\nTHIS!!!!!!!!\n\n\n")
            m_norm_t, v_norm_t = state["m_norm_t"], state["v_norm_t"]
            m_norm_t = gamma1 * scale * m_norm_t + (1 - gamma1 * scale) * grad_norm
            v_norm_t = gamma2 * v_norm_t + (1 - gamma2) * grad_norm ** 2

            m_norm_hat = m_norm_t / (1 - (gamma1 * scale) ** (i_grad + 1))
            v_norm_hat = v_norm_t / (1 - gamma2 ** (i_grad + 1))

            c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat) + eps)

            gns_values["m_norm_t"][i_grad] = m_norm_t.numpy()
            gns_values["v_norm_t"][i_grad] = v_norm_t.numpy()
            gns_values["m_norm_hat"][i_grad] = m_norm_hat.numpy()
            gns_values["v_norm_hat"][i_grad] = v_norm_hat.numpy()
            gns_values["c_norm_t"][i_grad] = c_norm_t.numpy()
            gns_values["ratio"][i_grad] = grad_norm.numpy()**2 / v_norm_hat
            gns_values["grad_avg"][i_grad] = g.mean()
            gns_values["grad_std"][i_grad] = g.std()


            if grad_norm > 0:
                grad_data_gns_recomputed[i_grad] = g / grad_norm * c_norm_t

            state["m_norm_t"], state["v_norm_t"] = m_norm_t, v_norm_t

            # # centralized gradient: https://arxiv.org/pdf/2004.01461
            # # x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
            # # grad_data[i_grad] -= g.mean(dim=tuple(range(1, len(list(g.size())))), keepdim=True)
            # dim = tuple(range(1, len(list(g.size()))))
            # grad_data_gns_recomputed[i_grad] = g - g.mean() #(dim=0, keepdim=True)

        fig = plt.figure(num=None, figsize=(10, 16), dpi=120, facecolor='w', edgecolor='k')
        fig.suptitle(exp_date.replace("_", " ") + " " +
                     model_name +
                     ", optim: " + exp_info["optim_name"] +
                     "\nfn: " + exp_info["fn"] +
                     ", step: " + str(exp_info["step"]) +
                     ", factor: " + str(exp_info["factor"]) +
                     ", p: " + str(exp_info["elements"]) +
                     ", layers: " + str(exp_info["layer_number"]) +
                     f"\nmin(loss): {llama_loss.min():.6f}"
                     f"\ngradient norm scaling values")

        subfig = fig.add_subplot(8, 1, 1)
        subfig.set_title(f"{layer_idx}: {layer_name}")
        subfig.plot(range(x_limit_ini, x_limit_end), llama_loss)
        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x

        subfig = fig.add_subplot(8, 1, 2)
        subfig.set_title(f"# of trajectories with spike in gradients after NS = 0")
        if grad_data.ndim > 2:
            grad_data_trajs = grad_data[:, true_indices_detach[1], true_indices_detach[2]].numpy()
        else:
            grad_data_trajs = grad_data[:, true_indices_detach[1]].numpy()
        traj_ids = np.argsort(-np.abs(grad_data_trajs).max(axis=(0)))
        grad_data_plot = grad_data_trajs[:, traj_ids[:max_traj]]
        subfig.plot(range(x_limit_ini, x_limit_end), grad_data_plot)
        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x


        subfig = fig.add_subplot(8, 1, 3)
        subfig.set_title("m_norm_t")
        subfig.plot(range(x_limit_ini, x_limit_end), gns_values["m_norm_t"])
        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x

        subfig = fig.add_subplot(8, 1, 4)
        subfig.set_title("v_norm_t")
        subfig.plot(range(x_limit_ini, x_limit_end), gns_values["v_norm_t"])
        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x

        subfig = fig.add_subplot(8, 1, 5)
        subfig.set_title("m_norm_hat")
        subfig.plot(range(x_limit_ini, x_limit_end), gns_values["m_norm_hat"])
        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x

        subfig = fig.add_subplot(8, 1, 6)
        subfig.set_title("v_norm_hat")
        subfig.plot(range(x_limit_ini, x_limit_end), gns_values["v_norm_hat"])
        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x

        subfig = fig.add_subplot(8, 1, 7)
        subfig.set_title("c_norm_t")
        subfig.plot(range(x_limit_ini, x_limit_end), gns_values["c_norm_t"])
        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x

        # subfig = fig.add_subplot(8, 1, 8)
        # subfig.set_title("ratio grad**2/v")
        # subfig.plot(range(x_limit_ini, x_limit_end), gns_values["ratio"])
        # subfig.grid(True)
        # subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x

        subfig = fig.add_subplot(8, 1, 8)
        subfig.set_title("grad avg")
        subfig.plot(range(x_limit_ini, x_limit_end), gns_values["grad_avg"])

        # plt.fill_between(range(x_limit_ini, x_limit_end),
        #                  gns_values["grad_avg"] - gns_values["grad_std"], #y - std,
        #                  gns_values["grad_avg"] + gns_values["grad_std"], #y + std,
        #                  color='b', alpha=0.2, label='Std')

        subfig.grid(True)
        subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x

        plt.subplots_adjust(hspace=0.4)

        fig_path = os.path.join(os.path.dirname(grad_path),
                                f"fig_{layer_idx}_{layer_name.replace('.', '-')}_grads_gns.png")

        if save_figs:
            plt.savefig(fig_path)
        if show_figs:
            plt.show()
        plt.close()
        exit()
