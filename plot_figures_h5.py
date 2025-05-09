import json
import os.path

import numpy as np
import math
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
import h5py
import pandas as pd

# real example of training for 100 update steps in asus saving all layers
exp_date = "2025-04-10_16-35-53"
grad_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_grads.h5"
loss_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_loss.npy"
model_path = f"logs/c4_llama_60m_none/{exp_date}/model_layers.json"
exp_info_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_grad_injection.json"
thresh = 25#50
x_limit_ini = 0  #300 #0
x_limit_end = 700  #450 #700
max_traj = 100
range_xs_spike_detection = 10
show_figs = True
save_figs = False
load_embed_and_head = False

# exp_date = "2025-04-16_21-42-53_with_gradients_before"
# grad_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_AdamWGradientSaving_grads.h5"
# # grad_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_MuonGradientSaving_grads.h5"
# loss_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_loss.npy"
# model_path = f"logs/c4_llama_60m_none/{exp_date}/model_layers.json"
# thresh = 40
# x_limit_ini = 300 #0
# x_limit_end = 450 #700
# max_traj = 100
# range_xs_spike_detection = 10
# show_figs = True # False
# save_figs = False
# # tests with layers 1-4
# exp_date = "2025-04-24_11-16-10" # factor 1 layers 1-4 p 1 step 5 dur 5
# # exp_date = "2025-04-24_11-19-03" # factor 0.5 layers 1-4 p 1 step 5 dur 5
# # exp_date = "2025-04-24_11-22-18" # factor 0.1 layers 1-4 p 1 step 5 dur 5
#
# exp_date = "2025-04-24_11-43-04" # factor 10 layers 1-4 p 1 step 5 dur 5
# exp_date = "2025-04-24_11-52-12" # factor 10 layers 4 13 22 31 40 49 58 67 p 1 step 5 dur 5
# exp_date = "2025-04-24_14-57-09" #YESSSSS
# # exp_date = "2025-04-24_15-53-14"
# exp_date = "2025-04-24_15-56-50"
#
# exp_date = "2025-04-24_16-05-34"
# # exp_date = "2025-04-24_16-09-08"
# exp_date = "2025-04-24_16-13-38"
# exp_date = "2025-04-24_16-17-10"
#
# #TODO: test new gaussian with sum instead of multi, DEBUG with factor=5 and params=1.0 and duration=5
# # exp_date = "2025-04-23_10-16-36" # gaussian(SUM instead of MULT) at 15 with factor=5 and 1.0 params and d=5
#
#
#
# # Adam and Muon, f=0.01, p=0.05
# exp_date = "2025-04-26_13-07-50"  # best
# exp_date = "2025-04-26_14-06-00"
# # Adam and Muon, f=0.01, p=0.01
# exp_date = "2025-04-26_15-26-54"
# exp_date = "2025-04-26_17-57-39"
# # Adam and Muon, f=0.01, p=0.02
# exp_date = "2025-04-27_20-02-37"
# exp_date = "2025-04-27_14-49-35"
# # Adam and Muon, f=0.01, p=0.03
# exp_date = "2025-04-28_09-53-19"
# exp_date = "2025-04-28_10-59-29"
#
# # Adam no injection
# exp_date = "2025-04-29_11-09-03"
# # # Muon no injection
# exp_date = "2025-04-29_12-17-40"
#
# grad_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_AdamWGradientInjection_grads.h5"
# # grad_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_MuonGradientInjection_grads.h5"
# loss_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_loss.npy"
# model_path = f"logs/c4_llama_60m_none/{exp_date}/model_layers.json"
# exp_info_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_grad_injection.json"
# thresh = 25#50
# x_limit_ini = 0
# x_limit_end = 700#25 #400
# max_traj = 100
# range_xs_spike_detection = 10
# show_figs = True
# save_figs = True
# load_embed_and_head = False


# for gradient norm clipping
# exp_date = "2025-05-06_11-12-32"
# exp_date = "2025-05-06_12-43-48"
#
# exp_date = "2025-05-06_13-24-30_noINJ_noGNS_noAGC"
# exp_date = "2025-05-06_21-22-02_yesINJ0.01_noGNS_noAGC"
# exp_date = "2025-05-06_20-08-27_yesINJ0.05_noGNS_noAGC"
#
# # FOLLOWING 3 ARE WITH GNS AFTER MUON
# exp_date = "2025-05-06_14-36-08_noINJ_yesGNS_noAGC"
# exp_date = "2025-05-06_17-07-44_yesINJ0.01_yesGNS_noAGC"
# # exp_date = "2025-05-06_15-52-02_yesINJ0.05_yesGNS_noAGC"
#
# # FOLLOWING 2 ARE WITH GNS BEFORE MUON
# exp_date = "2025-05-07_18-27-18_noINJ_yesGNSbeforeMUON_noAGC" # YESSSS!!!!!!!!
# exp_date = "2025-05-07_19-45-02_yesINJ0.01_yesGNSbeforeMUON_noAGC"
#
# # grad_path = f"logs/gradnormscaling_c4_llama_60m_none/{exp_date}/gradnormscalingc_4_llama_60m_none_AdamWGradientInjection_grads.h5"
# grad_path = f"logs/gradnormscaling_c4_llama_60m_none/{exp_date}/gradnormscaling_c4_llama_60m_none_MuonGradientInjection_grads.h5"
# loss_path = f"logs/gradnormscaling_c4_llama_60m_none/{exp_date}/gradnormscaling_c4_llama_60m_none_loss.npy"
# model_path = f"logs/gradnormscaling_c4_llama_60m_none/{exp_date}/model_layers.json"
# exp_info_path = f"logs/gradnormscaling_c4_llama_60m_none/{exp_date}/gradnormscaling_c4_llama_60m_none_grad_injection_muon.json"
# thresh = 5#50
# x_limit_ini = 0
# x_limit_end = 700#25 #400
# max_traj = 100
# range_xs_spike_detection = 10
# show_figs = True
# save_figs = True
# load_embed_and_head = False


exp_date = "2025-05-07_11-47-11_noINJ_noGNS_yesAGCafterMUON"
exp_date = "2025-05-07_21-03-31_noINJ_noGNS_yesAGCbeforeMUON"
# exp_date = "2025-05-07_22-33-54_yesINJ0.01_noGNS_yesAGCbeforeMUON"

# grad_path = f"logs/adagradclipping_c4_llama_60m_none/{exp_date}/adagradclipping_4_llama_60m_none_AdamWGradientInjection_grads.h5"
grad_path = f"logs/adagradclipping_c4_llama_60m_none/{exp_date}/adagradclipping_c4_llama_60m_none_MuonGradientInjection_grads.h5"
loss_path = f"logs/adagradclipping_c4_llama_60m_none/{exp_date}/adagradclipping_c4_llama_60m_none_loss.npy"
model_path = f"logs/adagradclipping_c4_llama_60m_none/{exp_date}/model_layers.json"
exp_info_path = f"logs/adagradclipping_c4_llama_60m_none/{exp_date}/adagradclipping_c4_llama_60m_none_grad_injection_muon.json"
thresh = 5#50
x_limit_ini = 0
x_limit_end = 700#25 #400
max_traj = 100
range_xs_spike_detection = 10
show_figs = True
save_figs = True
load_embed_and_head = False


# # agc_gns
# exp_date = "2025-05-08_13-56-28_noINJ_yesAGC_yesGNS"
# exp_date = "2025-05-08_15-21-38_yesINJ0.01_yesAGC_yesGNS"
#
# # grad_path = f"logs/agc_gns_c4_llama_60m_none/{exp_date}/agc_gns_4_llama_60m_none_AdamWGradientInjection_grads.h5"
# grad_path = f"logs/agc_gns_c4_llama_60m_none/{exp_date}/agc_gns_c4_llama_60m_none_MuonGradientInjection_grads.h5"
# loss_path = f"logs/agc_gns_c4_llama_60m_none/{exp_date}/agc_gns_c4_llama_60m_none_loss.npy"
# model_path = f"logs/agc_gns_c4_llama_60m_none/{exp_date}/model_layers.json"
# exp_info_path = f"logs/agc_gns_c4_llama_60m_none/{exp_date}/agc_gns_c4_llama_60m_none_grad_injection_muon.json"
# thresh = 5#50
# x_limit_ini = 0
# x_limit_end = 700#25 #400
# max_traj = 100
# range_xs_spike_detection = 10
# show_figs = True
# save_figs = True
# load_embed_and_head = False
#
# # gc_agc_gns
# exp_date = "2025-05-08_16-41-41_noINJ_yesAGC_yesGNS_yesGC"
# exp_date = "2025-05-08_18-01-42_yesINJ0.01_yesAGC_yesGNS_yesGC"
#
# # grad_path = f"logs/gc_agc_gns_c4_llama_60m_none/{exp_date}/gc_agc_gns_4_llama_60m_none_AdamWGradientInjection_grads.h5"
# grad_path = f"logs/gc_agc_gns_c4_llama_60m_none/{exp_date}/gc_agc_gns_c4_llama_60m_none_MuonGradientInjection_grads.h5"
# loss_path = f"logs/gc_agc_gns_c4_llama_60m_none/{exp_date}/gc_agc_gns_c4_llama_60m_none_loss.npy"
# model_path = f"logs/gc_agc_gns_c4_llama_60m_none/{exp_date}/model_layers.json"
# exp_info_path = f"logs/gc_agc_gns_c4_llama_60m_none/{exp_date}/gc_agc_gns_c4_llama_60m_none_grad_injection_muon.json"
# thresh = 5#50
# x_limit_ini = 0
# x_limit_end = 700#25 #400
# max_traj = 100
# range_xs_spike_detection = 10
# show_figs = True
# save_figs = True
# load_embed_and_head = False

# gc_gns
exp_date = "2025-05-08_21-48-37_noINJ_noAGC_yesGNS_yesGC"
exp_date = "2025-05-08_23-11-35_yesINJ0.01_noAGC_yesGNS_yesGC"

# grad_path = f"logs/gc_gns_c4_llama_60m_none/{exp_date}/gc_gns_4_llama_60m_none_AdamWGradientInjection_grads.h5"
grad_path = f"logs/gc_gns_c4_llama_60m_none/{exp_date}/gc_gns_c4_llama_60m_none_MuonGradientInjection_grads.h5"
loss_path = f"logs/gc_gns_c4_llama_60m_none/{exp_date}/gc_gns_c4_llama_60m_none_loss.npy"
model_path = f"logs/gc_gns_c4_llama_60m_none/{exp_date}/model_layers.json"
exp_info_path = f"logs/gc_gns_c4_llama_60m_none/{exp_date}/gc_gns_c4_llama_60m_none_grad_injection_muon.json"
thresh = 5#50
x_limit_ini = 0
x_limit_end = 700#25 #400
max_traj = 100
range_xs_spike_detection = 10
show_figs = True
save_figs = True
load_embed_and_head = False


# # 130M
# exp_date = "2025-04-29_18-22-37"  # Adam
# exp_date = "2025-04-29_22-19-20"  # Muon
# # exp_date = "2025-04-30_13-18-51"
# # exp_date = "2025-04-30_14-14-57"
#
# # exp_date = "2025-04-30_16-37-37"
# exp_date = "2025-04-30_17-34-08"
#
# grad_path = f"logs/c4_llama_130m_none/{exp_date}/c4_llama_130m_none_AdamWGradientInjection_grads.h5"
# grad_path = f"logs/c4_llama_130m_none/{exp_date}/c4_llama_130m_none_MuonGradientInjection_grads.h5"
# loss_path = f"logs/c4_llama_130m_none/{exp_date}/c4_llama_130m_none_loss.npy"
# model_path = f"logs/c4_llama_130m_none/{exp_date}/model_layers.json"
# exp_info_path = f"logs/c4_llama_130m_none/{exp_date}/c4_llama_130m_none_grad_injection.json"
# thresh = 5 #25#50
# x_limit_ini = 0
# x_limit_end = 350 #400#700#25 #400
# max_traj = 100
# range_xs_spike_detection = 10
# show_figs = True
# save_figs = True
# load_embed_and_head = False

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
        if f"{layer_name}_before" in grad_data_file.keys():
            try:
                grad_data_before = grad_data_file[layer_name + '_before'][x_limit_ini:x_limit_end]
            except KeyError as e:
                print(f"{layer_name}_before not found in {grad_path}!")
                continue

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

            fig = plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
            fig.suptitle(exp_date.replace("_", " ") + " " +
                         model_name +
                         ", optim: " + exp_info["optim_name"] +
                         "\nfn: " + exp_info["fn"] +
                         ", step: " + str(exp_info["step"]) +
                         ", factor: " + str(exp_info["factor"]) +
                         ", p: " + str(exp_info["elements"]) +
                         ", layers: " + str(exp_info["layer_number"]) +
                         f"\nmin(loss): {llama_loss.min():.6f}")
            subfig = fig.add_subplot(3, 1, 1)
            subfig.set_title(f"{layer_idx}: {layer_name}")
            subfig.plot(range(x_limit_ini, x_limit_end), llama_loss)
            subfig.grid(True)
            subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x
            # subfig.set_xticks(range(0, x_limit + 1, 100))  # X-axis grid every 100 units

            num_traj = true_indices_detach[1].shape[0]
            subfig = fig.add_subplot(3, 1, 2)
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

            subfig = fig.add_subplot(3, 1, 3)
            subfig.set_title(f"# of traj. with spike (before NS): {true_indices_detach[1].shape[0]} (showing {max_traj})"
                             f" in {len(steps_with_spike)} steps")


            # print('THIS!!!!')
            # # grad_data_gns = grad_data_file[layer_name + '_gns'][x_limit_ini:x_limit_end]
            # grad_data_gns = grad_data_file[layer_name + '_agc'][x_limit_ini:x_limit_end]
            # subfig.plot(grad_data_gns[:, true_indices_detach[1][:max_traj], true_indices_detach[2][:max_traj]])
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

            fig_path = os.path.join(os.path.dirname(grad_path),
                                    f"fig_{layer_idx}_{layer_name.replace('.', '-')}_grads_before.png")

            if save_figs:
                plt.savefig(fig_path)
            if show_figs:
                plt.show()
            plt.close()
            exit()
        else:

            fig = plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
            fig.suptitle(exp_date.replace("_", " ") + " " +
                         model_name +
                         ", optim: " + exp_info["optim_name"] +
                         "\nfn: " + exp_info["fn"] +
                         ", step: " + str(exp_info["step"]) +
                         ", factor: " + str(exp_info["factor"]) +
                         ", p: " + str(exp_info["elements"]) +
                         ", layers: " + str(exp_info["layer_number"]) +
                         f"\nmin(loss): {llama_loss.min():.6f}")
            subfig = fig.add_subplot(2, 1, 1)
            subfig.set_title(f"{layer_idx}: {layer_name}")
            subfig.plot(range(x_limit_ini, x_limit_end), llama_loss)
            subfig.grid(True)
            subfig.set_xlim(x_limit_ini, x_limit_end)  # Establecer límites en el eje x
            # subfig.set_xticks(range(0, x_limit + 1, 100))  # X-axis grid every 100 units

            # show start and end of spikes
            subfig.text(0.95, 0.95, f"steps: {summarize_numbers(steps_with_spike)}",
                        transform=subfig.transAxes,
                        horizontalalignment='right',
                        verticalalignment='top',
                        fontsize=10)

            # # Add dashed vertical lines for detected spikes
            # spikes_xs_ini = [r[0] + x_limit_ini for r in summarize_numbers(steps_with_spike, as_int=True)]
            # spikes_xs_end = [r[1] + x_limit_ini for r in summarize_numbers(steps_with_spike, as_int=True)]
            # ymin, ymax = plt.ylim()  # Get current y-limits
            # subfig.vlines(x=spikes_xs_ini,  ymin=ymin, ymax=ymax,
            #               colors='green', linestyles='dashed', linewidths=1)
            # subfig.vlines(x=spikes_xs_end, ymin=ymin, ymax=ymax,
            #               colors='red', linestyles='dashed', linewidths=1)


            subfig = fig.add_subplot(2, 1, 2)
            subfig.set_title(f"# of trajectories with spike: {true_indices_detach[1].shape[0]}"
                             f" in {len(steps_with_spike)} steps"
                             f" - max(times): {max_spikes.max():.2f}"
                             f" - max(abs(grad))): {max_abs_grad:.2f}")
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

            # show multiplier of each step
            max_spikes_thresh = np.sort(max_spikes)[-10]
            for i, (xi, yi) in enumerate(zip(range(grad_data_plot.shape[0]), grad_data_plot)):
                if max_spikes[i] > max_spikes_thresh:
                    yi = yi[np.argmax(np.abs(yi))]
                    plt.text(xi + x_limit_ini, yi + 0.05*yi,
                             f'{max_spikes[i]:.2f}', ha='center', va='bottom', rotation=45, fontsize=8)

            # # Add dashed vertical lines for detected spikes
            # ymin, ymax = plt.ylim()  # Get current y-limits
            # subfig.vlines(x=spikes_xs_ini, ymin=ymin, ymax=ymax,
            #               colors='green', linestyles='dashed', linewidths=1)
            # subfig.vlines(x=spikes_xs_end, ymin=ymin, ymax=ymax,
            #               colors='red', linestyles='dashed', linewidths=1)

            fig_path = os.path.join(os.path.dirname(grad_path), f"fig_{layer_idx}_{layer_name.replace('.', '-')}_grads.png")
            if save_figs:
                plt.savefig(fig_path)
            if show_figs:
                plt.show()
            plt.close()
            # continue
            exit()

            # # inspection of gradients close to the spikes detected
            # for spike_x in spikes_xs:
            #     for range_x in range(-range_xs_spike_detection, range_xs_spike_detection):
            #         pos = spike_x + range_x
            #         print(f"[{pos}]\tloss={llama_loss[pos]:.4f}\tgrads_abs_max={np.abs(grad_data_plot[pos]).max():.4f}")
            #
            #     fig = plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
            #     fig.suptitle(exp_date.replace("_", " ") + f" gradient spike {spike_x}")
            #     subfig = fig.add_subplot(2, 1, 1)
            #     subfig.set_title(f"{layer_idx}: {layer_name}")
            #     subfig.plot(llama_loss[spike_x - range_xs_spike_detection: spike_x + range_xs_spike_detection])
            #     subfig.grid(True)
            #     subfig.set_xlim(0, 2 * range_xs_spike_detection - 1)  # Establecer límites en el eje x
            #     subfig.set_xticks(range(0, 2 * range_xs_spike_detection, 1))  # X-axis grid every 100 units
            #     subfig.set_xticklabels(range(spike_x - range_xs_spike_detection, spike_x + range_xs_spike_detection))
            #
            #     # Add dashed vertical lines for detected spikes
            #     ymin, ymax = plt.ylim()  # Get current y-limits
            #     subfig.vlines(x=range_xs_spike_detection, ymin=ymin, ymax=ymax,
            #                   colors='black', linestyles='dashed', linewidths=1)
            #
            #     subfig = fig.add_subplot(2, 1, 2)
            #     subfig.set_title(f"# of trajectories with spike: {true_indices_detach[1].shape[0]}"
            #                      f" in {len(steps_with_spike)} steps")
            #     if grad_data.ndim > 2:
            #         grad_data_trajs = grad_data[:, true_indices_detach[1], true_indices_detach[2]]
            #     else:
            #         grad_data_trajs = grad_data[:, true_indices_detach[1]]
            #     traj_ids = np.argsort(-np.abs(grad_data_trajs).max(axis=(0)))
            #     grad_data_plot = grad_data_trajs[:, traj_ids[:max_traj]]
            #     subfig.plot(grad_data_plot[spike_x - range_xs_spike_detection: spike_x + range_xs_spike_detection])
            #     subfig.grid(True)
            #     subfig.set_xlim(0, 2 * range_xs_spike_detection - 1)  # Establecer límites en el eje x
            #     subfig.set_xticks(range(0, 2 * range_xs_spike_detection, 1))  # X-axis grid every 100 units
            #     subfig.set_xticklabels(range(spike_x - range_xs_spike_detection, spike_x + range_xs_spike_detection))
            #
            #     subfig.text(0.95, 0.95, f"steps: {summarize_numbers(steps_with_spike)}",
            #                 transform=subfig.transAxes,
            #                 horizontalalignment='right',
            #                 verticalalignment='top',
            #                 fontsize=10)
            #
            #     # Add dashed vertical lines for detected spikes
            #     ymin, ymax = plt.ylim()  # Get current y-limits
            #     subfig.vlines(x=range_xs_spike_detection, ymin=ymin, ymax=ymax,
            #                   colors='black', linestyles='dashed', linewidths=1)
            #
            #     fig_path = os.path.join(os.path.dirname(grad_path),
            #                             f"fig_{layer_idx}_{layer_name.replace('.', '-')}_grads_inspection_{spike_x}.png")
            #     plt.savefig(fig_path)
            #     if show_figs:
            #         plt.show()
            #     plt.close()

    # Save the info of spikes to a csv
    spikes_info = pd.concat([spikes_info, pd.DataFrame(spikes_rows)], ignore_index=True)
    spikes_info.to_csv(os.path.join(os.path.dirname(grad_path), 'grad_spikes_report.csv'), index=False)