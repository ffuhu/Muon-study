import numpy as np
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt

# grad_data = np.load("./c4_llama_nonellama_60mgrad_dict1.npz")
# grad_dict = {key: grad_data[key] for key in grad_data}
# llama_loss = np.load("c4_llama_nonellama_60m_loss.npy")[:1000]

# real example of training for 500 update steps in asus saving only layer 20
exp_date = 'good loss bump training asus'
grad_data = np.load("_good_bumps_example/c4_llama_60m_none_grad_dict.npz")
grad_dict = {key: grad_data[key] for key in grad_data}
llama_loss = np.load("_good_bumps_example/c4_llama_60m_none_loss.npy")

# real example of training for 100 update steps in asus saving all layers
exp_date = '2025-04-10_12-17-17'
exp_date = '2025-04-18_20-52-58'
exp_date = '2025-04-22_09-45-53'
grad_data = np.load(f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_grad_dict.npz")
# grad_dict = {key: grad_data[key] for key in grad_data}
llama_loss = np.load(f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_loss.npy")

# exp_date = '2025-04-10_10-24-03'
# exp_date = '2025-04-10_10-58-00'
# exp_date = '2025-04-10_11-08-49'
# exp_date = '2025-04-10_11-17-16'
# grad_data = np.load(f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_grad_dict.npz")
# grad_dict = {key: grad_data[key] for key in grad_data}
# llama_loss = np.load(f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_loss.npy")
thresh = 30

num_spikes = []  # Used to hold the number of elements greater than 50 in each layer
for key in grad_data.keys():
    # if key == 0:
    #     continue
    print(f"layers at {key}")
    # Get the gradient data for each layer
    data = grad_data[key]
    # Calculate the absolute value and mean
    grad_abs = np.abs(data)
    grad_mean = np.mean(grad_abs, axis=0)
    # Calculate the times value
    times = grad_abs / (grad_mean + 1e-6)
    print(f"max(times): {np.nanmax(times)}")
    mask = times > thresh  # Set the threshold
    # Get the number of masks that are True
    count_above_thresh = np.sum(mask)
    print(f"Number of elements > {thresh} in layer {key}: {count_above_thresh}")
    # Save the number of elements > thresh in each layer
    num_spikes.append(count_above_thresh)

# Print the number of elements > 50 in each layer print("Number of elements > 50 for each layer:")
print(f"Number of elements > {thresh} for each layer:")
for i, count in enumerate(num_spikes):
    print(f"Layer {i}: {count}")
# grad_detach = grad_dict['20']

grad_detach = grad_data['20']
# grad_detach = grad_data['0']
grad_abs_detach = np.abs(grad_detach)
grad_mean_detach = np.mean(grad_abs_detach, axis=0)
times_detach = grad_abs_detach / (grad_mean_detach + 1e-6)
mask_detach = times_detach > thresh
print(mask_detach.sum())
true_indices_detach = np.where(mask_detach == True)
# a_detach = true_indices_detach[1][0]
# b_detach = true_indices_detach[2][0]

# study spikes
steps_with_spike = np.unique(true_indices_detach[0])
for step in steps_with_spike:
    steps_idxs = true_indices_detach[0] == step
    values_step_i = true_indices_detach[1][steps_idxs]
    values_step_j = true_indices_detach[2][steps_idxs]
    print(f"i values for step: {step}: {len(values_step_i)}")
    print(f"j values for step: {step}: {len(values_step_j)}")

# fig = figure(num=None, figsize=(22, 5), dpi=120, facecolor='w', edgecolor='k')
# vgg_all = fig.add_subplot(1, 1, 1)
# vgg_all.plot(grad_detach[:, true_indices_detach[1][:], true_indices_detach[2][:]])

# fig = plt.figure(num=None, figsize=(22, 5), dpi=120, facecolor='w', edgecolor='k')
# plt.plot(grad_detach[:, true_indices_detach[1][:], true_indices_detach[2][:]])
# plt.show()

max_x = min(llama_loss.shape[0], grad_detach[:, true_indices_detach[1][:]].shape[0])

fig = plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
fig.suptitle(exp_date.replace("_", " "))
subfig = fig.add_subplot(2, 1, 1)
subfig.plot(llama_loss)
subfig.grid(True)
subfig.set_xlim(0, max_x)  # Establecer límites en el eje x
subfig.set_xticks(range(0, max_x + 1, 100))  # X-axis grid every 100 units


subfig = fig.add_subplot(2, 1, 2)
subfig.plot(grad_detach[:, true_indices_detach[1][:], true_indices_detach[2][:]])
subfig.grid(True)
subfig.set_xlim(0, max_x)  # Establecer límites en el eje x
# Increase grid size by setting custom ticks
subfig.set_xticks(range(0, max_x + 1, 100))  # X-axis grid every 100 units
plt.show()

# for i in range(0, true_indices_detach[1].shape[0], 20):
#     fig = plt.figure(num=None, figsize=(22, 5), dpi=120, facecolor='w', edgecolor='k')
#     plt.plot(grad_detach[:, true_indices_detach[1][i], true_indices_detach[2][i]])
#     plt.show()

print('Done')


for thresh in [10, 20, 30, 40, 50]:

    num_spikes = []  # Used to hold the number of elements greater than thresh in each layer
    for key in grad_data.keys():
        print(f"layers at {key}")
        # Get the gradient data for each layer
        data = grad_data[key]
        # Calculate the absolute value and mean
        grad_abs = np.abs(data)
        grad_mean = np.mean(grad_abs, axis=0)
        # Calculate the times value
        times = grad_abs / grad_mean
        mask = times > thresh  # Set the threshold
        # Get the number of masks that are True
        count_above_thresh = np.sum(mask)
        print(f"Number of elements > {thresh} in layer {key}: {count_above_thresh}")
        # Save the number of elements > thresh in each layer
        num_spikes.append(count_above_thresh)

        true_indices_detach = np.where(mask == True)
        print(np.unique(true_indices_detach[0]))
