import numpy as np
import matplotlib.pyplot as plt

# Parameters
import random
import matplotlib
import matplotlib.pyplot as plt
import torch

# Parameters
T = 200
eta_0 = 10  # Big perturbation value
perturb_prob = 0.1  # Probability of perturbation
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8

# Generate cosine gradient values
steps = np.arange(1, T + 1)
gradients = np.random.rand(T) * 0.1 + 0.1

# Initialize m and v
m = np.zeros(T)
v = np.zeros(T)
m_perturbed = np.zeros(T)
v_perturbed = np.zeros(T)

# Initialize m and v for restart every 30 steps
m_restart = np.zeros(T)
v_restart = np.zeros(T)
m_perturbed_restart = np.zeros(T)
v_perturbed_restart = np.zeros(T)

# Compute m and v with and without random perturbation
for t in range(1, T + 1):
    g = gradients[t - 1]

    # Apply random perturbation with 20% probability
    if t == 30:
        print("t", t)
        g_perturbed = g + eta_0
    else:
        g_perturbed = g

    # Update m and v without perturbation
    m[t - 1] = beta_1 * m[t - 2] + (1 - beta_1) * g if t > 1 else (1 - beta_1) * g
    v[t - 1] = beta_2 * v[t - 2] + (1 - beta_2) * g ** 2 if t > 1 else (1 - beta_2) * g ** 2

    # Update m and v with perturbation
    m_perturbed[t - 1] = beta_1 * m_perturbed[t - 2] + (1 - beta_1) * g_perturbed if t > 1 else (
                                                                                                            1 - beta_1) * g_perturbed
    v_perturbed[t - 1] = beta_2 * v_perturbed[t - 2] + (1 - beta_2) * g_perturbed ** 2 if t > 1 else (
                                                                                                                 1 - beta_2) * g_perturbed ** 2
    # Update m and v with restarts and perturbation
    if t % 30 == 1:
        m_restart[t - 1] = (1 - beta_1) * g
        # m_restart[t-1] = beta_1 * m_restart[t-2] + (1 - beta_1) * g
        v_restart[t - 1] = (1 - beta_2) * g ** 2
        m_perturbed_restart[t - 1] = (1 - beta_1) * g_perturbed
        # m_perturbed_restart[t-1] = beta_1 * m_perturbed_restart[t-2] + (1 - beta_1) * g_perturbed
        v_perturbed_restart[t - 1] = (1 - beta_2) * g_perturbed ** 2
    else:
        m_restart[t - 1] = beta_1 * m_restart[t - 2] + (1 - beta_1) * g
        v_restart[t - 1] = beta_2 * v_restart[t - 2] + (1 - beta_2) * g ** 2
        m_perturbed_restart[t - 1] = beta_1 * m_perturbed_restart[t - 2] + (1 - beta_1) * g_perturbed
        v_perturbed_restart[t - 1] = beta_2 * v_perturbed_restart[t - 2] + (1 - beta_2) * g_perturbed ** 2

# Compute bias-corrected estimates m_hat and v_hat
m_hat = m / (1 - beta_1 ** steps)
v_hat = v / (1 - beta_2 ** steps)
m_perturbed_hat = m_perturbed / (1 - beta_1 ** steps)
v_perturbed_hat = v_perturbed / (1 - beta_2 ** steps)
m_restart_hat = m_restart / (1 - beta_1 ** steps)
v_restart_hat = v_restart / (1 - beta_2 ** steps)
m_perturbed_restart_hat = m_perturbed_restart / (1 - beta_1 ** steps)
v_perturbed_restart_hat = v_perturbed_restart / (1 - beta_2 ** steps)

# Plotting all together
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(steps, m_hat, label='m_hat without perturbation')
plt.plot(steps, m_perturbed_hat, label='m_hat with random perturbation', linestyle='--')
plt.plot(steps, m_restart_hat, label='m_hat with restart every 30 steps')
plt.plot(steps, m_perturbed_restart_hat, label='m_hat with restart and random perturbation', linestyle='--', c='yellow')
plt.xlabel('Steps')
plt.ylabel('m_hat')
plt.legend()
plt.title('m_hat values over steps')
plt.subplot(2, 1, 2)
plt.plot(steps, v_hat, label='v_hat without perturbation')
plt.plot(steps, v_perturbed_hat, label='v_hat with random perturbation', linestyle='--')
plt.plot(steps, v_restart_hat, label='v_hat with restart every 30 steps')
plt.plot(steps, v_perturbed_restart_hat, label='v_hat with restart and random perturbation', linestyle='--', c='yellow')
plt.xlabel('Steps')
plt.ylabel('v_hat')
plt.legend()
plt.title('v_hat values over steps')
plt.tight_layout()
plt.show()
