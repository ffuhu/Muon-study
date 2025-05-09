import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create lists to store the data
data = []

# Define parameters
spike_factors = [0, 1e-5, 0.01, 0.1, 1, 10, 100]
spike_durations = [1, 2, 5, 10]
grad_magnitude = 1e-9
n_iters = 50

beta1 = 0.9
beta2 = 0.999
eps = 1e-8

# Simulate the calculations
for spike_factor in spike_factors:
    for spike_duration in spike_durations:
        grad = np.random.uniform(-grad_magnitude, grad_magnitude, (4, 4))
        exp_avg = np.zeros_like(grad)
        exp_avg_sq = np.zeros_like(grad)

        for i in range(n_iters):
            if i < spike_duration:
                grad = grad + spike_factor
            else:
                grad = np.random.uniform(-0.001, 0.001, (4, 4))

            # Update exponential averages
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)
            denom = np.sqrt(exp_avg_sq) + eps
            norm_grad = exp_avg / denom

            # Store the results
            data.append({
                'Spike Factor': spike_factor,
                'Spike Duration': spike_duration,
                'Iteration': i,
                'Grad (mean)': grad.mean(),
                'Exp Avg (mean)': exp_avg.mean(),
                'Exp Avg Sq (mean)': exp_avg_sq.mean(),
                'Denom (mean)': denom.mean(),
                'Norm Grad (mean)': norm_grad.mean()
            })

# Create DataFrame
df = pd.DataFrame(data)

# Format the DataFrame
pd.set_option('display.float_format', lambda x: '%.6f' % x)

# Save to both CSV and Excel
df.to_csv('gradient_analysis.csv', index=False)
# df.to_excel('gradient_analysis.xlsx', index=False)

# Display first few rows
print("\nFirst few rows of the data:")
print(df.head(10))

# Calculate some summary statistics
print("\nSummary statistics for different spike factors:")
summary = df.groupby('Spike Factor')[['Norm Grad (mean)']].agg(['mean', 'min', 'max'])
print(summary)

# Create a figure with larger size
plt.figure(figsize=(20, 12))

# Create subplots for each spike duration
for i, duration in enumerate([1, 2, 5], 1):
    plt.subplot(3, 1, i)

    # Filter data for this spike duration
    data = df[df['Spike Duration'] == duration]

    # Plot lines for each spike factor
    for factor in sorted(df['Spike Factor'].unique()):  # Sort to ensure 0 comes first
        factor_data = data[data['Spike Factor'] == factor]
        plt.plot(factor_data['Iteration'], factor_data['Norm Grad (mean)'],
                 label=f'Spike Factor={factor}')

    plt.title(f'Spike Duration = {duration}')
    plt.xlabel('Iteration')
    plt.ylabel('Norm Grad (mean)')
    plt.grid(True, alpha=0.3)
    if i == 1:  # Only show legend for the first subplot to avoid redundancy
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Create a heatmap
plt.figure(figsize=(15, 8))
pivot_data = df.pivot_table(
    values='Norm Grad (mean)',
    index='Spike Duration',
    columns='Spike Factor',
    aggfunc='mean'
)
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis')
plt.title('Average Norm Grad for Different Spike Factors and Durations')
plt.show()

# Create a focused plot for spike factor 1e-5
plt.figure(figsize=(12, 6))
zero_data = df[df['Spike Factor'] == 1e-5]
for duration in spike_durations:
    duration_data = zero_data[zero_data['Spike Duration'] == duration]
    plt.plot(duration_data['Iteration'], duration_data['Norm Grad (mean)'],
             label=f'Duration={duration}')

plt.title('Detailed View of Norm Grad (mean) for Spike Factor = 1e-5')
plt.xlabel('Iteration')
plt.ylabel('Norm Grad (mean)')
plt.grid(True)
plt.legend()
plt.show()

# Create a focused plot for spike factor 0.01
plt.figure(figsize=(12, 6))
zero_data = df[df['Spike Factor'] == 0.01]
for duration in spike_durations:
    duration_data = zero_data[zero_data['Spike Duration'] == duration]
    plt.plot(duration_data['Iteration'], duration_data['Norm Grad (mean)'],
             label=f'Duration={duration}')

plt.title('Detailed View of Norm Grad (mean) for Spike Factor = 0.01')
plt.xlabel('Iteration')
plt.ylabel('Norm Grad (mean)')
plt.grid(True)
plt.legend()
plt.show()

# Create a focused plot for spike factor 0.1
plt.figure(figsize=(12, 6))
zero_data = df[df['Spike Factor'] == 0.1]
for duration in spike_durations:
    duration_data = zero_data[zero_data['Spike Duration'] == duration]
    plt.plot(duration_data['Iteration'], duration_data['Norm Grad (mean)'],
             label=f'Duration={duration}')

plt.title('Detailed View of Norm Grad (mean) for Spike Factor = 0.1')
plt.xlabel('Iteration')
plt.ylabel('Norm Grad (mean)')
plt.grid(True)
plt.legend()
plt.show()

# Create a focused plot for spike factor 1
plt.figure(figsize=(12, 6))
zero_data = df[df['Spike Factor'] == 1]
for duration in spike_durations:
    duration_data = zero_data[zero_data['Spike Duration'] == duration]
    plt.plot(duration_data['Iteration'], duration_data['Norm Grad (mean)'],
             label=f'Duration={duration}')

plt.title('Detailed View of Norm Grad (mean) for Spike Factor = 1')
plt.xlabel('Iteration')
plt.ylabel('Norm Grad (mean)')
plt.grid(True)
plt.legend()
plt.show()