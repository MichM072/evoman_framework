import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, shapiro

# Set the base directory, number of individuals, and test runs
base_dir = ""
best_individuals = 10
test_runs = 5

# Folders and filenames for the experiments
folders = ["test_run_enemy2", "test_run_enemy4", "test_run_enemy8"]
file_names = ["results_test.txt", "results_test_sa.txt"]


# Function to extract individual gains from a file
def extract_individual_gains(file_path):
    individual_gains = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r"Individual Gain = ([\d\.\-e]+)", line)
            if match:
                individual_gains.append(float(match.group(1)))
    return individual_gains


# Calculate the mean individual gain
def calculate_individual_gain_mean(data):
    return [sum(ind) / len(ind) for ind in data]


# Store the extracted data
data = {}

# Loop through the folders and extract gains for both EA1 and EA2
for folder in folders:
    ea1_best_gains, ea2_best_gains = [], []

    for i in range(1, best_individuals + 1):
        best_folder = os.path.join(folder, f'best_{i}')
        ea1_path = os.path.join(base_dir, best_folder, file_names[0])
        ea2_path = os.path.join(base_dir, best_folder, file_names[1])

        ea1_best_gains.append(extract_individual_gains(ea1_path))
        ea2_best_gains.append(extract_individual_gains(ea2_path))

    # Store mean gains for both EA1 and EA2
    data[f"{folder}_EA1"] = calculate_individual_gain_mean(ea1_best_gains)
    data[f"{folder}_EA2"] = calculate_individual_gain_mean(ea2_best_gains)

# Prepare data for plotting
boxplot_data = [
    data["test_run_enemy2_EA1"], data["test_run_enemy2_EA2"],
    data["test_run_enemy4_EA1"], data["test_run_enemy4_EA2"],
    data["test_run_enemy8_EA1"], data["test_run_enemy8_EA2"]
]
labels = ["EA1_Enemy2", "EA2_Enemy2", "EA1_Enemy4", "EA2_Enemy4", "EA1_Enemy8", "EA2_Enemy8"]

# Plot the boxplot
plt.figure(figsize=(14, 6))
plt.boxplot(boxplot_data, labels=labels, patch_artist=True, showfliers=True,
            boxprops=dict(linestyle='-', linewidth=2, color='blue'),
            medianprops=dict(linestyle='-', linewidth=2.5, color='red'),
            widths=0.7)

# Perform statistical tests (Mann-Whitney U) and display p-values
comparisons = [(0, 1), (2, 3), (4, 5)]
p_values = []

for i, j in comparisons:
    _, p = mannwhitneyu(boxplot_data[i], boxplot_data[j], alternative='two-sided')
    p_values.append(p)

# Print the mean and standard deviation values for each group
print("Mean and standard deviation gain per enemy:")
for i in range(0, len(boxplot_data), 2):
    mean_ea1 = np.mean(boxplot_data[i])
    mean_ea2 = np.mean(boxplot_data[i + 1])
    std_ea1 = np.std(boxplot_data[i])
    std_ea2 = np.std(boxplot_data[i + 1])
    print(f"{labels[i]}: Mean = {mean_ea1:.3f}, Std = {std_ea1:.3f}, {labels[i + 1]}: Mean = {mean_ea2:.3f}, Std = {std_ea2:.3f}")

# Display p-values on the plot
for idx, (x1, x2) in enumerate(comparisons):
    y_max = max(max(boxplot_data[x1]), max(boxplot_data[x2])) + 10
    plt.plot([x1 + 1, x1 + 1, x2 + 1, x2 + 1], [y_max, y_max + 10, y_max + 10, y_max], lw=1.5, color='k')
    plt.text((x1 + x2 + 2) * .5, y_max + 10, f"p = {p_values[idx]:.3f}", ha='center', va='bottom', color='k')

# Add labels and formatting
plt.title("Optimized Individual Gain Comparison for EA1 and EA2 (Enemy 2, 4, and 8)", fontsize=18)
plt.ylabel("Individual Gain", fontsize=14)
plt.xlabel("Experiment Groups", fontsize=14)
plt.ylim(-100, 120)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()

shapiro_p_values = {}
for idx, dataset in enumerate(boxplot_data):
    stat, p_value = shapiro(dataset)
    label = labels[idx]
    shapiro_p_values[label] = p_value

# Display Shapiro-Wilk test results
print("Shapiro-Wilk test results (p-values):")
for label, p_value in shapiro_p_values.items():
    print(f"{label}: p = {p_value:.5f}")