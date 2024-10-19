import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, shapiro

best_individuals = 10

# Folders for the experiments (each corresponding to a different enemy group)
folders = ["test_enemy[1, 4, 7, 6]", "test_enemy[1, 8, 3, 7, 6, 5]"]
file_names = ["results_test_EA1.csv", "results_test_EA2.csv"]

# Function to extract individual gains from a CSV file
def extract_individual_gains(file_path):
    individual_gains = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) > 1 and row[1].strip():  # Ensure there's a gain value
                try:
                    individual_gains.append(float(row[1]))
                except ValueError:
                    continue
    return individual_gains

# Store the extracted data
data = {}

# Loop through the folders and extract gains EA1 and EA2
for folder in folders:
    ea1_best_gains = []
    ea2_best_gains = []

    # Extract EA1 data from results_test_EA1.csv
    ea1_path = os.path.join(folder, file_names[0])  # results_test_EA1.csv
    if os.path.exists(ea1_path):
        ea1_best_gains = extract_individual_gains(ea1_path)
    else:
        print(f"File not found: {ea1_path}")

    # Extract EA2 data from results_test_EA2.csv
    ea2_path = os.path.join(folder, file_names[1])  # results_test_EA2.csv
    if os.path.exists(ea2_path):
        ea2_best_gains = extract_individual_gains(ea2_path)
    else:
        print(f"File not found: {ea2_path}")

    # Store mean gains for both EA1 and EA2
    data[f"{folder}_EA1"] = ea1_best_gains
    data[f"{folder}_EA2"] = ea2_best_gains

# Prepare data for plotting
boxplot_data = [
    data[folders[0] + "_EA1"], data[folders[0] + "_EA2"],
    data[folders[1] + "_EA1"], data[folders[1] + "_EA2"]
]
labels = ["EA1_Enemy[1, 4, 7, 6]", "EA2_Enemy[1, 4, 7, 6]",
          "EA1_Enemy[1, 8, 3, 7, 6, 5]", "EA2_Enemy[1, 8, 3, 7, 6, 5]"]

# Plot the boxplot
plt.figure(figsize=(14, 6))
plt.boxplot(boxplot_data, labels=labels, patch_artist=True, showfliers=True,
            boxprops=dict(linestyle='-', linewidth=2, color='blue'),
            medianprops=dict(linestyle='-', linewidth=2.5, color='red'),
            widths=0.7)

# Perform statistical tests (Mann-Whitney U) and display p-values
comparisons = [(0, 1), (2, 3)]
p_values = []

for i, j in comparisons:
    _, p = mannwhitneyu(boxplot_data[i], boxplot_data[j], alternative='two-sided')
    p_values.append(p)

# Print the mean and std values for each group
print("Mean and standard deviation gain per enemy:")
for i in range(0, len(boxplot_data), 2):
    mean_ea1 = np.mean(boxplot_data[i])
    mean_ea2 = np.mean(boxplot_data[i + 1])
    std_ea1 = np.std(boxplot_data[i])
    std_ea2 = np.std(boxplot_data[i + 1])
    print(
        f"{labels[i]}: Mean = {mean_ea1:.3f}, Std = {std_ea1:.3f}, {labels[i + 1]}: Mean = {mean_ea2:.3f}, Std = {std_ea2:.3f}")

# Display p-values on the plot
for idx, (x1, x2) in enumerate(comparisons):
    y_max = max(max(boxplot_data[x1]), max(boxplot_data[x2])) + 10
    plt.plot([x1 + 1, x1 + 1, x2 + 1, x2 + 1], [y_max, y_max + 10, y_max + 10, y_max], lw=1.5, color='k')
    plt.text((x1 + x2 + 2) * .5, y_max + 10, f"p = {p_values[idx]:.3f}", ha='center', va='bottom', color='k')

# Add labels and formatting
plt.title("Optimized Individual Gain Comparison for EA1 and EA2 (Enemy Groups)", fontsize=18)
plt.ylabel("Individual Gain", fontsize=14)
plt.xlabel("Experiment Groups", fontsize=14)
plt.ylim(-100, 120)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()

# Perform Shapiro-Wilk test for normality
shapiro_p_values = {}
for idx, dataset in enumerate(boxplot_data):
    stat, p_value = shapiro(dataset)
    label = labels[idx]
    shapiro_p_values[label] = p_value

# Display Shapiro-Wilk test results
print("Shapiro-Wilk test results (p-values):")
for label, p_value in shapiro_p_values.items():
    print(f"{label}: p = {p_value:.5f}")
