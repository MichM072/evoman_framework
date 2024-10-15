import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

ENEMY = "8"

# Function to read and process files
def process_files(file_pattern):
    all_max_values = []
    all_avg_values = []

    for filename in glob.glob(file_pattern):
        # Read the file and parse the data
        with open(filename, 'r') as file:
            lines = file.readlines()
            generations = []

            for line in lines:
                # Check for lines containing "Generation"
                if 'Generation' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        generation_data = parts[1].strip().split()
                        max_value = float(generation_data[1])
                        avg_value = float(generation_data[3])

                        generations.append((max_value, avg_value))

            max_values, avg_values = zip(*generations)  # Unzip into separate lists
            all_max_values.append(max_values)
            all_avg_values.append(avg_values)

    # Check if we collected any data
    if not all_max_values or not all_avg_values:
        raise ValueError("No valid data found for the specified pattern.")

    # Calculate the average and std across all valid runs
    avg_max = [sum(values) / len(values) for values in zip(*all_max_values)]
    std_max = [pd.Series(values).std() for values in zip(*all_max_values)]

    avg_avg = [sum(values) / len(values) for values in zip(*all_avg_values)]
    std_avg = [pd.Series(values).std() for values in zip(*all_avg_values)]

    return avg_max, std_max, avg_avg, std_avg


# Define the folder path and file patterns
folder_path = 'train_run_enemy' + ENEMY
ga_files_pattern = os.path.join(folder_path, 'GA_train_run*_enemy' + ENEMY + '/results_GA_train_run*_enemy' + ENEMY + '.txt')
sa_files_pattern = os.path.join(folder_path, 'GA_SA_train_run*_enemy' + ENEMY + '/results_GA_SA_train_run*_enemy' + ENEMY + '.txt')

# Process the files
try:
    avg_max_ga, std_max_ga, avg_avg_ga, std_avg_ga = process_files(ga_files_pattern)
    avg_max_sa, std_max_sa, avg_avg_sa, std_avg_sa = process_files(sa_files_pattern)

    # Generations (X-axis)
    generations = list(range(31))  # 0 to 30

    # Create plots
    plt.figure(figsize=(12, 6))

    # Plot for EA1_GA
    plt.plot(generations, avg_max_ga, label='EA1_GA (Max)', marker='o', color='blue')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_max_ga, std_max_ga)],
                     [avg + std for avg, std in zip(avg_max_ga, std_max_ga)],
                     color='blue', alpha=0.2)

    plt.plot(generations, avg_avg_ga, label='EA1_GA (Avg)', marker='s', color='lightblue')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_avg_ga, std_avg_ga)],
                     [avg + std for avg, std in zip(avg_avg_ga, std_avg_ga)],
                     color='lightblue', alpha=0.2)

    # Plot for EA2_GA_SA
    plt.plot(generations, avg_max_sa, label='EA2_GA_SA (Max)', marker='o', color='orange')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_max_sa, std_max_sa)],
                     [avg + std for avg, std in zip(avg_max_sa, std_max_sa)],
                     color='orange', alpha=0.2)

    plt.plot(generations, avg_avg_sa, label='EA2_GA_SA (Avg)', marker='s', color='peachpuff')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_avg_sa, std_avg_sa)],
                     [avg + std for avg, std in zip(avg_avg_sa, std_avg_sa)],
                     color='peachpuff', alpha=0.2)

    # Set x and y limits, labels, title, and legend
    plt.xlim(0, 30)
    plt.ylim(0, 100)
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('Fitness', fontsize=16)
    plt.title('EA1_GA vs EA2_GA_SA Enemy ' + ENEMY + ' comparison', fontsize=12)
    plt.legend(fontsize=16)
    plt.grid()

    plt.tight_layout()
    plt.show()


except ValueError as e:
    print(e)
