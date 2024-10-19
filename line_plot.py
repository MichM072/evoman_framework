import os
import matplotlib.pyplot as plt
import pandas as pd

# Function to extract and process fitness data from CSV files
def process_files(file_list):
    all_max_values = []
    all_avg_values = []

    for filename in file_list:
        print(f"Processing file: {filename}")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)

        # Ensure that the required columns are present
        if 'Max' in df.columns and 'Avg' in df.columns:
            max_values = df['Max'].tolist()
            avg_values = df['Avg'].tolist()
            all_max_values.append(max_values)
            all_avg_values.append(avg_values)
        else:
            print(f"File {filename} is missing required columns.")

    # Check if we collected any data
    if not all_max_values or not all_avg_values:
        raise ValueError("No valid data found in the specified files.")

    # Calculate the average and std across all valid runs
    avg_max = [sum(values) / len(values) for values in zip(*all_max_values)]
    std_max = [pd.Series(values).std() for values in zip(*all_max_values)]

    avg_avg = [sum(values) / len(values) for values in zip(*all_avg_values)]
    std_avg = [pd.Series(values).std() for values in zip(*all_avg_values)]

    return avg_max, std_max, avg_avg, std_avg

def main():
    base_dir = "."

    ea1_files_enemy1 = [
        os.path.join(base_dir, "train_run_enemy[1, 4, 7, 6]", f"EA1_train_run{i}_enemy[1, 4, 7, 6]", "results_EA1.csv")
        for i in range(1, 11)  # Assuming 10 runs
    ]

    ea2_files_enemy1 = [
        os.path.join(base_dir, "train_run_enemy[1, 4, 7, 6]", f"EA2_train_run{i}_enemy[1, 4, 7, 6]", "results_EA2.csv")
        for i in range(1, 11)  # Assuming 10 runs
    ]

    ea1_files_enemy2 = [
        os.path.join(base_dir, "train_run_enemy[1, 8, 3, 7, 6, 5]", f"EA1_train_run{i}_enemy[1, 8, 3, 7, 6, 5]", "results_EA1.csv")
        for i in range(1, 11)  # Assuming 10 runs
    ]

    ea2_files_enemy2 = [
        os.path.join(base_dir, "train_run_enemy[1, 8, 3, 7, 6, 5]", f"EA2_train_run{i}_enemy[1, 8, 3, 7, 6, 5]", "results_EA2.csv")
        for i in range(1, 11)  # Assuming 10 runs
    ]

    # Process files for EA1 and EA2 (enemy group 1)
    avg_max_ea1, std_max_ea1, avg_avg_ea1, std_avg_ea1 = process_files(ea1_files_enemy1)
    avg_max_ea2, std_max_ea2, avg_avg_ea2, std_avg_ea2 = process_files(ea2_files_enemy1)

    # Generations (X-axis)
    generations = list(range(len(avg_max_ea1)))

    plt.figure(figsize=(12, 6))

    # Plot for EA1
    plt.plot(generations, avg_max_ea1, label='EA1 (Max) - Enemy Group 1', marker='o', color='blue')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_max_ea1, std_max_ea1)],
                     [avg + std for avg, std in zip(avg_max_ea1, std_max_ea1)],
                     color='blue', alpha=0.2)

    plt.plot(generations, avg_avg_ea1, label='EA1 (Avg) - Enemy Group 1', marker='s', color='lightblue')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_avg_ea1, std_avg_ea1)],
                     [avg + std for avg, std in zip(avg_avg_ea1, std_avg_ea1)],
                     color='lightblue', alpha=0.2)

    # Plot for EA2
    plt.plot(generations, avg_max_ea2, label='EA2 (Max) - Enemy Group 1', marker='o', color='orange')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_max_ea2, std_max_ea2)],
                     [avg + std for avg, std in zip(avg_max_ea2, std_max_ea2)],
                     color='orange', alpha=0.2)

    plt.plot(generations, avg_avg_ea2, label='EA2 (Avg) - Enemy Group 1', marker='s', color='peachpuff')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_avg_ea2, std_avg_ea2)],
                     [avg + std for avg, std in zip(avg_avg_ea2, std_avg_ea2)],
                     color='peachpuff', alpha=0.2)

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.ylim(0, 100)
    plt.title('EA1 vs EA2 Comparison for Enemy Group 1')
    plt.legend()
    plt.grid()
    plt.show()

    # Process files for EA1 and EA2 (enemy group 2)
    avg_max_ea1_2, std_max_ea1_2, avg_avg_ea1_2, std_avg_ea1_2 = process_files(ea1_files_enemy2)
    avg_max_ea2_2, std_max_ea2_2, avg_avg_ea2_2, std_avg_ea2_2 = process_files(ea2_files_enemy2)

    # Generations (X-axis)
    generations = list(range(len(avg_max_ea1_2)))  # Assuming both EA1 and EA2 have the same number of generations

    plt.figure(figsize=(12, 6))

    # Plot for EA1
    plt.plot(generations, avg_max_ea1_2, label='EA1 (Max) - Enemy Group 2', marker='o', color='blue')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_max_ea1_2, std_max_ea1_2)],
                     [avg + std for avg, std in zip(avg_max_ea1_2, std_max_ea1_2)],
                     color='blue', alpha=0.2)

    plt.plot(generations, avg_avg_ea1_2, label='EA1 (Avg) - Enemy Group 2', marker='s', color='lightblue')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_avg_ea1_2, std_avg_ea1_2)],
                     [avg + std for avg, std in zip(avg_avg_ea1_2, std_avg_ea1_2)],
                     color='lightblue', alpha=0.2)

    # Plot for EA2
    plt.plot(generations, avg_max_ea2_2, label='EA2 (Max) - Enemy Group 2', marker='o', color='orange')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_max_ea2_2, std_max_ea2_2)],
                     [avg + std for avg, std in zip(avg_max_ea2_2, std_max_ea2_2)],
                     color='orange', alpha=0.2)

    plt.plot(generations, avg_avg_ea2_2, label='EA2 (Avg) - Enemy Group 2', marker='s', color='peachpuff')
    plt.fill_between(generations,
                     [avg - std for avg, std in zip(avg_avg_ea2_2, std_avg_ea2_2)],
                     [avg + std for avg, std in zip(avg_avg_ea2_2, std_avg_ea2_2)],
                     color='peachpuff', alpha=0.2)

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('EA1 vs EA2 Comparison for Enemy Group 2')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid()
    plt.show()

main()
