# student performance
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode
from prettytable import PrettyTable


def print_available_columns_table(data):
    print("Available columns:")
    table = PrettyTable(["Column Name"])
    for column in data.columns:
        table.add_row([column])
    print(table)


# Load Datasets
data_secondary_30 = pd.read_csv(r"C:\Users\Shubham Gandhi\Desktop\Student Performance\Student Performance\Data\Secondary 30.csv")
data_secondary_50 = pd.read_csv(r"C:\Users\Shubham Gandhi\Desktop\Student Performance\Student Performance\Data\Secondary 50.csv")
data_secondary_100 = pd.read_csv(r"C:\Users\Shubham Gandhi\Desktop\Student Performance\Student Performance\Data\Secondary 100.csv")

# Middle School Datasets
data_middle_30 = pd.read_csv(r"C:\Users\Shubham Gandhi\Desktop\Student Performance\Student Performance\Data\Middle 30.csv")
data_middle_50 = pd.read_csv(r"C:\Users\Shubham Gandhi\Desktop\Student Performance\Student Performance\Data\Middle 50.csv")
data_middle_100 = pd.read_csv(r"C:\Users\Shubham Gandhi\Desktop\Student Performance\Student Performance\Data\Middle 100.csv")

# Elementary School Datasets
data_elementary_30 = pd.read_csv(r"C:\Users\Shubham Gandhi\Desktop\Student Performance\Student Performance\Data\Elementary 30.csv")
data_elementary_50 = pd.read_csv(r"C:\Users\Shubham Gandhi\Desktop\Student Performance\Student Performance\Data\Elementary 50.csv")
data_elementary_100 = pd.read_csv(r"C:\Users\Shubham Gandhi\Desktop\Student Performance\Student Performance\Data\Elementary 100.csv")

# Display available columns for each dataset in a table format
print("\nColumns in Secondary 30 dataset:")
print_available_columns_table(data_secondary_30)

print("\nColumns in Secondary 50 dataset:")
print_available_columns_table(data_secondary_50)

print("\nColumns in Secondary 100 dataset:")
print_available_columns_table(data_secondary_100)

print("\nColumns in Middle 30 dataset:")
print_available_columns_table(data_middle_30)

print("\nColumns in Middle 50 dataset:")
print_available_columns_table(data_middle_50)

print("\nColumns in Middle 100 dataset:")
print_available_columns_table(data_middle_100)

print("\nColumns in Elementary 30 dataset:")
print_available_columns_table(data_elementary_30)

print("\nColumns in Elementary 50 dataset:")
print_available_columns_table(data_elementary_50)

print("\nColumns in Elementary 100 dataset:")
print_available_columns_table(data_elementary_100)

# Concatenate Datasets
data_secondary_30['strength'] = 30
data_secondary_50['strength'] = 50
data_secondary_100['strength'] = 100

data_middle_30['strength'] = 30
data_middle_50['strength'] = 50
data_middle_100['strength'] = 100

data_elementary_30['strength'] = 30
data_elementary_50['strength'] = 50
data_elementary_100['strength'] = 100

all_data = pd.concat([data_secondary_30, data_secondary_50, data_secondary_100,
                      data_middle_30, data_middle_50, data_middle_100,
                      data_elementary_30, data_elementary_50, data_elementary_100], ignore_index=True)

# Train a Prediction Model for each dataset
models = {}
for strength, data in zip([30, 50, 100, 30, 50, 100],
                          [data_secondary_30, data_secondary_50, data_secondary_100,
                           data_middle_30, data_middle_50, data_middle_100,
                           data_elementary_30, data_elementary_50, data_elementary_100
                           ]):
    print(f"\nTraining model for dataset with strength {strength}:")
    print_available_columns_table(data)

    # Check if 'class_size' is present in the dataset
    if 'class_size' in data.columns:
        X = data[['pay_attention', 'activity_participation', 'speakup', 'contribution']]
        y = data['overall_performance']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        models[strength] = model
    else:
        print("Warning: 'class_size' column not found in the dataset. Skipping this dataset.")


def generate_pair_plot_and_stats(data, strength, models):
    pairplot_data = data[
        ['pay_attention', 'activity_participation', 'speakup', 'contribution', 'grades', 'overall_performance']]

    # Print a few rows of the DataFrame for debugging
    print("Sample of data for pair plot:")
    print(pairplot_data.head())

    sns.pairplot(pairplot_data)
    plt.show()
    # Ask the user for further actions
    while True:
        action_choice = input(
            "Do you want to:\n1. Calculate mean, median, and mode for this dataset"
            "\n2. Generate pair plot for another dataset"
            "\n3. Plot mean bar chart for selected columns"
            "\n4. Return to main menu"
            "\nEnter your choice (1-4): ")

        if action_choice == '2':
            return True  # Indicates to generate a pair plot for another dataset

        elif action_choice == '1':
            # Calculate and display summary statistics for 'overall_performance' column
            if 'overall_performance' in data.columns:
                summary_statistics(data['overall_performance'])
            else:
                print("Warning: 'overall_performance' column not found in the dataset. Skipping summary statistics.")

        elif action_choice == '3':
            plot_mean_bar_chart(data, strength, models)

        elif action_choice == '4':
            # Show all the plots before returning to the main menu
            return False  # Indicates to return to the main menu

        else:
            print("Invalid choice. Please enter a valid option (1-4).")


# noinspection PyListCreation
def plot_mean_bar_chart(data, strength, models):
    print("Select section:")
    print("1. Secondary")
    print("2. Middle")
    print("3. Elementary")

    section_choice = input("Enter your choice (1-3): ")

    if section_choice == '1':
        data_section = pd.concat([data_secondary_30, data_secondary_50, data_secondary_100], ignore_index=True)
    elif section_choice == '2':
        data_section = pd.concat([data_middle_30, data_middle_50, data_middle_100], ignore_index=True)
    elif section_choice == '3':
        data_section = pd.concat([data_elementary_30, data_elementary_50, data_elementary_100], ignore_index=True)
    else:
        print("Invalid choice. Please enter a valid option (1-3).")
        return

    # Display specified columns in a table using PrettyTable
    available_columns = ['pay_attention', 'activity_participation', 'speakup', 'contribution', 'student_interaction',
                         'grades', 'overall_performance']
    available_columns.append('all')  # Add "all" option

    available_columns_table = PrettyTable()
    available_columns_table.field_names = ["Number", "Specified Columns"]

    for i, column in enumerate(available_columns, start=1):
        available_columns_table.add_row([str(i), column])

    print("\nSpecified columns for analysis:")
    print(available_columns_table)

    # Allowable columns for selection
    allowed_columns = ['pay_attention', 'activity_participation', 'speakup', 'contribution', 'student_interaction',
                       'grades', 'overall_performance', 'all']

    # Choose only allowable columns by number
    selected_columns_numbers = input("Enter column numbers (comma-separated): ").split(',')
    selected_columns = [available_columns[int(num) - 1] for num in selected_columns_numbers if num.strip().isdigit()]

    # Verify selected columns
    if not selected_columns:
        print("No valid columns selected. Please choose from the allowed columns.")
        return

    # Verify 'class_size' column
    if 'class_size' not in data_section.columns:
        print("Error: 'class_size' column not found in the dataset. Unable to plot.")
        return

    if 'all' in selected_columns:
        # Plot mean values against class size for all specified columns in a single bar chart
        plt.figure(figsize=(15, 8))
        colors = sns.color_palette("husl", n_colors=len(available_columns[:-1]))

        for i, column in enumerate(available_columns[:-1]):
            mean_values = data_section.groupby('strength')[column].mean()

            # Manually set the width of the bars
            width = 2

            # Plot the bar chart
            plt.bar(mean_values.index + i * width - (len(available_columns[:-1]) - 1) * width / 2, mean_values,
                    width=width, color=colors[i], label=column)

            # Add labels on the bars
            for x, y in zip(mean_values.index + i * width - (len(available_columns[:-1]) - 1) * width / 2, mean_values):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)

        plt.title(f'Mean Values for All Columns in {section_choice}')
        plt.xlabel('Strength')
        plt.ylabel('Mean Value')

        # Set xticks to include all strengths
        plt.xticks(mean_values.index + 0.05, mean_values.index)

        # Add legend
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(available_columns[:-1]))

        plt.tight_layout()
        plt.show()

    else:
        # Plot mean values for selected columns against class size in a single bar chart
        plt.figure(figsize=(15, 8))
        colors = sns.color_palette("husl", n_colors=len(selected_columns))

        for i, column in enumerate(selected_columns):
            mean_values = data_section.groupby('strength')[column].mean()

            # Manually set the width of the bars
            width = 2

            # Plot the bar chart
            plt.bar(mean_values.index + i * width - (len(selected_columns) - 1) * width / 2, mean_values,
                    width=width, color=colors[i], label=column)

            # Add labels on the bars
            for x, y in zip(mean_values.index + i * width - (len(selected_columns) - 1) * width / 2, mean_values):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)

        plt.title(f'Mean Values for Selected Columns in {section_choice}')
        plt.xlabel('Strength')
        plt.ylabel('Mean Value')

        # Set xticks to include all strengths
        plt.xticks(mean_values.index + 0.05, mean_values.index)

        # Add legend
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(selected_columns))

        plt.tight_layout()
        plt.show()


def summary_statistics(overall_performance):
    # Calculate mean, median, and mode
    mean_value = overall_performance.mean()
    median_value = overall_performance.median()
    mode_result = mode(overall_performance)
    mode_value = mode_result if len(mode_result) > 0 else None

    # Print in table format using PrettyTable
    summary_table = PrettyTable(["Statistic", "Value"])
    summary_table.add_row(["Mean", mean_value])
    summary_table.add_row(["Median", median_value])
    summary_table.add_row(["Mode", mode_value])

    print("Summary Statistics:")
    print(summary_table)


def compare_options():
    print("1. Select dataset and analyze")
    print("2. Exit")


def get_valid_input(prompt, valid_options):
    while True:
        user_input = input(prompt).strip()
        if user_input in valid_options:
            return user_input
        else:
            print("Invalid input. Please try again.")


while True:
    compare_options()
    main_choice = get_valid_input("Enter your choice (1-2): ", ['1', '2'])

    if main_choice == '1':
        while True:
            print("Available options:")
            print("1. Generate pair plot and stats")
            print("2. Return to main menu")

            analysis_choice = get_valid_input("Enter your choice (1-2): ", ['1', '2'])

            # Inside the analysis_choice == '1' block:
            if analysis_choice == '1':
                print("Available datasets:")
                print("1. Secondary")
                print("2. Middle")
                print("3. Elementary")
                dataset_type = get_valid_input("Enter dataset type (1-3): ", ['1', '2', '3'])

                print("Available strengths:")
                print("1. Strength 30")
                print("2. Strength 50")
                print("3. Strength 100")
                strength_choice = get_valid_input("Enter strength choice (1-3): ", ['1', '2', '3'])

                if dataset_type == '1':
                    data = data_secondary_30 if strength_choice == '1' else data_secondary_50 if strength_choice == '2' else data_secondary_100
                    pair_plot_choice = generate_pair_plot_and_stats(data, strength_choice, models)

                elif dataset_type == '2':
                    data = data_middle_30 if strength_choice == '1' else data_middle_50 if strength_choice == '2' else data_middle_100
                    pair_plot_choice = generate_pair_plot_and_stats(data, strength_choice, models)

                elif dataset_type == '3':
                    data = data_elementary_30 if strength_choice == '1' else data_elementary_50 if strength_choice == '2' else data_elementary_100
                    pair_plot_choice = generate_pair_plot_and_stats(data, strength_choice, models)

                else:
                    print("Invalid choice. Please enter a valid option (1-3).")
                    continue

                if not pair_plot_choice:
                    break

            elif analysis_choice == '2':
                break  # Return to the main menu

            else:
                print("Invalid choice. Please enter a valid option (1-2).")

    elif main_choice == '2':
        print("Exiting the program.")
        break

    else:
        print("Invalid choice. Please enter a valid option (1-2).")
