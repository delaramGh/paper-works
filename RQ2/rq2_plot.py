import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'raw_results - Copy.csv'
data = pd.read_csv(file_path)


# Clean up the 'parameters' column
data['model'] = data['model'].str.replace("[',]", "", regex=True).str.strip()

# Plotting the data with lines connecting points within each model group
fig, ax = plt.subplots()
for label, group_data in data.groupby('model'):
    sorted_data = group_data.sort_values('human effort')  # Ensure the data is sorted by 'human effort'
    ax.plot(sorted_data['human effort'], sorted_data['accuracy'], label=label, marker='o')  # Connect points with lines

ax.set_xlabel('Human Effort')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy vs Human Effort for Different Models (with Lines)')
ax.legend(title='Model')
ax.invert_xaxis()  # Reverse the y-axis
plt.show()


