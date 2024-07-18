import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# data = pd.read_csv("D1_baseline_exp1.csv")

# # Create a boxplot for accuracy, separated by the 'effort' column
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='effort', y='accuracy', data=data)
# plt.title('Boxplot of Accuracy by Human Effort for cifar dataset, baseline_exp1')
# plt.xlabel('Human Effort')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.show()



import matplotlib.pyplot as plt


# Load the data from the provided files
file_paths = [
    'results\D2_baseline_exp1.csv',
    'results\D2_baseline_exp2.csv',
    'results\D2_baseline_VIF.csv',
    'results\D2_baseline_VAE.csv'
]

# Reading the CSV files into dataframes
dfs = [pd.read_csv(file_path) for file_path in file_paths]


# Extracting the necessary columns from each dataframe
parameter = 'F1-Score'
df1 = dfs[0][['effort', parameter]]
df2 = dfs[1][['effort', parameter]]
df3 = dfs[2][['effort', parameter]]
df4 = dfs[3][['effort', parameter]]

# Adding a source column to identify the source of each data point
df1['Method'] = 'Classifier'
df2['Method'] = 'Active Learning'
df3['Method'] = 'VIF'
df4['Method'] = 'VAE'

# Combining all data into a single dataframe
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Creating a box plot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(x='effort', y=parameter, hue='Method', data=combined_df, palette=['pink', 'orange', 'yellow', 'purple'])
ax.set_title(f'Box Plot of {parameter} vs. Effort for Different Methods')
plt.grid(True)

efforts = combined_df['effort'].unique()
for effort in efforts:
    ax.axvline(efforts.tolist().index(effort) + 0.5, linestyle='--', color='gray')


plt.show()

