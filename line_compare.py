import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV files for three models
df_tams = pd.read_csv('moving/8_15/test_data/15_TAMS_transposed_output.csv', index_col=0)
df_ms = pd.read_csv('moving/8_15/test_data/tiny_target_TAMS_transposed_output.csv', index_col=0)
df_ga = pd.read_csv('moving/8_15/test_data/bigger_target_TAMS_transposed_output.csv', index_col=0)

# Rename columns for plotting
df_tams.columns = [f'Move {i} Timestep {i}' for i in range(df_tams.shape[1])]
df_ms.columns = [f'Move {i} Timestep {i}' for i in range(df_ms.shape[1])]
df_ga.columns = [f'Move {i} Timestep {i}' for i in range(df_ga.shape[1])]

# Extract 'medium' values for each model
medium_tams = df_tams.loc['MEDIUM']
medium_ms = df_ms.loc['MEDIUM']
medium_ga = df_ga.loc['MEDIUM']

# Create line plot
plt.figure(figsize=[20, 10])
plt.plot(medium_tams, label='Normal target', color='red')
plt.plot(medium_ga, label='Bigger target', color='blue')
plt.plot(medium_ms, label='Tiny target', color='green')

# Set up the rest of the elements for the chart
plt.legend(loc='upper right')
plt.ylabel('Value')
xtick_positions = range(0, len(medium_tams), 10)
xtick_labels = [f'Move {i}' for i in xtick_positions]
plt.xticks(xtick_positions, xtick_labels, rotation=45)
plt.xlim(0, len(medium_tams) - 1)  # Adjust x-axis range
plt.title('Comparison of Medium Values among different target size')

# Save plot as comparison_graph.jpg
plt.savefig('moving/8_15/test_data/target_comparison.jpg')

# Show the plot (optional, if you only want to save the image, you can remove this line)
# plt.show()
