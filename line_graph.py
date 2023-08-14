import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file
df = pd.read_csv('moving/8_13/test_data/transposed_output.csv', index_col=0)

# Rename columns for plotting
df.columns = [f'Move {i//10} Timestep {i%10}' for i in range(df.shape[1])]

# Create line plot
plt.figure(figsize=[20,10])
plt.plot(df.loc['LOW'], label='LOW', color='#5BDAF9')
plt.plot(df.loc['MEDIUM'], label='MEDIUM', color='#F56C0D')
plt.plot(df.loc['HIGH'], label='HIGH', color='#F71E10')

# Set up the rest of the elements for the chart
plt.legend(loc='upper right')
plt.ylabel('Value')
plt.xticks(range(0, len(df.columns) + 1, 10), labels=[f'Move {i}' for i in range((len(df.columns)//10)+1)], rotation=45)
plt.title('One target (Grid 14 * 14) with Virtual Setting (Heat_MS)')

# Save plot as line_graph.jpg
plt.savefig('moving/8_13/test_data/line_graph.jpg')

# Show the plot (optional, if you only want to save the image, you can remove this line)
# plt.show()
