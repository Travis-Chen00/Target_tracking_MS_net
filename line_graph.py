# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Read data from CSV file
# df = pd.read_csv('csv_temp_file/50_TAMS_transposed_output.csv', index_col=0)
#
# # Calculate median for every 10 time steps
# median_df = df.transpose().groupby(np.arange(len(df.columns))//10).median().transpose()
#
# # Rename columns for plotting
# median_df.columns = [f'{i}' for i in range(median_df.shape[1])]
#
# # Create line plot
# plt.figure(figsize=[14, 9])
# plt.plot(median_df.loc['LOW'], label='blue', color='#5BDAF9')
# plt.plot(median_df.loc['MEDIUM'], label='orange', color='#F56C0D')
# plt.plot(median_df.loc['HIGH'], label='light blue', color='#F71E10')
#
# # Set up the rest of the elements for the chart
# plt.legend(loc='upper right', fontsize=20)
# plt.ylabel('Number of agents', fontsize=24)
# plt.xlabel('Move', fontsize=24)
# plt.xticks(range(len(median_df.columns)), labels=median_df.columns, rotation=45, fontsize=22)
#
# # Save plot as line_graph.jpg
# plt.savefig('csv_temp_file/50_agents.pdf')

# Show the plot (optional, if you only want to save the image, you can remove this line)
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_for_file(file_num):
    file_path = f'csv_temp_file/{file_num}_TAMS_transposed_output.csv.'

    df = pd.read_csv(file_path, index_col=0)

    median_df = df.transpose().groupby(np.arange(len(df.columns)) // 10).median().transpose()

    median_df.columns = [f'{i}' for i in range(median_df.shape[1])]

    plt.figure(figsize=[20, 16])
    plt.plot(median_df.loc['LOW'], label='blue', color='#5BDAF9')
    plt.plot(median_df.loc['MEDIUM'], label='orange', color='#F56C0D')
    plt.plot(median_df.loc['HIGH'], label='light blue', color='#F71E10')

    plt.legend(loc='upper right', fontsize=30)
    plt.ylabel('Number of agents', fontsize=44)
    plt.xlabel('Move', fontsize=44)
    plt.xticks(range(len(median_df.columns)), labels=median_df.columns, rotation=45, fontsize=42)

    plt.savefig(f'csv_temp_file/{file_num}.pdf')

    # plt.show()

# 对11到20的文件号进行遍历
for num in range(11, 21):
    plot_for_file(num)
