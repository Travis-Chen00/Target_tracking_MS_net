import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(filename):
    df = pd.read_csv(filename)
    return df['TimeStep'].values


def plot_boxplot(data, labels, ax):
    sns.boxplot(data=data, ax=ax, palette="Set2")
    ax.set_title("Wasted Times for Different Models")
    ax.set_ylabel('TimeStep')
    ax.set_xticklabels(labels)


data_B = load_data('F://self//evaluation//csv_files//other_results//MS_agent_time_counts.csv')
data_A = load_data('F://self//evaluation//csv_files//other_results//TAMS_agent_time_counts.csv')
data_C = load_data('F://self//evaluation//csv_files//other_results//GA_agent_time_counts.csv')

data = [data_A, data_B, data_C]
labels = ['Target-Aware Minimal Surprise (TAMS)', 'Minimal Surprise', 'Genetic Algorithm']

fig, ax = plt.subplots(figsize=(10, 6))
plot_boxplot(data, labels, ax)

plt.tight_layout()
plt.savefig('F://self//evaluation//figures//Time_Wasted.jpg')
# plt.show()
