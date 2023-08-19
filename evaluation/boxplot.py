import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(filename):
    df = pd.read_csv(filename)
    return df['TimeStep'].values


def plot_boxplot(data, labels, ax):
    sns.boxplot(data=data, ax=ax, palette="Set2")
    ax.set_title("Realized Times for Different Models")
    ax.set_ylabel('TimeStep')
    ax.set_xticklabels(labels)


data_B = load_data('F://self//evaluation//csv_files//realize_result//MS_agent_time_counts.csv')
data_A = load_data('F://self//evaluation//csv_files//realize_result//TAMS_agent_time_counts.csv')
data_C = load_data('F://self//evaluation//csv_files//realize_result//GA_agent_time_counts.csv')

data = [data_A, data_B, data_C]
labels = ['Target-Aware Minimal Surprise', 'Minimal Surprise', 'Genetic Algorithm']

fig, ax = plt.subplots(figsize=(10, 6))
plot_boxplot(data, labels, ax)

plt.tight_layout()
plt.savefig('F://self//evaluation//figures//Time_Realize.jpg')
plt.show()
