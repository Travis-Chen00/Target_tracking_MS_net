import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(filename):
    df = pd.read_csv(filename)
    data = df.pivot("Generation", "Timestep", "Normalized_True_Count")
    return data


def plot_heatmap(data, model_name, ax, vmin, vmax):
    sns.heatmap(data, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Number of Robots in Correct Area'},
                ax=ax, vmin=vmin, vmax=vmax)
    ax.set_title(f'{model_name}')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Generation')
    ax.invert_yaxis()


data_A = load_data('F://self//evaluation//csv_files//results//MS_timestep_counts.csv')
data_B = load_data('F://self//evaluation//csv_files//results//TAMS_timestep_counts.csv')
data_C = load_data('F://self//evaluation//csv_files//results//GA_timestep_counts.csv')

vmin = min(data_A.min().min(), data_B.min().min(), data_C.min().min())
vmax = max(data_A.max().max(), data_B.max().max(), data_C.max().max())

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

plot_heatmap(data_A, "Minimal Surprise", axes[1], vmin, vmax)
plot_heatmap(data_B, "Target-Aware Minimal Surprise", axes[0], vmin, vmax)
plot_heatmap(data_C, "Genetic Algorithm", axes[2], vmin, vmax)

fig.suptitle("10 agents in correct positions")

plt.tight_layout()
fig.subplots_adjust(top=0.85)
plt.savefig('F://self//evaluation//figures//heat.jpg')

