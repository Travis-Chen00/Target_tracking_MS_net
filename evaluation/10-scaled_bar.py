import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(filename):
    return pd.read_csv(filename)


def plot_stacked_bars(data_list, model_names, ax):
    bar_width = 0.2  # Adjusted width for visibility
    n_agents = len(data_list[0]['Agent'])
    base_index = range(n_agents)

    for idx, (data, model_name) in enumerate(zip(data_list, model_names)):
        adjusted_index = [i + idx * bar_width for i in base_index]
        probs_before = data['Leaving_Probability_Before'] * 10  # Scaling data by a factor of 10

        ax.bar(adjusted_index, probs_before, bar_width, label=model_name)

    ax.set_title('Leaving Probabilities of Agents (Scaled x10)')
    ax.set_xlabel('Agent')
    ax.set_ylabel('Leaving Probability (Scaled)')
    ax.set_xticks(base_index)
    ax.set_xticklabels(data_list[0]['Agent'])  # Assuming same agent order for all models
    ax.set_ylim(0, 1.5)  # Adjusted to account for the scaling
    ax.legend()


data_A = load_data('F://self//evaluation//csv_files//leave_prob//MS_leave_probabilities.csv')
data_B = load_data('F://self//evaluation//csv_files//leave_prob//TAMS_leave_probabilities.csv')
data_C = load_data('F://self//evaluation//csv_files//leave_prob//GA_leave_probabilities.csv')

fig, ax = plt.subplots(figsize=(10, 6))

plot_stacked_bars([data_B, data_A, data_C],
                  ["Target-Aware Minimal Surprise", "Minimal Surprise", "Genetic Algorithm"], ax)

plt.tight_layout()
fig.subplots_adjust(top=0.85)
plt.savefig('F://self//evaluation//figures//scaled_stacked_bars.jpg')
