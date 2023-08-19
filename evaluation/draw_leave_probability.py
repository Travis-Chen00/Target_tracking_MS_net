import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(filename):
    return pd.read_csv(filename)


def plot_difference(data_ref, data_list, model_names, ax):
    bar_width = 0.2
    n_agents = len(data_ref['Agent'])
    base_index = range(n_agents)

    for idx, (data, model_name) in enumerate(zip(data_list, model_names)):
        adjusted_index = [i + idx * bar_width for i in base_index]
        difference = data['Leaving_Probability_Before'] - data_ref['Leaving_Probability_Before']
        ax.bar(adjusted_index, difference, bar_width, label=model_name + ' Difference')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Difference in Leaving Probabilities Compared to Target-Aware Minimal Surprise')
    ax.set_xlabel('Agent')
    ax.set_ylabel('Difference in Leaving Probability')
    ax.set_xticks(base_index)
    ax.set_xticklabels(data_ref['Agent'])
    ax.legend()


data_B = load_data('F://self//evaluation//csv_files//leave_prob//MS_leave_probabilities.csv')
data_A = load_data('F://self//evaluation//csv_files//leave_prob//TAMS_leave_probabilities.csv')
data_C = load_data('F://self//evaluation//csv_files//leave_prob//GA_leave_probabilities.csv')

fig, ax = plt.subplots(figsize=(10, 6))

plot_difference(data_A, [data_B, data_C], ["Minimal Surprise", "Genetic Algorithm"], ax)

plt.tight_layout()
fig.subplots_adjust(top=0.85)
plt.savefig('F://self//evaluation//figures//difference_bars.jpg')
