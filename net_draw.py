# import matplotlib.pyplot as plt
#
# def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
#     '''
#     Draw a neural network cartoon using matplotilb.
#     '''
#     n_layers = len(layer_sizes)
#     v_spacing = (top - bottom)/float(max(layer_sizes))
#     h_spacing = (right - left)/float(len(layer_sizes) - 1)
#     # Nodes
#     for n, layer_size in enumerate(layer_sizes):
#         layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
#         for m in range(layer_size):
#             circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
#                                 color='w', ec='k', zorder=4)
#             ax.add_artist(circle)
#     # Edges
#     for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
#         layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
#         layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
#         for m in range(layer_size_a):
#             for o in range(layer_size_b):
#                 line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
#                                   [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
#                 ax.add_artist(line)
#
#     inputs = ['s0(t)', '...', 's13(t)', 's_T(t)', 'a0(t - 1)']
#     layer_top_1 = v_spacing*(layer_sizes[0] - 1)/2. + (top + bottom)/2.
#     for i, label in enumerate(inputs):
#         ax.text(left - 0.05, layer_top_1 - i*v_spacing, label, ha='right')
#
#     hidden_layer = ['h0', '...', 'h7']
#     layer_top_2 = v_spacing*(layer_sizes[1] - 1)/2. + (top + bottom)/2.
#     for i, label in enumerate(hidden_layer):
#         ax.text((h_spacing + left) - 0.05, layer_top_2 - i*v_spacing, label, ha='center')
#
#     outputs = ['a0(t + 1)', 'a1(t + 1)']
#     layer_top_3 = v_spacing*(layer_sizes[2] - 1)/2. + (top + bottom)/2.
#     for i, label in enumerate(outputs):
#         ax.text(right + 0.05, layer_top_3 - i*v_spacing, label, ha='left')
#
# fig = plt.figure(figsize=(12, 12))
# ax = fig.gca()
# ax.axis('off')
# draw_neural_net(ax, .1, .9, .1, .9, [5, 3, 2])
# plt.show()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Arrow, Arc

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            # Add recurrent arrow in hidden layer
            if n == 1:
                xc = n * h_spacing + left
                yc = layer_top - m * v_spacing
                r = v_spacing / 4.
                arrow = FancyArrowPatch((xc + r, yc), (xc + r, yc), connectionstyle="arc3,rad=.5",
                                        arrowstyle='Simple, tail_width=0.5, head_width=4, head_length=8',
                                        color='k')
                ax.add_patch(arrow)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)

    # Labels
    inputs = ['s0(t)', '...', 's13(t)', 'S_T(t)', 'a0(t)']
    layer_top_1 = v_spacing * (layer_sizes[0] - 1) / 2. + (top + bottom) / 2.
    for i, label in enumerate(inputs):
        ax.text(left - 0.05, layer_top_1 - i * v_spacing, label, ha='right')

    hidden_layer = ['h1', '...', 'h2']
    layer_top_2 = v_spacing * (layer_sizes[1] - 1) / 2. + (top + bottom) / 2.
    ax.text((h_spacing + left) - 0.05, layer_top_2 - 0.5 * v_spacing, 'RNN', ha='center')
    for i, label in enumerate(hidden_layer):
        ax.text((h_spacing + left) - 0.05, layer_top_2 - i * v_spacing, label, ha='center')

    outputs = ['s0(t + 1)', '...', 's13(t + 1)']
    layer_top_3 = v_spacing * (layer_sizes[2] - 1) / 2. + (top + bottom) / 2.
    for i, label in enumerate(outputs):
        ax.text(right + 0.05, layer_top_3 - i * v_spacing, label, ha='left')

fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, [6, 3, 3])
plt.show()


