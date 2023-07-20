import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from utils.util import Agent, Pos
from parameters.STD14 import *
import random



def random_location(p, p_next, sizeX, sizeY):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 设置画布大小

    for ax in axs:
        ax.plot([1, sizeX, sizeY, 1, 1], [1, 1, sizeX, sizeY, 0], color='black', linewidth=0.4, linestyle="dashed")
        for i in range(1, sizeY):
            ax.plot([i, i], [1, sizeY], color='black', linewidth=0.5, linestyle="dashed")
        for i in range(1, sizeX):
            ax.plot([1, sizeX], [i, i], color='black', linewidth=0.5, linestyle="dashed")
        ax.set_xlim(0, sizeX + 1)
        ax.set_ylim(0, sizeY + 1)

    data = [p, p_next]
    for idx, ax in enumerate(axs):
        for i in range(len(data[idx])):
            x = data[idx][i].coord.x + 1
            y = data[idx][i].coord.y + 1
            if data[idx][i].heading.x == 1 and data[idx][i].heading.y == 0:  # Face east
                vertices = [(x - 0.28, y - 0.28), (x - 0.28, y + 0.28), (x + 0.28, y)]
            elif data[idx][i].heading.x == -1 and data[idx][i].heading.y == 0:  # Face west
                vertices = [(x + 0.28, y - 0.28), (x + 0.28, y + 0.28), (x - 0.28, y)]
            elif data[idx][i].heading.x == 0 and data[idx][i].heading.y == 1:  # Face North
                vertices = [(x - 0.28, y - 0.28), (x + 0.28, y - 0.28), (x, y + 0.28)]
            elif data[idx][i].heading.x == 0 and data[idx][i].heading.y == -1:  # Face south
                vertices = [(x - 0.28, y + 0.28), (x + 0.28, y + 0.28), (x, y - 0.28)]

            triangle = Polygon(vertices, closed=True, edgecolor='black', facecolor='black')
            ax.add_patch(triangle)

    plt.show()
# End Location Initialisation


